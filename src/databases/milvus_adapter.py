"""Milvus adapter for vector database benchmarking."""

import time
from typing import Any, Dict, List, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from .base import VectorDBAdapter, QueryResult, IndexStats


class MilvusAdapter(VectorDBAdapter):
    """Adapter for Milvus vector database."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Milvus adapter.

        Config options:
            host: str - Milvus server host
            port: int - Milvus server port (default: 19530)
            user: str - Username (optional)
            password: str - Password (optional)
            alias: str - Connection alias (default: "default")
        """
        super().__init__("milvus", config)
        self.alias = config.get("alias", "default")
        self.collections: Dict[str, Collection] = {}

    def connect(self) -> None:
        """Connect to Milvus."""
        connections.connect(
            alias=self.alias,
            host=self.config.get("host", "localhost"),
            port=self.config.get("port", 19530),
            user=self.config.get("user", ""),
            password=self.config.get("password", "")
        )
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        connections.disconnect(self.alias)
        self.collections = {}
        self._is_connected = False

    def create_index(
        self,
        collection_name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new collection in Milvus."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Milvus")

        # Map metric names to Milvus format
        metric_map = {
            "cosine": "COSINE",
            "l2": "L2",
            "euclidean": "L2",
            "ip": "IP",
            "inner_product": "IP"
        }
        milvus_metric = metric_map.get(metric.lower(), "COSINE")

        # Drop if exists
        if utility.has_collection(collection_name, using=self.alias):
            utility.drop_collection(collection_name, using=self.alias)

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
        ]

        # Add metadata fields if specified
        if kwargs.get("metadata_fields"):
            for field_name, field_type in kwargs["metadata_fields"].items():
                if field_type == "string":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.VARCHAR, max_length=1024))
                elif field_type == "int":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.INT64))
                elif field_type == "float":
                    fields.append(FieldSchema(name=field_name, dtype=DataType.FLOAT))

        schema = CollectionSchema(
            fields=fields,
            description=f"VectorDB-Bench collection: {collection_name}"
        )

        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.alias
        )

        # Create index
        index_type = kwargs.get("index_type", "HNSW")
        index_params = {
            "metric_type": milvus_metric,
            "index_type": index_type,
        }

        if index_type == "HNSW":
            index_params["params"] = {
                "M": kwargs.get("m", 16),
                "efConstruction": kwargs.get("ef_construction", 200)
            }
        elif index_type == "IVF_FLAT":
            index_params["params"] = {
                "nlist": kwargs.get("nlist", 1024)
            }
        elif index_type == "IVF_SQ8":
            index_params["params"] = {
                "nlist": kwargs.get("nlist", 1024)
            }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        # Also create index on doc_id for filtering
        collection.create_index(
            field_name="doc_id",
            index_params={"index_type": "Trie"}
        )

        self.collections[collection_name] = collection

    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Insert vectors into the collection."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Milvus")

        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name, using=self.alias)

        collection = self.collections[collection_name]

        batch_size = 1000
        start_time = time.perf_counter()

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]

            data = [
                batch_ids,      # doc_id
                batch_vectors,  # embedding
            ]

            collection.insert(data)

        # Flush to ensure data is persisted
        collection.flush()

        return time.perf_counter() - start_time

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Search for similar vectors."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Milvus")

        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name, using=self.alias)

        collection = self.collections[collection_name]

        # Load collection into memory
        collection.load()

        # Build filter expression
        expr = self._build_filter_expression(filter) if filter else None

        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128}  # HNSW search parameter
        }

        start_time = time.perf_counter()

        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["doc_id"]
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        ids = []
        scores = []

        if results and len(results) > 0:
            for hit in results[0]:
                ids.append(hit.entity.get("doc_id"))
                scores.append(hit.score)

        return QueryResult(
            ids=ids,
            scores=scores,
            latency_ms=latency_ms,
            metadata=None
        )

    def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> QueryResult:
        """
        Hybrid search combining dense and sparse retrieval.

        Note: Milvus 2.4+ supports sparse vectors for hybrid search.
        This is a basic implementation.
        """
        # Milvus supports hybrid search with sparse vectors
        # For now, fall back to dense search
        return self.search(collection_name, query_vector, top_k)

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get statistics about the collection."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Milvus")

        if collection_name not in self.collections:
            self.collections[collection_name] = Collection(collection_name, using=self.alias)

        collection = self.collections[collection_name]

        # Get collection stats
        stats = collection.num_entities

        # Get schema info
        schema = collection.schema
        dimensions = 0
        for field in schema.fields:
            if field.name == "embedding":
                dimensions = field.params.get("dim", 0)
                break

        return IndexStats(
            num_vectors=stats,
            dimensions=dimensions,
            index_size_bytes=0,  # Not directly available
            build_time_seconds=0,
            memory_usage_bytes=0
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete a collection."""
        if utility.has_collection(collection_name, using=self.alias):
            utility.drop_collection(collection_name, using=self.alias)
        self.collections.pop(collection_name, None)

    def _build_filter_expression(self, filter: Dict[str, Any]) -> str:
        """Build Milvus filter expression from dict."""
        expressions = []

        for key, value in filter.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "$eq":
                        if isinstance(val, str):
                            expressions.append(f'{key} == "{val}"')
                        else:
                            expressions.append(f'{key} == {val}')
                    elif op == "$gt":
                        expressions.append(f'{key} > {val}')
                    elif op == "$gte":
                        expressions.append(f'{key} >= {val}')
                    elif op == "$lt":
                        expressions.append(f'{key} < {val}')
                    elif op == "$lte":
                        expressions.append(f'{key} <= {val}')
                    elif op == "$ne":
                        if isinstance(val, str):
                            expressions.append(f'{key} != "{val}"')
                        else:
                            expressions.append(f'{key} != {val}')
                    elif op == "$in":
                        if isinstance(val[0], str):
                            vals = ', '.join(f'"{v}"' for v in val)
                        else:
                            vals = ', '.join(str(v) for v in val)
                        expressions.append(f'{key} in [{vals}]')
            else:
                if isinstance(value, str):
                    expressions.append(f'{key} == "{value}"')
                else:
                    expressions.append(f'{key} == {value}')

        return " and ".join(expressions) if expressions else None

    def filtered_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter: Dict[str, Any],
        top_k: int = 10
    ) -> QueryResult:
        """Search with metadata filtering."""
        return self.search(collection_name, query_vector, top_k, filter=filter)
