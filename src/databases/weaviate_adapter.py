"""Weaviate adapter for vector database benchmarking."""

import time
from typing import Any, Dict, List, Optional
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery

from .base import VectorDBAdapter, QueryResult, IndexStats


class WeaviateAdapter(VectorDBAdapter):
    """Adapter for Weaviate vector database."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Weaviate adapter.

        Config options:
            host: str - Weaviate server host
            port: int - Weaviate server port (default: 8080)
            grpc_port: int - Weaviate gRPC port (default: 50051)
            api_key: str - API key for Weaviate Cloud
            url: str - Full URL for Weaviate Cloud
            deployment: str - "local", "docker", or "cloud"
        """
        super().__init__("weaviate", config)
        self.client: Optional[weaviate.Client] = None

    def connect(self) -> None:
        """Connect to Weaviate."""
        deployment = self.config.get("deployment", "docker")

        if deployment == "cloud":
            url = self.config.get("url")
            api_key = self.config.get("api_key")
            if not url:
                raise ValueError("Weaviate Cloud requires 'url' in config")

            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(api_key) if api_key else None
            )
        elif deployment in ("local", "docker"):
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 8080)
            grpc_port = self.config.get("grpc_port", 50051)

            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port
            )
        else:
            raise ValueError(f"Unknown deployment type: {deployment}")

        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self.client:
            self.client.close()
        self.client = None
        self._is_connected = False

    def create_index(
        self,
        collection_name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new collection in Weaviate."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Weaviate")

        # Map metric names to Weaviate format
        metric_map = {
            "cosine": "cosine",
            "l2": "l2-squared",
            "euclidean": "l2-squared",
            "ip": "dot",
            "inner_product": "dot",
            "dot": "dot"
        }
        weaviate_metric = metric_map.get(metric.lower(), "cosine")

        # Delete if exists
        try:
            self.client.collections.delete(collection_name)
        except Exception:
            pass

        # HNSW configuration
        ef_construction = kwargs.get("ef_construction", 128)
        max_connections = kwargs.get("m", 16)
        ef = kwargs.get("ef", 64)

        # Create collection
        self.client.collections.create(
            name=collection_name,
            vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=weaviate_metric,
                ef_construction=ef_construction,
                max_connections=max_connections,
                ef=ef
            ),
            properties=[
                Property(name="doc_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ]
        )

    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Insert vectors into the collection."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Weaviate")

        collection = self.client.collections.get(collection_name)

        batch_size = 100
        start_time = time.perf_counter()

        with collection.batch.dynamic() as batch:
            for i, (doc_id, vector) in enumerate(zip(ids, vectors)):
                meta = metadata[i] if metadata else {}

                properties = {
                    "doc_id": doc_id,
                    **{k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
                }

                batch.add_object(
                    properties=properties,
                    vector=vector,
                    uuid=weaviate.util.generate_uuid5(doc_id)
                )

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
            raise RuntimeError("Not connected to Weaviate")

        collection = self.client.collections.get(collection_name)

        # Build filter
        weaviate_filter = self._build_filter(filter) if filter else None

        start_time = time.perf_counter()

        if weaviate_filter:
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=weaviate_filter,
                return_metadata=MetadataQuery(distance=True)
            )
        else:
            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True)
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        ids = []
        scores = []
        metadatas = []

        for obj in results.objects:
            ids.append(obj.properties.get("doc_id", str(obj.uuid)))
            # Weaviate returns distance, convert to similarity for cosine
            distance = obj.metadata.distance if obj.metadata else 0
            scores.append(1 - distance)  # Cosine similarity = 1 - cosine distance
            metadatas.append(obj.properties)

        return QueryResult(
            ids=ids,
            scores=scores,
            latency_ms=latency_ms,
            metadata=metadatas
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
        Hybrid search combining dense and sparse (BM25) retrieval.

        Weaviate natively supports hybrid search.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to Weaviate")

        collection = self.client.collections.get(collection_name)

        start_time = time.perf_counter()

        results = collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,  # 0 = pure BM25, 1 = pure vector
            limit=top_k,
            return_metadata=MetadataQuery(distance=True, score=True)
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        ids = []
        scores = []
        metadatas = []

        for obj in results.objects:
            ids.append(obj.properties.get("doc_id", str(obj.uuid)))
            scores.append(obj.metadata.score if obj.metadata else 0)
            metadatas.append(obj.properties)

        return QueryResult(
            ids=ids,
            scores=scores,
            latency_ms=latency_ms,
            metadata=metadatas
        )

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get statistics about the collection."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Weaviate")

        collection = self.client.collections.get(collection_name)

        # Get object count
        response = collection.aggregate.over_all(total_count=True)
        num_vectors = response.total_count

        return IndexStats(
            num_vectors=num_vectors,
            dimensions=0,  # Not easily available
            index_size_bytes=0,
            build_time_seconds=0,
            memory_usage_bytes=0
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete a collection."""
        if self.client:
            try:
                self.client.collections.delete(collection_name)
            except Exception:
                pass

    def _build_filter(self, filter: Dict[str, Any]):
        """Build Weaviate filter from dict."""
        from weaviate.classes.query import Filter

        conditions = []

        for key, value in filter.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "$eq":
                        conditions.append(Filter.by_property(key).equal(val))
                    elif op == "$ne":
                        conditions.append(Filter.by_property(key).not_equal(val))
                    elif op == "$gt":
                        conditions.append(Filter.by_property(key).greater_than(val))
                    elif op == "$gte":
                        conditions.append(Filter.by_property(key).greater_or_equal(val))
                    elif op == "$lt":
                        conditions.append(Filter.by_property(key).less_than(val))
                    elif op == "$lte":
                        conditions.append(Filter.by_property(key).less_or_equal(val))
                    elif op == "$like":
                        conditions.append(Filter.by_property(key).like(val))
            else:
                conditions.append(Filter.by_property(key).equal(value))

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            # Combine with AND
            result = conditions[0]
            for cond in conditions[1:]:
                result = result & cond
            return result

    def filtered_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter: Dict[str, Any],
        top_k: int = 10
    ) -> QueryResult:
        """Search with metadata filtering."""
        return self.search(collection_name, query_vector, top_k, filter=filter)
