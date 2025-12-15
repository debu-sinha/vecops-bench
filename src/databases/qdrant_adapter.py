"""Qdrant adapter for vector database benchmarking."""

import time
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    HnswConfigDiff,
)

from .base import VectorDBAdapter, QueryResult, IndexStats


class QdrantAdapter(VectorDBAdapter):
    """Adapter for Qdrant vector database."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Qdrant adapter.

        Config options:
            host: str - Qdrant server host
            port: int - Qdrant server port (default: 6333)
            grpc_port: int - Qdrant gRPC port (default: 6334)
            api_key: str - API key for Qdrant Cloud
            url: str - Full URL for Qdrant Cloud
            deployment: str - "local", "docker", or "cloud"
            prefer_grpc: bool - Use gRPC for better performance
        """
        super().__init__("qdrant", config)
        self.client: Optional[QdrantClient] = None

    def connect(self) -> None:
        """Connect to Qdrant."""
        deployment = self.config.get("deployment", "docker")

        if deployment == "cloud":
            url = self.config.get("url")
            api_key = self.config.get("api_key")
            if not url:
                raise ValueError("Qdrant Cloud requires 'url' in config")
            self.client = QdrantClient(url=url, api_key=api_key)
        elif deployment in ("local", "docker"):
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 6333)
            grpc_port = self.config.get("grpc_port", 6334)
            prefer_grpc = self.config.get("prefer_grpc", True)

            self.client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc
            )
        elif deployment == "memory":
            # In-memory mode for testing
            self.client = QdrantClient(":memory:")
        else:
            raise ValueError(f"Unknown deployment type: {deployment}")

        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Qdrant."""
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
        """Create a new collection in Qdrant."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Qdrant")

        # Map metric names to Qdrant format
        metric_map = {
            "cosine": Distance.COSINE,
            "l2": Distance.EUCLID,
            "euclidean": Distance.EUCLID,
            "ip": Distance.DOT,
            "inner_product": Distance.DOT,
            "dot": Distance.DOT
        }
        qdrant_metric = metric_map.get(metric.lower(), Distance.COSINE)

        # HNSW configuration
        hnsw_config = HnswConfigDiff(
            m=kwargs.get("m", 16),
            ef_construct=kwargs.get("ef_construct", 100),
            full_scan_threshold=kwargs.get("full_scan_threshold", 10000)
        )

        # Delete if exists
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=qdrant_metric
            ),
            hnsw_config=hnsw_config,
            on_disk_payload=kwargs.get("on_disk_payload", False)
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
            raise RuntimeError("Not connected to Qdrant")

        batch_size = 1000
        start_time = time.perf_counter()

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size] if metadata else [{}] * len(batch_ids)

            points = [
                PointStruct(
                    id=idx,  # Qdrant prefers integer IDs
                    vector=vec,
                    payload={**meta, "_original_id": doc_id}
                )
                for idx, (doc_id, vec, meta) in enumerate(
                    zip(batch_ids, batch_vectors, batch_metadata),
                    start=i
                )
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
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
            raise RuntimeError("Not connected to Qdrant")

        # Convert filter to Qdrant format
        qdrant_filter = self._convert_filter(filter) if filter else None

        search_params = SearchParams(
            hnsw_ef=128,  # Higher ef for better recall
            exact=False
        )

        start_time = time.perf_counter()

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            search_params=search_params,
            with_payload=True
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        ids = [hit.payload.get("_original_id", str(hit.id)) for hit in results]
        scores = [hit.score for hit in results]
        metadatas = [
            {k: v for k, v in hit.payload.items() if k != "_original_id"}
            for hit in results
        ]

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
        Hybrid search combining dense and sparse retrieval.

        Note: Requires Qdrant with sparse vectors enabled.
        Falls back to dense search if not available.
        """
        # Basic implementation - Qdrant supports sparse vectors
        # but requires additional setup. Fall back to dense for now.
        return self.search(collection_name, query_vector, top_k)

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get statistics about the collection."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Qdrant")

        info = self.client.get_collection(collection_name)

        return IndexStats(
            num_vectors=info.points_count,
            dimensions=info.config.params.vectors.size,
            index_size_bytes=0,  # Not directly available
            build_time_seconds=0,
            memory_usage_bytes=0
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete a collection."""
        if self.client:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

    def _convert_filter(self, filter: Dict[str, Any]) -> Filter:
        """Convert generic filter format to Qdrant Filter."""
        conditions = []

        for key, value in filter.items():
            if isinstance(value, dict):
                # Handle operators like {"$gt": 5}
                for op, val in value.items():
                    if op == "$eq":
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=val))
                        )
                    elif op == "$gt":
                        conditions.append(
                            FieldCondition(key=key, range=models.Range(gt=val))
                        )
                    elif op == "$gte":
                        conditions.append(
                            FieldCondition(key=key, range=models.Range(gte=val))
                        )
                    elif op == "$lt":
                        conditions.append(
                            FieldCondition(key=key, range=models.Range(lt=val))
                        )
                    elif op == "$lte":
                        conditions.append(
                            FieldCondition(key=key, range=models.Range(lte=val))
                        )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=conditions) if conditions else None

    def filtered_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter: Dict[str, Any],
        top_k: int = 10
    ) -> QueryResult:
        """Search with metadata filtering."""
        return self.search(collection_name, query_vector, top_k, filter=filter)
