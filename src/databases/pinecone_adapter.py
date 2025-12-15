"""Pinecone adapter for vector database benchmarking."""

import time
from typing import Any, Dict, List, Optional
from pinecone import Pinecone, ServerlessSpec

from .base import VectorDBAdapter, QueryResult, IndexStats


class PineconeAdapter(VectorDBAdapter):
    """Adapter for Pinecone vector database."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pinecone adapter.

        Config options:
            api_key: str - Pinecone API key
            environment: str - Pinecone environment (e.g., "gcp-starter")
            cloud: str - Cloud provider for serverless (default: "aws")
            region: str - Region for serverless (default: "us-east-1")
        """
        super().__init__("pinecone", config)
        self.pc: Optional[Pinecone] = None
        self.indexes: Dict[str, Any] = {}

    def connect(self) -> None:
        """Connect to Pinecone."""
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("Pinecone requires 'api_key' in config")

        self.pc = Pinecone(api_key=api_key)
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self.pc = None
        self.indexes = {}
        self._is_connected = False

    def create_index(
        self,
        collection_name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new index in Pinecone."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Pinecone")

        # Map metric names to Pinecone format
        metric_map = {
            "cosine": "cosine",
            "l2": "euclidean",
            "euclidean": "euclidean",
            "ip": "dotproduct",
            "inner_product": "dotproduct",
            "dot": "dotproduct"
        }
        pinecone_metric = metric_map.get(metric.lower(), "cosine")

        # Delete if exists
        try:
            if collection_name in self.pc.list_indexes().names():
                self.pc.delete_index(collection_name)
                time.sleep(5)  # Wait for deletion
        except Exception:
            pass

        # Create serverless index (free tier compatible)
        cloud = self.config.get("cloud", "aws")
        region = self.config.get("region", "us-east-1")

        self.pc.create_index(
            name=collection_name,
            dimension=dimensions,
            metric=pinecone_metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )

        # Wait for index to be ready
        while not self.pc.describe_index(collection_name).status['ready']:
            time.sleep(1)

        self.indexes[collection_name] = self.pc.Index(collection_name)

    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Insert vectors into the index."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Pinecone")

        if collection_name not in self.indexes:
            self.indexes[collection_name] = self.pc.Index(collection_name)

        index = self.indexes[collection_name]

        batch_size = 100  # Pinecone recommends smaller batches
        start_time = time.perf_counter()

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size] if metadata else [{}] * len(batch_ids)

            # Prepare vectors for upsert
            vectors_to_upsert = [
                {
                    "id": doc_id,
                    "values": vec,
                    "metadata": meta
                }
                for doc_id, vec, meta in zip(batch_ids, batch_vectors, batch_metadata)
            ]

            index.upsert(vectors=vectors_to_upsert)

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
            raise RuntimeError("Not connected to Pinecone")

        if collection_name not in self.indexes:
            self.indexes[collection_name] = self.pc.Index(collection_name)

        index = self.indexes[collection_name]

        # Convert filter to Pinecone format
        pinecone_filter = self._convert_filter(filter) if filter else None

        start_time = time.perf_counter()

        results = index.query(
            vector=query_vector,
            top_k=top_k,
            filter=pinecone_filter,
            include_metadata=True
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        ids = [match.id for match in results.matches]
        scores = [match.score for match in results.matches]
        metadatas = [match.metadata for match in results.matches]

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

        Note: Pinecone supports sparse-dense vectors for hybrid search
        with their paid tiers. Falls back to dense for free tier.
        """
        # Pinecone hybrid search requires sparse vectors to be indexed
        # For now, fall back to dense search
        return self.search(collection_name, query_vector, top_k)

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get statistics about the index."""
        if not self._is_connected:
            raise RuntimeError("Not connected to Pinecone")

        if collection_name not in self.indexes:
            self.indexes[collection_name] = self.pc.Index(collection_name)

        index = self.indexes[collection_name]
        stats = index.describe_index_stats()

        return IndexStats(
            num_vectors=stats.total_vector_count,
            dimensions=stats.dimension,
            index_size_bytes=0,  # Not available
            build_time_seconds=0,
            memory_usage_bytes=0
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete an index."""
        if self.pc:
            try:
                self.pc.delete_index(collection_name)
            except Exception:
                pass
            self.indexes.pop(collection_name, None)

    def _convert_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generic filter format to Pinecone filter."""
        # Pinecone uses {"field": {"$eq": value}} format
        pinecone_filter = {}

        for key, value in filter.items():
            if isinstance(value, dict):
                pinecone_filter[key] = value
            else:
                pinecone_filter[key] = {"$eq": value}

        return pinecone_filter

    def filtered_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter: Dict[str, Any],
        top_k: int = 10
    ) -> QueryResult:
        """Search with metadata filtering."""
        return self.search(collection_name, query_vector, top_k, filter=filter)
