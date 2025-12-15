"""ChromaDB adapter for vector database benchmarking."""

import time
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .base import IndexStats, QueryResult, VectorDBAdapter


class ChromaAdapter(VectorDBAdapter):
    """Adapter for ChromaDB vector database."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ChromaDB adapter.

        Config options:
            persist_directory: str - Directory for persistent storage (optional)
            host: str - Chroma server host (for client/server mode)
            port: int - Chroma server port
            deployment: str - "local" (in-process) or "server" (client/server)
        """
        super().__init__("chroma", config)
        self.client: Optional[chromadb.Client] = None
        self.collections: Dict[str, chromadb.Collection] = {}

    def connect(self) -> None:
        """Connect to ChromaDB."""
        deployment = self.config.get("deployment", "local")

        if deployment == "local":
            persist_dir = self.config.get("persist_directory")
            if persist_dir:
                self.client = chromadb.PersistentClient(path=persist_dir)
            else:
                self.client = chromadb.Client()
        elif deployment == "server":
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 8000)
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            raise ValueError(f"Unknown deployment type: {deployment}")

        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        self.client = None
        self.collections = {}
        self._is_connected = False

    def create_index(
        self, collection_name: str, dimensions: int, metric: str = "cosine", **kwargs
    ) -> None:
        """Create a new collection in ChromaDB."""
        if not self._is_connected:
            raise RuntimeError("Not connected to ChromaDB")

        # Map metric names to ChromaDB format
        metric_map = {
            "cosine": "cosine",
            "l2": "l2",
            "euclidean": "l2",
            "ip": "ip",
            "inner_product": "ip",
        }
        chroma_metric = metric_map.get(metric.lower(), "cosine")

        # Delete if exists
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.collections[collection_name] = self.client.create_collection(
            name=collection_name, metadata={"hnsw:space": chroma_metric}
        )

    def _get_collection(self, collection_name: str) -> chromadb.Collection:
        """Get collection, reconnecting if necessary."""
        if collection_name not in self.collections:
            # Try to get existing collection from server
            try:
                self.collections[collection_name] = self.client.get_collection(collection_name)
            except Exception:
                raise ValueError(f"Collection {collection_name} does not exist")
        return self.collections[collection_name]

    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Insert vectors into the collection."""
        collection = self._get_collection(collection_name)

        # ChromaDB has a batch size limit - use smaller batches for HTTP mode
        batch_size = 500  # Reduced for HTTP server compatibility
        start_time = time.perf_counter()

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size] if metadata else None

            collection.add(ids=batch_ids, embeddings=batch_vectors, metadatas=batch_metadata)

        return time.perf_counter() - start_time

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """Search for similar vectors."""
        collection = self._get_collection(collection_name)

        # Convert filter to ChromaDB format
        where_filter = None
        if filter:
            where_filter = self._convert_filter(filter)

        start_time = time.perf_counter()

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["distances", "metadatas"],
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else None

        # Convert distances to similarity scores (ChromaDB returns distances)
        # For cosine, distance = 1 - similarity
        scores = [1 - d for d in distances]

        return QueryResult(ids=ids, scores=scores, latency_ms=latency_ms, metadata=metadatas)

    def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> QueryResult:
        """
        Hybrid search combining dense and sparse retrieval.

        Note: ChromaDB doesn't natively support hybrid search.
        This implementation uses dense-only search.
        """
        # ChromaDB doesn't support native hybrid search
        # Fall back to dense search only
        return self.search(collection_name, query_vector, top_k)

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get statistics about the collection."""
        collection = self._get_collection(collection_name)
        count = collection.count()

        # ChromaDB doesn't expose detailed stats
        return IndexStats(
            num_vectors=count,
            dimensions=0,  # Not available
            index_size_bytes=0,  # Not available
            build_time_seconds=0,
            memory_usage_bytes=0,
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete a collection."""
        if self.client:
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            self.collections.pop(collection_name, None)

    def _convert_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generic filter format to ChromaDB where clause."""
        # Simple pass-through for now - extend as needed
        # ChromaDB uses {"field": {"$eq": value}} format
        chroma_filter = {}
        for key, value in filter.items():
            if isinstance(value, dict):
                chroma_filter[key] = value
            else:
                chroma_filter[key] = {"$eq": value}
        return chroma_filter

    def filtered_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter: Dict[str, Any],
        top_k: int = 10,
    ) -> QueryResult:
        """Search with metadata filtering."""
        return self.search(collection_name, query_vector, top_k, filter=filter)
