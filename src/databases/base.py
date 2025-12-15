"""Base class for vector database adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time


@dataclass
class QueryResult:
    """Result from a vector search query."""
    ids: List[str]
    scores: List[float]
    latency_ms: float
    metadata: Optional[List[Dict[str, Any]]] = None


@dataclass
class IndexStats:
    """Statistics about a vector index."""
    num_vectors: int
    dimensions: int
    index_size_bytes: int
    build_time_seconds: float
    memory_usage_bytes: int


class VectorDBAdapter(ABC):
    """Abstract base class for vector database adapters."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._is_connected = False

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the database."""
        pass

    @abstractmethod
    def create_index(
        self,
        collection_name: str,
        dimensions: int,
        metric: str = "cosine",
        **kwargs
    ) -> None:
        """Create a new vector index/collection."""
        pass

    @abstractmethod
    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        """Insert vectors into the index. Returns time taken in seconds."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> QueryResult:
        """Hybrid search combining dense and sparse retrieval."""
        pass

    @abstractmethod
    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get statistics about the index."""
        pass

    @abstractmethod
    def delete_index(self, collection_name: str) -> None:
        """Delete an index/collection."""
        pass

    def measure_cold_start(self, collection_name: str, query_vector: List[float]) -> float:
        """Measure cold-start latency (disconnect, reconnect, first query)."""
        self.disconnect()

        start_time = time.perf_counter()
        self.connect()
        result = self.search(collection_name, query_vector, top_k=10)
        cold_start_time = time.perf_counter() - start_time

        return cold_start_time * 1000  # Return in milliseconds

    def benchmark_qps(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        duration_seconds: float = 10.0
    ) -> Tuple[float, List[float]]:
        """
        Benchmark queries per second.

        Returns:
            Tuple of (QPS, list of latencies in ms)
        """
        latencies = []
        query_count = 0
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_seconds:
            query_vector = query_vectors[query_count % len(query_vectors)]
            result = self.search(collection_name, query_vector, top_k=top_k)
            latencies.append(result.latency_ms)
            query_count += 1

        elapsed = time.perf_counter() - start_time
        qps = query_count / elapsed

        return qps, latencies
