"""
Faiss Adapter - Speed of Light Baseline

This adapter provides the theoretical maximum performance for vector search
without database overhead. Use it to measure how much overhead each database
adds compared to raw in-memory HNSW.

Faiss is NOT a database - it's a library. This baseline shows:
1. Best possible latency (no network, no serialization)
2. Best possible QPS (pure CPU/memory bound)
3. Reference point for evaluating DB overhead
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

from .base import IndexStats, QueryResult, VectorDBAdapter


class FaissAdapter(VectorDBAdapter):
    """
    Faiss baseline adapter for measuring theoretical speed of light.

    This is an IN-MEMORY baseline. It does not persist data, does not
    handle crashes, and does not support filtering. Its purpose is to
    establish the minimum possible latency for ANN search.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("faiss", config)

        # HNSW parameters
        # NOTE: FAISS uses lower ef_construction (64) vs other DBs (200) for practical
        # build times at 10M+ scale. FAISS HNSW builds incrementally (each insert
        # searches the existing graph), making high ef_construction extremely slow.
        #
        # TRADEOFF: Lower ef_construction = faster build, slightly lower recall.
        # This is acceptable because:
        # 1. FAISS is a "speed of light" baseline, not a production database
        # 2. At ef_search=100, recall difference is minimal (<1-2%)
        # 3. Building 10M vectors with ef_construction=200 takes 20+ hours
        #
        # For fair recall comparison, we use ef_search=100 for all databases.
        self.index_type = config.get("index_type", "HNSW")
        self.M = config.get("M", 16)  # HNSW connections per node
        self.ef_construction = config.get("ef_construction", 128)  # Lower for faster build
        self.ef_search = config.get("ef_search", 100)

        # Index storage
        self.indices: Dict[str, faiss.Index] = {}
        self.id_maps: Dict[str, Dict[int, str]] = {}  # faiss int -> string id
        self.dimensions: Dict[str, int] = {}
        self.build_times: Dict[str, float] = {}
        self.next_int_id: Dict[str, int] = {}  # Track next available int ID per collection

    def connect(self) -> None:
        """No connection needed for in-memory library."""
        self._is_connected = True

    def disconnect(self) -> None:
        """Clear indices on disconnect."""
        self.indices.clear()
        self.id_maps.clear()
        self.next_int_id.clear()
        self._is_connected = False

    def create_index(
        self, collection_name: str, dimensions: int, metric: str = "cosine", **kwargs
    ) -> None:
        """Create a Faiss HNSW index."""
        self.dimensions[collection_name] = dimensions

        if self.index_type == "HNSW":
            # HNSW Flat index (no quantization for fair comparison)
            base_index = faiss.IndexHNSWFlat(dimensions, self.M)
            base_index.hnsw.efConstruction = self.ef_construction
            base_index.hnsw.efSearch = self.ef_search
        elif self.index_type == "IVF_FLAT":
            # IVF for comparison
            nlist = kwargs.get("nlist", 100)
            quantizer = faiss.IndexFlatL2(dimensions)
            base_index = faiss.IndexIVFFlat(quantizer, dimensions, nlist)
        else:
            # Default to flat (brute force - slowest but exact)
            base_index = faiss.IndexFlatL2(dimensions)

        # Always wrap with IndexIDMap to support custom IDs
        # Note: For cosine similarity, we normalize vectors in insert_vectors()
        index = faiss.IndexIDMap(base_index)

        self.indices[collection_name] = index
        self.id_maps[collection_name] = {}
        self.next_int_id[collection_name] = 0  # Initialize ID counter

    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """Insert vectors into Faiss index."""
        start_time = time.perf_counter()

        index = self.indices[collection_name]
        vectors_np = np.array(vectors, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors_np)

        # Create sequential integer IDs (Faiss requires int64)
        # CRITICAL: Use next_int_id to ensure unique IDs across batches
        start_id = self.next_int_id[collection_name]
        int_ids = np.arange(start_id, start_id + len(ids), dtype="int64")
        self.next_int_id[collection_name] = start_id + len(ids)

        # Store mapping from int ID to string ID
        for int_id, str_id in zip(int_ids, ids):
            self.id_maps[collection_name][int(int_id)] = str_id

        # Handle training for IVF (check the base index inside IndexIDMap)
        base_index = index.index if isinstance(index, faiss.IndexIDMap) else index
        if hasattr(base_index, "is_trained") and not base_index.is_trained:
            base_index.train(vectors_np)

        # Add vectors with IDs
        index.add_with_ids(vectors_np, int_ids)

        elapsed = time.perf_counter() - start_time
        self.build_times[collection_name] = self.build_times.get(collection_name, 0) + elapsed

        return elapsed

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Search for similar vectors.

        Note: Faiss does NOT support filtering. If filter is provided,
        we ignore it and log a warning. This is intentional - Faiss is
        the "speed of light" for pure ANN, not filtered search.
        """
        if filter:
            # Log but don't fail - this is a baseline
            pass  # Faiss doesn't support filtering

        index = self.indices[collection_name]
        id_map = self.id_maps[collection_name]

        # Prepare query
        query_np = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(query_np)

        # Search
        start_time = time.perf_counter()
        distances, indices = index.search(query_np, top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Convert to string IDs
        result_ids = []
        result_scores = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # Faiss returns -1 for missing results
                result_ids.append(id_map.get(int(idx), str(idx)))
                # Convert L2 distance to similarity score
                result_scores.append(float(1 / (1 + dist)))

        return QueryResult(ids=result_ids, scores=result_scores, latency_ms=latency_ms)

    def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> QueryResult:
        """Faiss doesn't support hybrid search - pure vector only."""
        # Fall back to pure vector search
        return self.search(collection_name, query_vector, top_k)

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get index statistics."""
        index = self.indices[collection_name]

        return IndexStats(
            num_vectors=index.ntotal,
            dimensions=self.dimensions.get(collection_name, 0),
            index_size_bytes=self._estimate_index_size(index),
            build_time_seconds=self.build_times.get(collection_name, 0),
            memory_usage_bytes=self._estimate_index_size(index),
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete an index."""
        if collection_name in self.indices:
            del self.indices[collection_name]
        if collection_name in self.id_maps:
            del self.id_maps[collection_name]
        if collection_name in self.dimensions:
            del self.dimensions[collection_name]
        if collection_name in self.next_int_id:
            del self.next_int_id[collection_name]

    def _estimate_index_size(self, index: faiss.Index) -> int:
        """Estimate index size in bytes."""
        # HNSW: vectors + graph structure
        # Rough estimate: vectors * dims * 4 bytes * 2 (for graph overhead)
        if hasattr(index, "ntotal") and hasattr(index, "d"):
            return int(index.ntotal * index.d * 4 * 2)
        return 0

    # =========================================================================
    # Faiss-specific methods for baseline analysis
    # =========================================================================

    def get_search_parameters(self, collection_name: str) -> Dict[str, Any]:
        """Get current search parameters."""
        index = self.indices[collection_name]
        # Handle IndexIDMap wrapper
        base_index = index.index if isinstance(index, faiss.IndexIDMap) else index

        params = {
            "index_type": self.index_type,
            "ntotal": index.ntotal,
        }

        if hasattr(base_index, "hnsw"):
            params["M"] = self.M
            params["efConstruction"] = base_index.hnsw.efConstruction
            params["efSearch"] = base_index.hnsw.efSearch

        return params

    def set_ef_search(self, collection_name: str, ef_search: int) -> None:
        """Adjust ef_search parameter for recall/speed tradeoff."""
        index = self.indices[collection_name]
        # Handle IndexIDMap wrapper
        base_index = index.index if isinstance(index, faiss.IndexIDMap) else index
        if hasattr(base_index, "hnsw"):
            base_index.hnsw.efSearch = ef_search

    def benchmark_recall_vs_speed(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        ground_truth: List[List[str]],
        ef_values: List[int] = [16, 32, 64, 128, 256, 512],
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Benchmark recall vs speed tradeoff by varying ef_search.

        This is useful to establish the Pareto frontier for HNSW.
        """
        results = []

        for ef in ef_values:
            self.set_ef_search(collection_name, ef)

            latencies = []
            recalls = []

            for query_vec, gt_ids in zip(query_vectors, ground_truth):
                result = self.search(collection_name, query_vec, top_k=10)
                latencies.append(result.latency_ms)

                # Calculate recall
                retrieved = set(result.ids[:10])
                relevant = set(gt_ids[:10])
                recall = len(retrieved & relevant) / len(relevant) if relevant else 0
                recalls.append(recall)

            results.append(
                {
                    "ef_search": ef,
                    "mean_latency_ms": np.mean(latencies),
                    "p99_latency_ms": np.percentile(latencies, 99),
                    "mean_recall": np.mean(recalls),
                    "qps": 1000 / np.mean(latencies),
                }
            )

        return {"recall_speed_tradeoff": results}
