"""
Elasticsearch Adapter - Enterprise Incumbent Baseline

Elasticsearch is the industry standard for search. Since v8.0, it supports
dense_vector fields with kNN search. This adapter benchmarks ES as the
"enterprise baseline" that practitioners are likely comparing against.

Key features:
- Mature query optimizer
- Production-grade operations (snapshots, monitoring, etc.)
- Hybrid search (BM25 + dense vectors)
"""

import time
from typing import Any, Dict, List, Optional

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
except ImportError:
    raise ImportError("elasticsearch not installed. Run: pip install elasticsearch>=8.0.0")

from .base import IndexStats, QueryResult, VectorDBAdapter


class ElasticsearchAdapter(VectorDBAdapter):
    """
    Elasticsearch adapter for dense vector search.

    Uses the kNN search API introduced in Elasticsearch 8.0.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__("elasticsearch", config)

        self.host = config.get("host", "localhost")
        self.port = config.get("port", 9200)
        self.scheme = config.get("scheme", "http")
        self.user = config.get("user", "elastic")
        self.password = config.get("password", "")

        # kNN parameters
        self.num_candidates = config.get("num_candidates", 100)
        self.similarity = config.get("similarity", "cosine")

        # Index settings
        self.shards = config.get("shards", 1)
        self.replicas = config.get("replicas", 0)

        self.client: Optional[Elasticsearch] = None
        self.build_times: Dict[str, float] = {}

    def connect(self) -> None:
        """Connect to Elasticsearch."""
        auth = (self.user, self.password) if self.password else None

        self.client = Elasticsearch(
            hosts=[{"host": self.host, "port": self.port, "scheme": self.scheme}],
            basic_auth=auth,
            verify_certs=False,  # For local testing
            request_timeout=60,
        )

        # Verify connection
        if not self.client.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {self.host}:{self.port}")

        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnect from Elasticsearch."""
        if self.client:
            self.client.close()
        self._is_connected = False

    def create_index(
        self, collection_name: str, dimensions: int, metric: str = "cosine", **kwargs
    ) -> None:
        """Create an Elasticsearch index with dense_vector field."""
        # Map metric names
        similarity_map = {"cosine": "cosine", "l2": "l2_norm", "dot": "dot_product"}
        similarity = similarity_map.get(metric, "cosine")

        # Index mapping with dense_vector
        mapping = {
            "settings": {
                "number_of_shards": self.shards,
                "number_of_replicas": self.replicas,
                "index.knn": True,
                # Optimize for search (not frequent updates)
                "refresh_interval": "-1",  # Disable during bulk load
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimensions,
                        "index": True,
                        "similarity": similarity,
                    },
                    "text": {"type": "text"},
                    "category": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": False},
                }
            },
        }

        # Delete if exists
        if self.client.indices.exists(index=collection_name):
            self.client.indices.delete(index=collection_name)

        # Create index
        self.client.indices.create(index=collection_name, body=mapping)

    def insert_vectors(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> float:
        """
        Bulk insert vectors into Elasticsearch.

        NOTE: Does NOT force merge after each batch - call finalize_index()
        after all insertions are complete for optimal search performance.
        """
        start_time = time.perf_counter()

        # Prepare bulk actions
        actions = []
        for i, (doc_id, vector) in enumerate(zip(ids, vectors)):
            doc = {
                "_index": collection_name,
                "_id": doc_id,
                "_source": {
                    "id": doc_id,
                    "embedding": vector,
                    "category": metadata[i].get("category") if metadata else "default",
                },
            }
            actions.append(doc)

        # Bulk insert with optimizations
        success, failed = bulk(
            self.client, actions, chunk_size=1000, request_timeout=300, raise_on_error=False
        )

        elapsed = time.perf_counter() - start_time
        self.build_times[collection_name] = self.build_times.get(collection_name, 0) + elapsed

        return elapsed

    def finalize_index(self, collection_name: str) -> float:
        """
        Finalize index after all insertions are complete.

        Call this ONCE after all insert_vectors calls to:
        1. Re-enable refresh
        2. Force refresh to make documents searchable
        3. Force merge for optimal search performance

        Returns time taken for finalization.
        """
        start_time = time.perf_counter()

        # Re-enable refresh
        self.client.indices.put_settings(index=collection_name, body={"refresh_interval": "1s"})

        # Force refresh to make all documents searchable
        self.client.indices.refresh(index=collection_name)

        # Force merge for optimal search performance
        # This is expensive - only call once after all data is loaded
        self.client.indices.forcemerge(index=collection_name, max_num_segments=1)

        elapsed = time.perf_counter() - start_time
        return elapsed

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        kNN search with optional filtering.

        Elasticsearch supports pre-filtering with kNN, which can be
        more efficient than post-filtering for selective queries.
        """
        start_time = time.perf_counter()

        # Build kNN query
        knn_query = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": self.num_candidates,
        }

        # Add filter if provided
        if filter:
            es_filter = self._convert_filter(filter)
            knn_query["filter"] = es_filter

        # Execute search
        response = self.client.search(
            index=collection_name, knn=knn_query, size=top_k, _source=["id"]
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        ids = []
        scores = []
        for hit in response["hits"]["hits"]:
            ids.append(hit["_source"]["id"])
            scores.append(hit["_score"])

        return QueryResult(ids=ids, scores=scores, latency_ms=latency_ms)

    def hybrid_search(
        self,
        collection_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> QueryResult:
        """
        Hybrid search combining BM25 text search with kNN vector search.

        Uses Reciprocal Rank Fusion (RRF) to combine results.
        """
        start_time = time.perf_counter()

        # Hybrid query with RRF
        query = {
            "size": top_k,
            "query": {
                "bool": {"should": [{"match": {"text": {"query": query_text, "boost": 1 - alpha}}}]}
            },
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": self.num_candidates,
                "boost": alpha,
            },
        }

        response = self.client.search(index=collection_name, body=query)
        latency_ms = (time.perf_counter() - start_time) * 1000

        ids = [hit["_source"]["id"] for hit in response["hits"]["hits"]]
        scores = [hit["_score"] for hit in response["hits"]["hits"]]

        return QueryResult(ids=ids, scores=scores, latency_ms=latency_ms)

    def get_index_stats(self, collection_name: str) -> IndexStats:
        """Get index statistics."""
        stats = self.client.indices.stats(index=collection_name)
        index_stats = stats["indices"][collection_name]["primaries"]

        return IndexStats(
            num_vectors=index_stats["docs"]["count"],
            dimensions=0,  # ES doesn't expose this directly
            index_size_bytes=index_stats["store"]["size_in_bytes"],
            build_time_seconds=self.build_times.get(collection_name, 0),
            memory_usage_bytes=index_stats.get("segments", {}).get("memory_in_bytes", 0),
        )

    def delete_index(self, collection_name: str) -> None:
        """Delete an index."""
        if self.client.indices.exists(index=collection_name):
            self.client.indices.delete(index=collection_name)

    def _convert_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generic filter format to Elasticsearch DSL."""
        es_filter = {"bool": {"must": []}}

        for field, condition in filter_dict.items():
            if isinstance(condition, dict):
                if "$eq" in condition:
                    es_filter["bool"]["must"].append({"term": {field: condition["$eq"]}})
                elif "$in" in condition:
                    es_filter["bool"]["must"].append({"terms": {field: condition["$in"]}})
                elif "$gte" in condition or "$lte" in condition:
                    range_query = {}
                    if "$gte" in condition:
                        range_query["gte"] = condition["$gte"]
                    if "$lte" in condition:
                        range_query["lte"] = condition["$lte"]
                    es_filter["bool"]["must"].append({"range": {field: range_query}})
            else:
                # Simple equality
                es_filter["bool"]["must"].append({"term": {field: condition}})

        return es_filter

    # =========================================================================
    # Elasticsearch-specific methods for analysis
    # =========================================================================

    def get_query_profile(
        self, collection_name: str, query_vector: List[float], top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get query execution profile for root cause analysis.

        Similar to EXPLAIN ANALYZE in SQL databases.
        """
        response = self.client.search(
            index=collection_name,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": self.num_candidates,
            },
            size=top_k,
            profile=True,
        )

        return {
            "took_ms": response["took"],
            "profile": response.get("profile", {}),
            "shards": response["_shards"],
        }

    def get_cluster_health(self) -> Dict[str, Any]:
        """Get cluster health for operational metrics."""
        return dict(self.client.cluster.health())

    def get_node_stats(self) -> Dict[str, Any]:
        """Get node statistics for resource monitoring."""
        return dict(self.client.nodes.stats())
