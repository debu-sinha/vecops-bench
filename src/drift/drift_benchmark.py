"""
Drift Benchmark: Measuring retrieval degradation over time.

This is the key novel contribution - evaluating how vector database
performance degrades as the underlying corpus evolves.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .temporal_drift import (
    TemporalDriftSimulator,
    DriftPattern,
    CorpusSnapshot,
)
from ..databases.base import VectorDBAdapter, QueryResult
from ..metrics import recall_at_k, ndcg_at_k, latency_percentiles


@dataclass
class DriftBenchmarkResult:
    """Results from a drift benchmark run."""
    database: str
    dataset: str
    drift_pattern: str
    num_timestamps: int

    # Per-timestamp metrics
    timestamps: List[int]
    recall_at_10: List[float]
    recall_at_100: List[float]
    ndcg_at_10: List[float]
    latency_p50: List[float]
    latency_p95: List[float]
    corpus_size: List[int]
    survival_rate: List[float]

    # Aggregate metrics
    recall_degradation_rate: float  # Recall loss per timestamp
    recall_half_life: Optional[int]  # Timestamps until recall drops 50%
    final_recall: float
    initial_recall: float

    # Drift statistics
    drift_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "database": self.database,
            "dataset": self.dataset,
            "drift_pattern": self.drift_pattern,
            "num_timestamps": self.num_timestamps,
            "timestamps": self.timestamps,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "ndcg_at_10": self.ndcg_at_10,
            "latency_p50": self.latency_p50,
            "latency_p95": self.latency_p95,
            "corpus_size": self.corpus_size,
            "survival_rate": self.survival_rate,
            "recall_degradation_rate": self.recall_degradation_rate,
            "recall_half_life": self.recall_half_life,
            "final_recall": self.final_recall,
            "initial_recall": self.initial_recall,
            "drift_stats": self.drift_stats,
        }


class DriftBenchmark:
    """
    Benchmark that measures retrieval performance degradation
    as the corpus evolves over time.

    Key insight: Production corpora are not static. Documents are
    added, updated, and deleted. This benchmark measures how well
    vector databases maintain retrieval quality under corpus drift.
    """

    def __init__(
        self,
        adapter: VectorDBAdapter,
        collection_name: str,
        seed: int = 42
    ):
        """
        Initialize drift benchmark.

        Args:
            adapter: Connected vector database adapter
            collection_name: Name for the benchmark collection
            seed: Random seed for reproducibility
        """
        self.adapter = adapter
        self.collection_name = collection_name
        self.seed = seed

    def run(
        self,
        doc_ids: List[str],
        doc_embeddings: np.ndarray,
        query_ids: List[str],
        query_embeddings: np.ndarray,
        relevance_map: Dict[str, List[str]],  # query_id -> relevant_doc_ids
        num_timestamps: int = 10,
        drift_pattern: DriftPattern = DriftPattern.MODERATE,
        queries_per_timestamp: int = 100,
        top_k: int = 100,
        reindex_each_step: bool = True,
    ) -> DriftBenchmarkResult:
        """
        Run the drift benchmark.

        Args:
            doc_ids: Document IDs
            doc_embeddings: Document embeddings (num_docs, dim)
            query_ids: Query IDs
            query_embeddings: Query embeddings (num_queries, dim)
            relevance_map: Mapping from query_id to list of relevant doc_ids
            num_timestamps: Number of drift time steps
            drift_pattern: Type of drift to simulate
            queries_per_timestamp: Queries to evaluate at each step
            top_k: Maximum k for recall computation
            reindex_each_step: Whether to rebuild index at each step

        Returns:
            DriftBenchmarkResult with per-timestamp metrics
        """
        print(f"\nRunning drift benchmark: {drift_pattern.value} pattern, {num_timestamps} steps")

        # Initialize simulator
        simulator = TemporalDriftSimulator(
            doc_ids=doc_ids,
            embeddings=doc_embeddings,
            seed=self.seed
        )

        # Results storage
        timestamps = []
        recall_at_10_list = []
        recall_at_100_list = []
        ndcg_at_10_list = []
        latency_p50_list = []
        latency_p95_list = []
        corpus_size_list = []
        survival_rate_list = []

        # Sample queries for consistent evaluation
        query_sample_indices = np.random.RandomState(self.seed).choice(
            len(query_ids),
            size=min(queries_per_timestamp, len(query_ids)),
            replace=False
        )

        # Initial evaluation (timestamp 0)
        print("\n[T=0] Initial corpus evaluation...")
        self._index_corpus(simulator.snapshots[0])
        t0_metrics = self._evaluate_queries(
            query_ids, query_embeddings, relevance_map,
            query_sample_indices, top_k
        )

        timestamps.append(0)
        recall_at_10_list.append(t0_metrics["recall@10"])
        recall_at_100_list.append(t0_metrics["recall@100"])
        ndcg_at_10_list.append(t0_metrics["ndcg@10"])
        latency_p50_list.append(t0_metrics["latency_p50"])
        latency_p95_list.append(t0_metrics["latency_p95"])
        corpus_size_list.append(simulator.snapshots[0].size)
        survival_rate_list.append(1.0)

        initial_recall = t0_metrics["recall@10"]

        # Run drift simulation
        for t in tqdm(range(1, num_timestamps + 1), desc="Drift simulation"):
            # Advance time (apply drift)
            snapshot = simulator.advance_time(pattern=drift_pattern)

            if reindex_each_step:
                # Rebuild index with drifted corpus
                self._index_corpus(snapshot)
            else:
                # Just update the index incrementally
                self._update_index_incrementally(simulator, t)

            # Evaluate queries
            # Note: We evaluate against ORIGINAL relevance judgments
            # This measures how well we can still find originally-relevant docs
            metrics = self._evaluate_queries(
                query_ids, query_embeddings, relevance_map,
                query_sample_indices, top_k,
                current_doc_ids=set(snapshot.doc_ids)
            )

            timestamps.append(t)
            recall_at_10_list.append(metrics["recall@10"])
            recall_at_100_list.append(metrics["recall@100"])
            ndcg_at_10_list.append(metrics["ndcg@10"])
            latency_p50_list.append(metrics["latency_p50"])
            latency_p95_list.append(metrics["latency_p95"])
            corpus_size_list.append(snapshot.size)

            # Compute survival rate
            original_set = set(doc_ids)
            current_set = set(snapshot.doc_ids)
            survival = len(original_set & current_set) / len(original_set)
            survival_rate_list.append(survival)

        # Compute aggregate metrics
        final_recall = recall_at_10_list[-1]
        recall_degradation_rate = (initial_recall - final_recall) / num_timestamps

        # Compute recall half-life (timestamps until recall drops to 50% of initial)
        recall_half_life = None
        target_recall = initial_recall * 0.5
        for i, r in enumerate(recall_at_10_list):
            if r <= target_recall:
                recall_half_life = i
                break

        # Get drift statistics
        drift_stats = simulator.get_drift_statistics()
        drift_stats["avg_embedding_drift"] = simulator.get_average_embedding_drift()

        return DriftBenchmarkResult(
            database=self.adapter.name,
            dataset=self.collection_name,
            drift_pattern=drift_pattern.value,
            num_timestamps=num_timestamps,
            timestamps=timestamps,
            recall_at_10=recall_at_10_list,
            recall_at_100=recall_at_100_list,
            ndcg_at_10=ndcg_at_10_list,
            latency_p50=latency_p50_list,
            latency_p95=latency_p95_list,
            corpus_size=corpus_size_list,
            survival_rate=survival_rate_list,
            recall_degradation_rate=recall_degradation_rate,
            recall_half_life=recall_half_life,
            final_recall=final_recall,
            initial_recall=initial_recall,
            drift_stats=drift_stats,
        )

    def _index_corpus(self, snapshot: CorpusSnapshot) -> None:
        """Index a corpus snapshot."""
        # Delete existing collection
        try:
            self.adapter.delete_index(self.collection_name)
        except Exception:
            pass

        # Create new index
        self.adapter.create_index(
            self.collection_name,
            dimensions=snapshot.embeddings.shape[1],
            metric="cosine"
        )

        # Insert vectors
        self.adapter.insert_vectors(
            self.collection_name,
            ids=snapshot.doc_ids,
            vectors=snapshot.embeddings.tolist()
        )

    def _update_index_incrementally(
        self,
        simulator: TemporalDriftSimulator,
        current_timestamp: int
    ) -> None:
        """
        Update index incrementally based on drift events.

        Note: Not all databases support efficient incremental updates.
        This is a best-effort implementation.
        """
        # Get events from last timestamp
        events = [
            e for e in simulator.drift_history
            if e.timestamp == current_timestamp
        ]

        for event in events:
            if event.drift_type.value == "delete":
                # Try to delete (may not be supported)
                pass  # Most adapters need full reindex
            elif event.drift_type.value == "add":
                # Try to add
                if event.new_embedding is not None:
                    try:
                        self.adapter.insert_vectors(
                            self.collection_name,
                            ids=[event.doc_id],
                            vectors=[event.new_embedding.tolist()]
                        )
                    except Exception:
                        pass  # Fall back to full reindex
            elif event.drift_type.value in ("update", "semantic"):
                # Updates typically require delete + insert
                pass  # Most adapters need full reindex

    def _evaluate_queries(
        self,
        query_ids: List[str],
        query_embeddings: np.ndarray,
        relevance_map: Dict[str, List[str]],
        sample_indices: np.ndarray,
        top_k: int,
        current_doc_ids: Optional[set] = None
    ) -> Dict[str, float]:
        """Evaluate queries against current index."""
        recalls_10 = []
        recalls_100 = []
        ndcgs_10 = []
        latencies = []

        for idx in sample_indices:
            query_id = query_ids[idx]
            query_vector = query_embeddings[idx].tolist()

            # Get relevant docs (filter to those still in corpus if specified)
            relevant_docs = relevance_map.get(query_id, [])
            if current_doc_ids is not None:
                # Only count docs that still exist as potentially retrievable
                relevant_docs = [d for d in relevant_docs if d in current_doc_ids]

            if not relevant_docs:
                continue  # Skip queries with no relevant docs in current corpus

            # Search
            result = self.adapter.search(
                self.collection_name,
                query_vector,
                top_k=top_k
            )

            latencies.append(result.latency_ms)

            # Compute metrics
            recalls_10.append(recall_at_k(result.ids, relevant_docs, 10))
            recalls_100.append(recall_at_k(result.ids, relevant_docs, 100))
            ndcgs_10.append(ndcg_at_k(result.ids, relevant_docs, 10))

        latency_stats = latency_percentiles(latencies) if latencies else {}

        return {
            "recall@10": np.mean(recalls_10) if recalls_10 else 0,
            "recall@100": np.mean(recalls_100) if recalls_100 else 0,
            "ndcg@10": np.mean(ndcgs_10) if ndcgs_10 else 0,
            "latency_p50": latency_stats.get("p50", 0),
            "latency_p95": latency_stats.get("p95", 0),
        }


def run_drift_benchmark(
    adapter: VectorDBAdapter,
    doc_ids: List[str],
    doc_embeddings: np.ndarray,
    query_ids: List[str],
    query_embeddings: np.ndarray,
    relevance_map: Dict[str, List[str]],
    collection_name: str = "drift_benchmark",
    num_timestamps: int = 10,
    drift_patterns: Optional[List[DriftPattern]] = None,
    seed: int = 42,
) -> Dict[str, DriftBenchmarkResult]:
    """
    Run drift benchmark across multiple drift patterns.

    Args:
        adapter: Connected database adapter
        doc_ids: Document IDs
        doc_embeddings: Document embeddings
        query_ids: Query IDs
        query_embeddings: Query embeddings
        relevance_map: Query to relevant docs mapping
        collection_name: Base collection name
        num_timestamps: Number of drift steps
        drift_patterns: Patterns to test (default: STABLE, MODERATE, HIGH_CHURN)
        seed: Random seed

    Returns:
        Dict mapping pattern name to results
    """
    if drift_patterns is None:
        drift_patterns = [
            DriftPattern.STABLE,
            DriftPattern.MODERATE,
            DriftPattern.HIGH_CHURN,
        ]

    results = {}

    for pattern in drift_patterns:
        print(f"\n{'='*60}")
        print(f"Testing drift pattern: {pattern.value}")
        print(f"{'='*60}")

        benchmark = DriftBenchmark(
            adapter=adapter,
            collection_name=f"{collection_name}_{pattern.value}",
            seed=seed
        )

        result = benchmark.run(
            doc_ids=doc_ids,
            doc_embeddings=doc_embeddings,
            query_ids=query_ids,
            query_embeddings=query_embeddings,
            relevance_map=relevance_map,
            num_timestamps=num_timestamps,
            drift_pattern=pattern,
        )

        results[pattern.value] = result

        # Print summary
        print(f"\nPattern: {pattern.value}")
        print(f"  Initial Recall@10: {result.initial_recall:.4f}")
        print(f"  Final Recall@10: {result.final_recall:.4f}")
        print(f"  Degradation Rate: {result.recall_degradation_rate:.4f}/step")
        if result.recall_half_life:
            print(f"  Recall Half-Life: {result.recall_half_life} steps")

    return results
