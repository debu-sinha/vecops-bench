"""Evaluation metrics for vector database benchmarking."""

from typing import List, Dict, Any
import numpy as np


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Recall@K."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate Precision@K."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if k == 0:
        return 0.0
    return len(retrieved_set & relevant_set) / k


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Calculate NDCG@K."""
    def dcg(relevances: List[int], k: int) -> float:
        relevances = relevances[:k]
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

    retrieved = retrieved_ids[:k]
    relevances = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved]

    ideal_relevances = sorted(relevances, reverse=True)

    dcg_score = dcg(relevances, k)
    idcg_score = dcg(ideal_relevances, k)

    if idcg_score == 0:
        return 0.0
    return dcg_score / idcg_score


def latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency percentiles."""
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def cost_per_million_queries(
    cloud_cost_per_hour: float,
    qps: float,
    duration_hours: float = 1.0
) -> float:
    """Calculate cost per 1 million queries."""
    queries_per_hour = qps * 3600
    cost_per_query = cloud_cost_per_hour / queries_per_hour
    return cost_per_query * 1_000_000


def operational_complexity_score(
    deployment_steps: int,
    config_options: int,
    monitoring_integrations: int,
    backup_difficulty: int,  # 1-5 scale
    documentation_quality: int,  # 1-5 scale
) -> float:
    """
    Calculate operational complexity score (lower is better).

    Returns a score from 0-100 where lower indicates simpler operations.
    """
    # Normalize each factor
    deployment_score = min(deployment_steps / 10, 1.0) * 20
    config_score = min(config_options / 50, 1.0) * 20
    monitoring_score = (1 - min(monitoring_integrations / 10, 1.0)) * 20
    backup_score = (backup_difficulty / 5) * 20
    doc_score = (1 - documentation_quality / 5) * 20

    return deployment_score + config_score + monitoring_score + backup_score + doc_score
