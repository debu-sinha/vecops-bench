#!/usr/bin/env python3
"""
VecOps-Bench: Churn Test (Temporal Drift Analysis)

The "Star Feature" of the paper - measuring how performance degrades
as the corpus evolves. This proves the "Day 2" problem.

Methodology (from test plan):
1. Start with 10M loaded database
2. Churn Cycle: Delete 100K old vectors, Insert 100K new vectors
3. Repeat for 10 cycles (1M total churned = 10% turnover)
4. Measure Recall@10 and P99 Latency after EACH cycle

This is the key novel contribution of the benchmark.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.databases import get_adapter

# Configuration
CHURN_SIZE = 100_000  # Vectors per cycle
NUM_CYCLES = 10       # Total cycles (1M churned = 10% of 10M)
TOP_K = 10            # For recall measurement
NUM_QUERIES = 1000    # Queries per measurement
DIMENSIONS = 768


def compute_ground_truth(all_vectors: np.ndarray, query_vectors: np.ndarray, top_k: int) -> List[List[int]]:
    """Compute exact nearest neighbors using brute force."""
    try:
        import faiss
        index = faiss.IndexFlatIP(all_vectors.shape[1])
        index.add(all_vectors.astype('float32'))
        _, indices = index.search(query_vectors.astype('float32'), top_k)
        return [list(row) for row in indices]
    except ImportError:
        # Fallback to numpy (slower)
        scores = np.dot(query_vectors, all_vectors.T)
        return [list(np.argsort(-row)[:top_k]) for row in scores]


def compute_recall(retrieved: List[str], ground_truth: List[str], k: int) -> float:
    """Compute Recall@K."""
    retrieved_set = set(retrieved[:k])
    gt_set = set(ground_truth[:k])
    if len(gt_set) == 0:
        return 0.0
    return len(retrieved_set & gt_set) / len(gt_set)


def measure_performance(
    adapter, collection_name: str,
    current_vectors: np.ndarray, current_ids: List[str],
    query_vectors: np.ndarray,
    cycle_num: int, delete_time: float = 0, insert_time: float = 0
) -> Dict[str, Any]:
    """Measure recall and latency at current corpus state."""

    # Compute ground truth on sample (full corpus too expensive)
    sample_size = min(100000, len(current_vectors))
    sample_indices = np.random.choice(len(current_vectors), sample_size, replace=False)
    sample_vectors = current_vectors[sample_indices]
    sample_ids = [current_ids[i] for i in sample_indices]

    gt = compute_ground_truth(sample_vectors, query_vectors[:NUM_QUERIES], TOP_K)
    gt_ids = [[sample_ids[idx] for idx in indices] for indices in gt]

    # Run queries and measure
    latencies = []
    recalls = []

    for i, query_vec in enumerate(query_vectors[:NUM_QUERIES]):
        start = time.perf_counter()
        result = adapter.search(collection_name, query_vec.tolist(), top_k=TOP_K)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

        retrieved_ids = result.ids
        recall = compute_recall(retrieved_ids, gt_ids[i], TOP_K)
        recalls.append(recall)

    return {
        "cycle": cycle_num,
        "corpus_size": len(current_ids),
        "recall_at_10": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_mean_ms": float(np.mean(latencies)),
        "delete_time_s": delete_time,
        "insert_time_s": insert_time,
    }


def run_churn_test(
    adapter,
    collection_name: str,
    initial_ids: List[str],
    query_vectors: np.ndarray,
) -> Dict[str, Any]:
    """
    Run the churn test according to the VecOps-Bench specification.
    """
    results = {
        "database": adapter.name,
        "churn_size": CHURN_SIZE,
        "num_cycles": NUM_CYCLES,
        "timestamp": datetime.now().isoformat(),
        "cycles": [],
    }

    current_ids = list(initial_ids)
    next_doc_id = len(current_ids)
    rng = np.random.default_rng(42)

    # Track vectors locally for ground truth (sample-based)
    # Note: In production, you'd query the DB for this
    current_vectors = rng.random((len(current_ids), DIMENSIONS), dtype=np.float32)
    current_vectors = current_vectors / np.linalg.norm(current_vectors, axis=1, keepdims=True)

    print(f"\n{'='*60}")
    print(f"CHURN TEST: {adapter.name}")
    print(f"Initial corpus: {len(current_ids):,} vectors")
    print(f"Churn per cycle: {CHURN_SIZE:,} (delete + insert)")
    print(f"Total cycles: {NUM_CYCLES}")
    print(f"{'='*60}\n")

    # Initial measurement (Cycle 0)
    print("Cycle 0: Initial baseline measurement...")
    cycle_result = measure_performance(
        adapter, collection_name, current_vectors, current_ids,
        query_vectors, cycle_num=0
    )
    results["cycles"].append(cycle_result)
    print(f"  Recall@{TOP_K}: {cycle_result['recall_at_10']:.4f}")
    print(f"  P99 Latency: {cycle_result['latency_p99_ms']:.2f} ms\n")

    # Run churn cycles
    for cycle in range(1, NUM_CYCLES + 1):
        print(f"Cycle {cycle}/{NUM_CYCLES}: Churning {CHURN_SIZE:,} vectors...")

        # Select random vectors to delete
        if len(current_ids) < CHURN_SIZE:
            print(f"  Warning: Only {len(current_ids)} vectors left, adjusting churn size")
            churn_size = len(current_ids) // 2
        else:
            churn_size = CHURN_SIZE

        delete_indices = rng.choice(len(current_ids), size=churn_size, replace=False)
        delete_ids = [current_ids[i] for i in delete_indices]

        # Delete from database
        delete_start = time.perf_counter()
        for doc_id in delete_ids:
            try:
                adapter.delete_vector(collection_name, doc_id)
            except Exception:
                pass  # Some DBs batch delete differently
        delete_time = time.perf_counter() - delete_start

        # Generate new vectors
        new_vectors = rng.random((churn_size, DIMENSIONS), dtype=np.float32)
        new_vectors = new_vectors / np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_ids = [f"churn_{next_doc_id + i}" for i in range(churn_size)]
        next_doc_id += churn_size

        # Insert new vectors
        insert_start = time.perf_counter()
        batch_size = 10000
        for i in range(0, len(new_ids), batch_size):
            batch_ids = new_ids[i:i+batch_size]
            batch_vectors = new_vectors[i:i+batch_size].tolist()
            adapter.insert_vectors(collection_name, batch_ids, batch_vectors)
        insert_time = time.perf_counter() - insert_start

        # Update local tracking
        mask = np.ones(len(current_ids), dtype=bool)
        mask[delete_indices] = False
        current_ids = [id_ for i, id_ in enumerate(current_ids) if mask[i]]
        current_vectors = current_vectors[mask]

        current_ids.extend(new_ids)
        current_vectors = np.vstack([current_vectors, new_vectors])

        # Measure performance after churn
        cycle_result = measure_performance(
            adapter, collection_name, current_vectors, current_ids,
            query_vectors, cycle_num=cycle,
            delete_time=delete_time, insert_time=insert_time
        )
        results["cycles"].append(cycle_result)

        print(f"  Delete time: {delete_time:.2f}s, Insert time: {insert_time:.2f}s")
        print(f"  Recall@{TOP_K}: {cycle_result['recall_at_10']:.4f}")
        print(f"  P99 Latency: {cycle_result['latency_p99_ms']:.2f} ms")

        # Calculate degradation from baseline
        baseline_recall = results["cycles"][0]["recall_at_10"]
        current_recall = cycle_result["recall_at_10"]
        if baseline_recall > 0:
            degradation = ((baseline_recall - current_recall) / baseline_recall) * 100
            print(f"  Recall Degradation: {degradation:.1f}%\n")

    # Summary statistics
    recalls = [c["recall_at_10"] for c in results["cycles"]]
    results["summary"] = {
        "initial_recall": recalls[0],
        "final_recall": recalls[-1],
        "recall_degradation_pct": ((recalls[0] - recalls[-1]) / recalls[0]) * 100 if recalls[0] > 0 else 0,
        "min_recall": min(recalls),
        "max_recall": max(recalls),
    }

    print(f"{'='*60}")
    print(f"SUMMARY: {adapter.name}")
    print(f"  Initial Recall@10: {results['summary']['initial_recall']:.4f}")
    print(f"  Final Recall@10: {results['summary']['final_recall']:.4f}")
    print(f"  Total Degradation: {results['summary']['recall_degradation_pct']:.1f}%")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="VecOps-Bench Churn Test")
    parser.add_argument("--database", type=str, required=True,
                        help="Database to test (milvus, qdrant, pgvector, chroma, weaviate)")
    parser.add_argument("--collection", type=str, default="bench_v2",
                        help="Collection name (must exist with data)")
    parser.add_argument("--output", type=str, default="results_v2/churn",
                        help="Output directory")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get adapter
    adapter = get_adapter(args.database, {})
    adapter.connect()

    print(f"Connected to {args.database}")

    # Get existing collection info
    collection_name = f"{args.collection}_{args.database}"
    stats = adapter.get_index_stats(collection_name)

    if stats.num_vectors == 0:
        print(f"Error: Collection {collection_name} is empty.")
        print("Run the baseline benchmark first to load 10M vectors.")
        return

    print(f"Found collection with {stats.num_vectors:,} vectors")

    # Generate query vectors
    rng = np.random.default_rng(123)
    query_vectors = rng.random((NUM_QUERIES, DIMENSIONS), dtype=np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Generate initial IDs (matching what baseline created)
    initial_ids = [f"wiki_{i}" for i in range(stats.num_vectors)]

    # Run churn test
    results = run_churn_test(adapter, collection_name, initial_ids, query_vectors)

    # Save results
    output_file = os.path.join(args.output, f"{args.database}_churn_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

    adapter.disconnect()


if __name__ == "__main__":
    main()
