#!/usr/bin/env python3
"""
VecOps-Bench: SIFT1M Recall Benchmark

Uses SIFT1M's pre-computed ground truth (100 neighbors per query).
This is the gold standard from INRIA TEXMEX - no need to compute GT ourselves.

Key advantage: Direct comparison with ann-benchmarks results.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.databases import get_adapter

# Configuration
TOP_K_VALUES = [1, 10, 100]
HNSW_EF_SEARCH = 100


def compute_recall_at_k(retrieved_ids: List[str], ground_truth_indices: np.ndarray, k: int) -> float:
    """
    Compute Recall@K.

    Args:
        retrieved_ids: List of retrieved IDs (format: "sift_N")
        ground_truth_indices: Array of ground truth indices
        k: Number of results to consider
    """
    # Extract indices from IDs
    retrieved_indices = set()
    for rid in retrieved_ids[:k]:
        if rid.startswith("sift_"):
            retrieved_indices.add(int(rid.split("_")[1]))

    gt_set = set(ground_truth_indices[:k].tolist())

    if len(gt_set) == 0:
        return 0.0

    return len(retrieved_indices & gt_set) / len(gt_set)


def run_recall_benchmark(
    adapter,
    collection_name: str,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    num_queries: int = 10000
) -> Dict[str, Any]:
    """Run recall benchmark using pre-computed ground truth."""
    print(f"\n{'='*60}")
    print(f"RECALL BENCHMARK: {adapter.name} (SIFT1M)")
    print(f"  Queries: {num_queries}")
    print(f"  Ground truth: {ground_truth.shape}")
    print(f"  Top-K values: {TOP_K_VALUES}")
    print(f"{'='*60}\n")

    # Set ef_search
    adapter.set_ef_search(collection_name, HNSW_EF_SEARCH)

    results = {
        "num_queries": num_queries,
        "ef_search": HNSW_EF_SEARCH,
    }

    query_subset = queries[:num_queries]
    gt_subset = ground_truth[:num_queries]

    for k in TOP_K_VALUES:
        print(f"  Computing Recall@{k}...")
        recalls = []

        for i, (query_vec, gt) in enumerate(zip(query_subset, gt_subset)):
            result = adapter.search(collection_name, query_vec.tolist(), top_k=max(TOP_K_VALUES))
            recall = compute_recall_at_k(result.ids, gt, k)
            recalls.append(recall)

            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{num_queries} queries, current avg: {np.mean(recalls):.4f}")

        results[f"recall_at_{k}"] = {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls)),
            "min": float(np.min(recalls)),
            "max": float(np.max(recalls)),
        }
        print(f"  Recall@{k}: {results[f'recall_at_{k}']['mean']:.4f} (+/- {results[f'recall_at_{k}']['std']:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser(description="VecOps-Bench SIFT1M Recall")
    parser.add_argument("--database", type=str, required=True,
                        help="Database to test")
    parser.add_argument("--data-dir", type=str, default="data/sift1m",
                        help="SIFT1M data directory")
    parser.add_argument("--output", type=str, default="results_v2/sift1m/recall",
                        help="Output directory")
    parser.add_argument("--collection", type=str, default="sift1m",
                        help="Collection name prefix")
    parser.add_argument("--num-queries", type=int, default=10000,
                        help="Number of queries")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    print(f"\nLoading SIFT1M data from {args.data_dir}...")

    query_shape = np.load(os.path.join(args.data_dir, "queries.memmap.shape.npy"))
    queries = np.memmap(
        os.path.join(args.data_dir, "queries.memmap"),
        dtype='float32', mode='r', shape=tuple(query_shape)
    )
    ground_truth = np.load(os.path.join(args.data_dir, "ground_truth.npy"))

    print(f"  Queries: {queries.shape}")
    print(f"  Ground truth: {ground_truth.shape}")

    # Connect to database
    adapter = get_adapter(args.database, {})
    adapter.connect()

    collection_name = f"{args.collection}_{args.database}"

    # Verify collection exists
    stats = adapter.get_index_stats(collection_name)
    if stats.num_vectors == 0:
        print(f"ERROR: Collection {collection_name} is empty. Run baseline first.")
        adapter.disconnect()
        return

    print(f"  Collection: {stats.num_vectors:,} vectors")

    # Run recall benchmark
    results = {
        "database": args.database,
        "dataset": "SIFT1M",
        "collection": collection_name,
        "timestamp": datetime.now().isoformat(),
    }

    results["recall"] = run_recall_benchmark(
        adapter, collection_name, queries, ground_truth, args.num_queries
    )

    # Save results
    output_file = os.path.join(
        args.output,
        f"{args.database}_sift1m_recall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    adapter.disconnect()


if __name__ == "__main__":
    main()
