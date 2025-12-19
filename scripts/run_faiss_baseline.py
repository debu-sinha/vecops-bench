#!/usr/bin/env python3
"""
VecOps-Bench: FAISS Baseline (Speed of Light)

Runs FAISS as pure in-memory baseline - represents theoretical maximum performance.
No network overhead, no persistence, no containerization.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# FAISS import
try:
    import faiss
except ImportError:
    print("ERROR: faiss not installed. Run: pip install faiss-cpu")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
TOP_K = 10
NUM_QUERIES = 1000
WARMUP_QUERIES = 100
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128
HNSW_EF_SEARCH = 100


def run_faiss_baseline(corpus_path: str, query_path: str, gt_path: str, num_queries: int = NUM_QUERIES) -> dict:
    """Run FAISS HNSW baseline benchmark."""

    # Load data
    print("Loading data...")
    corpus_shape = tuple(np.load(corpus_path + '.shape.npy'))
    corpus = np.memmap(corpus_path, dtype='float32', mode='r', shape=corpus_shape)

    query_shape = tuple(np.load(query_path + '.shape.npy'))
    queries = np.memmap(query_path, dtype='float32', mode='r', shape=query_shape)
    query_vectors = np.array(queries[:num_queries])

    ground_truth = np.load(gt_path)[:num_queries]

    n_vectors, dim = corpus_shape
    n_vectors = int(n_vectors)  # Convert numpy int64 to Python int for JSON
    dim = int(dim)
    print(f"  Corpus: {n_vectors:,} x {dim}")
    print(f"  Queries: {len(query_vectors)}")

    results = {
        "database": "faiss",
        "num_vectors": n_vectors,
        "dimension": dim,
        "num_queries": num_queries,
        "timestamp": datetime.now().isoformat(),
    }

    # Build HNSW index
    print(f"\nBuilding FAISS HNSW index (M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION})...")
    start_build = time.time()

    index = faiss.IndexHNSWFlat(dim, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION

    # Add vectors in batches
    batch_size = 100_000
    for i in range(0, n_vectors, batch_size):
        end = min(i + batch_size, n_vectors)
        batch = np.array(corpus[i:end])
        index.add(batch)
        print(f"  Added {end:,}/{n_vectors:,} vectors")

    build_time = time.time() - start_build
    results["build_time_s"] = build_time
    print(f"  Build time: {build_time:.1f}s ({build_time/60:.1f}min)")

    # Set search parameter
    index.hnsw.efSearch = HNSW_EF_SEARCH

    # Warmup
    print("\nWarmup...")
    for q in query_vectors[:WARMUP_QUERIES]:
        index.search(q.reshape(1, -1), TOP_K)

    # Latency test
    print(f"Running latency test ({num_queries} queries)...")
    latencies = []
    all_results = []

    for q in query_vectors:
        start = time.perf_counter()
        D, I = index.search(q.reshape(1, -1), TOP_K)
        latencies.append((time.perf_counter() - start) * 1000)
        all_results.append(I[0])

    results["latency"] = {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_std_ms": float(np.std(latencies)),
    }

    print(f"  p50: {results['latency']['latency_p50_ms']:.2f}ms")
    print(f"  p99: {results['latency']['latency_p99_ms']:.2f}ms")

    # QPS test (single-threaded)
    print(f"Running QPS test (30s)...")
    completed = 0
    start_time = time.time()
    end_time = start_time + 30

    while time.time() < end_time:
        q = query_vectors[completed % num_queries]
        index.search(q.reshape(1, -1), TOP_K)
        completed += 1

    elapsed = time.time() - start_time
    qps = completed / elapsed

    results["qps_single"] = {
        "duration_s": elapsed,
        "total_queries": completed,
        "qps": float(qps),
    }
    print(f"  QPS (single): {qps:.0f}")

    # Recall calculation
    print("Computing recall...")
    all_results = np.array(all_results)

    recall_at_k = []
    for i in range(len(query_vectors)):
        retrieved = set(all_results[i])
        relevant = set(ground_truth[i][:TOP_K])  # Only compare against top-K ground truth
        recall_at_k.append(len(retrieved & relevant) / TOP_K)

    results["recall"] = {
        "recall_at_10_mean": float(np.mean(recall_at_k)),
        "recall_at_10_std": float(np.std(recall_at_k)),
    }
    print(f"  Recall@10: {results['recall']['recall_at_10_mean']:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="FAISS Baseline Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/recall_test")
    parser.add_argument("--output", type=str, default="results_v2/baseline")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    corpus_path = os.path.join(args.data_dir, "corpus.memmap")
    query_path = os.path.join(args.data_dir, "queries.memmap")
    gt_path = os.path.join(args.data_dir, "ground_truth.npy")

    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus not found at {corpus_path}")
        return

    if not os.path.exists(gt_path):
        print(f"ERROR: Ground truth not found at {gt_path}")
        return

    print("=" * 60)
    print("FAISS BASELINE (Speed of Light)")
    print("=" * 60)

    results = run_faiss_baseline(corpus_path, query_path, gt_path, args.num_queries)

    # Save results
    output_file = os.path.join(
        args.output,
        f"faiss_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("FAISS BASELINE SUMMARY")
    print("=" * 60)
    print(f"  Build time: {results['build_time_s']:.1f}s")
    print(f"  Latency p50: {results['latency']['latency_p50_ms']:.2f}ms")
    print(f"  Latency p99: {results['latency']['latency_p99_ms']:.2f}ms")
    print(f"  QPS: {results['qps_single']['qps']:.0f}")
    print(f"  Recall@10: {results['recall']['recall_at_10_mean']:.4f}")


if __name__ == "__main__":
    main()
