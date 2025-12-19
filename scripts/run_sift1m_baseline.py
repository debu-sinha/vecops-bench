#!/usr/bin/env python3
"""
VecOps-Bench: SIFT1M Baseline Benchmark

Runs ingestion and query latency benchmarks on SIFT1M dataset.
SIFT1M: 1M vectors, 128 dimensions, image descriptors.

Metrics collected:
- Ingestion rate (vectors/second)
- Index build time
- Query latency (p50, p95, p99)
- QPS (single-threaded and concurrent)
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

# SIFT1M Configuration
DIMENSIONS = 128
CORPUS_SIZE = 1_000_000
NUM_QUERIES = 10_000
TOP_K = 10
BATCH_SIZE = 10_000

# HNSW parameters (standardized)
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 128
HNSW_EF_SEARCH = 100


def load_sift1m_data(data_dir: str) -> Dict[str, np.ndarray]:
    """Load SIFT1M corpus and queries from memmap files."""
    corpus_shape = np.load(os.path.join(data_dir, "corpus.memmap.shape.npy"))
    query_shape = np.load(os.path.join(data_dir, "queries.memmap.shape.npy"))

    corpus = np.memmap(
        os.path.join(data_dir, "corpus.memmap"),
        dtype='float32', mode='r', shape=tuple(corpus_shape)
    )
    queries = np.memmap(
        os.path.join(data_dir, "queries.memmap"),
        dtype='float32', mode='r', shape=tuple(query_shape)
    )

    print(f"  Loaded corpus: {corpus.shape}")
    print(f"  Loaded queries: {queries.shape}")

    return {"corpus": corpus, "queries": queries}


def run_ingestion_benchmark(
    adapter,
    collection_name: str,
    corpus: np.ndarray,
    batch_size: int = BATCH_SIZE
) -> Dict[str, Any]:
    """Run ingestion benchmark and return metrics."""
    print(f"\n{'='*60}")
    print(f"INGESTION BENCHMARK: {adapter.name} (SIFT1M)")
    print(f"  Vectors: {len(corpus):,}")
    print(f"  Dimensions: {corpus.shape[1]}")
    print(f"  Batch size: {batch_size:,}")
    print(f"{'='*60}\n")

    # Create collection with HNSW index
    adapter.create_collection(
        collection_name,
        dimension=DIMENSIONS,
        metric="cosine",
        hnsw_m=HNSW_M,
        hnsw_ef_construction=HNSW_EF_CONSTRUCTION
    )

    # Ingest in batches
    start_time = time.time()
    total_ingested = 0

    for i in range(0, len(corpus), batch_size):
        batch_end = min(i + batch_size, len(corpus))
        batch_vectors = corpus[i:batch_end]
        batch_ids = [f"sift_{j}" for j in range(i, batch_end)]

        # Add category metadata for filtered search compatibility
        batch_metadata = [{"category": chr(65 + (j % 10))} for j in range(i, batch_end)]

        adapter.insert(collection_name, batch_ids, batch_vectors.tolist(), batch_metadata)
        total_ingested += len(batch_ids)

        elapsed = time.time() - start_time
        rate = total_ingested / elapsed if elapsed > 0 else 0
        print(f"  Ingested: {total_ingested:,}/{len(corpus):,} ({rate:.0f} vec/s)")

    total_time = time.time() - start_time
    ingestion_rate = len(corpus) / total_time

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Ingestion rate: {ingestion_rate:.0f} vectors/second")

    return {
        "total_vectors": len(corpus),
        "total_time_s": total_time,
        "ingestion_rate": ingestion_rate,
        "batch_size": batch_size
    }


def run_latency_benchmark(
    adapter,
    collection_name: str,
    queries: np.ndarray,
    num_queries: int = 1000
) -> Dict[str, Any]:
    """Run query latency benchmark."""
    print(f"\n{'='*60}")
    print(f"LATENCY BENCHMARK: {adapter.name} (SIFT1M)")
    print(f"  Queries: {num_queries}")
    print(f"  Top-K: {TOP_K}")
    print(f"{'='*60}\n")

    # Set ef_search parameter
    adapter.set_ef_search(collection_name, HNSW_EF_SEARCH)

    # Sample queries
    query_subset = queries[:num_queries]

    # Warmup
    print("  Warming up...")
    for q in query_subset[:100]:
        adapter.search(collection_name, q.tolist(), top_k=TOP_K)

    # Measure latencies
    print("  Measuring latencies...")
    latencies = []
    for q in query_subset:
        start = time.perf_counter()
        adapter.search(collection_name, q.tolist(), top_k=TOP_K)
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)

    results = {
        "num_queries": num_queries,
        "top_k": TOP_K,
        "ef_search": HNSW_EF_SEARCH,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_std_ms": float(np.std(latencies)),
        "qps_single_thread": float(1000 / np.mean(latencies))
    }

    print(f"  p50: {results['latency_p50_ms']:.2f}ms")
    print(f"  p95: {results['latency_p95_ms']:.2f}ms")
    print(f"  p99: {results['latency_p99_ms']:.2f}ms")
    print(f"  QPS (single): {results['qps_single_thread']:.0f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="VecOps-Bench SIFT1M Baseline")
    parser.add_argument("--database", type=str, required=True,
                        help="Database to test")
    parser.add_argument("--data-dir", type=str, default="data/sift1m",
                        help="SIFT1M data directory")
    parser.add_argument("--output", type=str, default="results_v2/sift1m/baseline",
                        help="Output directory")
    parser.add_argument("--collection", type=str, default="sift1m",
                        help="Collection name prefix")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load data
    print(f"\nLoading SIFT1M data from {args.data_dir}...")
    data = load_sift1m_data(args.data_dir)

    # Connect to database
    adapter = get_adapter(args.database, {})
    adapter.connect()

    collection_name = f"{args.collection}_{args.database}"

    # Run benchmarks
    results = {
        "database": args.database,
        "dataset": "SIFT1M",
        "dimensions": DIMENSIONS,
        "corpus_size": len(data["corpus"]),
        "query_size": len(data["queries"]),
        "hnsw_m": HNSW_M,
        "hnsw_ef_construction": HNSW_EF_CONSTRUCTION,
        "hnsw_ef_search": HNSW_EF_SEARCH,
        "timestamp": datetime.now().isoformat(),
    }

    # Ingestion
    results["ingestion"] = run_ingestion_benchmark(
        adapter, collection_name, data["corpus"]
    )

    # Latency
    results["latency"] = run_latency_benchmark(
        adapter, collection_name, data["queries"]
    )

    # Save results
    output_file = os.path.join(
        args.output,
        f"{args.database}_sift1m_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    adapter.disconnect()


if __name__ == "__main__":
    main()
