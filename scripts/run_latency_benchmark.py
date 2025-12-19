#!/usr/bin/env python3
"""
VecOps-Bench: Latency and QPS Benchmark

Runs latency and throughput tests on an existing collection.
Assumes data is already ingested via ingest_from_memmap.py
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.databases import get_adapter

# Configuration
NUM_QUERIES = 1000

# Database connection configs
DB_CONFIGS = {
    "pgvector": {
        "host": "localhost",
        "port": 5432,
        "database": "vectordb_bench",
        "user": "postgres",
        "password": "postgres",
    },
    "qdrant": {
        "host": "localhost",
        "port": 6333,
    },
    "milvus": {
        "host": "localhost",
        "port": 19530,
    },
    "weaviate": {
        "host": "localhost",
        "port": 8080,
    },
    "chroma": {
        "host": "localhost",
        "port": 8000,
    },
    "faiss": {},
}
TOP_K = 10
QPS_DURATION = 30  # seconds
WARMUP_QUERIES = 100
HNSW_EF_SEARCH = 100


def run_latency_test(adapter, collection_name: str, query_vectors: np.ndarray) -> dict:
    """Run single-threaded latency test."""
    print(f"  Running latency test ({len(query_vectors)} queries)...")

    # Set ef_search
    try:
        adapter.set_ef_search(collection_name, HNSW_EF_SEARCH)
    except:
        pass

    # Warmup
    for q in query_vectors[:WARMUP_QUERIES]:
        adapter.search(collection_name, q.tolist(), top_k=TOP_K)

    # Measure
    latencies = []
    for q in query_vectors:
        start = time.perf_counter()
        adapter.search(collection_name, q.tolist(), top_k=TOP_K)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "num_queries": len(query_vectors),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_std_ms": float(np.std(latencies)),
    }


def run_qps_test(adapter, collection_name: str, query_vectors: np.ndarray, duration: int = QPS_DURATION) -> dict:
    """Run throughput test."""
    print(f"  Running QPS test ({duration}s)...")

    query_list = [q.tolist() for q in query_vectors]
    num_queries = len(query_list)

    completed = 0
    start_time = time.time()
    end_time = start_time + duration

    while time.time() < end_time:
        q = query_list[completed % num_queries]
        adapter.search(collection_name, q, top_k=TOP_K)
        completed += 1

    elapsed = time.time() - start_time
    qps = completed / elapsed

    return {
        "duration_s": elapsed,
        "total_queries": completed,
        "qps": float(qps),
    }


def run_concurrent_qps_test(db_name: str, db_config: dict, collection_name: str,
                            query_vectors: np.ndarray, threads: int = 4,
                            duration: int = QPS_DURATION) -> dict:
    """Run concurrent throughput test with separate connections per thread."""
    print(f"  Running concurrent QPS test ({threads} threads, {duration}s)...")

    query_list = [q.tolist() for q in query_vectors]
    num_queries = len(query_list)

    stop_flag = [False]

    def worker():
        # Each thread gets its own connection
        thread_adapter = get_adapter(db_name, db_config)
        thread_adapter.connect()

        local_count = 0
        idx = 0
        try:
            while not stop_flag[0]:
                q = query_list[idx % num_queries]
                thread_adapter.search(collection_name, q, top_k=TOP_K)
                local_count += 1
                idx += 1
        finally:
            thread_adapter.disconnect()
        return local_count

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(worker) for _ in range(threads)]
        time.sleep(duration)
        stop_flag[0] = True

        total = sum(f.result() for f in futures)

    elapsed = time.time() - start_time
    qps = total / elapsed

    return {
        "threads": threads,
        "duration_s": elapsed,
        "total_queries": total,
        "qps": float(qps),
    }


def main():
    parser = argparse.ArgumentParser(description="VecOps-Bench Latency Benchmark")
    parser.add_argument("--database", type=str, required=True)
    parser.add_argument("--collection", type=str, default="bench_v2")
    parser.add_argument("--data-dir", type=str, default="data/recall_test")
    parser.add_argument("--output", type=str, default="results_v2/baseline")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load query vectors
    query_path = os.path.join(args.data_dir, "queries.memmap")
    shape = tuple(np.load(query_path + '.shape.npy'))
    queries = np.memmap(query_path, dtype='float32', mode='r', shape=shape)
    query_vectors = np.array(queries[:args.num_queries])

    print(f"\n{'='*60}")
    print(f"LATENCY BENCHMARK: {args.database}")
    print(f"  Queries: {len(query_vectors)}")
    print(f"{'='*60}\n")

    # Connect with proper config
    config = DB_CONFIGS.get(args.database, {})
    adapter = get_adapter(args.database, config)
    adapter.connect()

    collection_name = f"{args.collection}_{args.database}"
    stats = adapter.get_index_stats(collection_name)

    if stats.num_vectors == 0:
        print(f"ERROR: Collection {collection_name} is empty")
        adapter.disconnect()
        return

    print(f"  Collection: {stats.num_vectors:,} vectors")

    # Run tests
    results = {
        "database": args.database,
        "collection": collection_name,
        "num_vectors": stats.num_vectors,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
    }

    results["latency"] = run_latency_test(adapter, collection_name, query_vectors)
    print(f"    p50: {results['latency']['latency_p50_ms']:.2f}ms")
    print(f"    p99: {results['latency']['latency_p99_ms']:.2f}ms")

    results["qps_single"] = run_qps_test(adapter, collection_name, query_vectors)
    print(f"    QPS (single): {results['qps_single']['qps']:.0f}")

    results["qps_concurrent"] = run_concurrent_qps_test(args.database, config, collection_name, query_vectors)
    print(f"    QPS (4 threads): {results['qps_concurrent']['qps']:.0f}")

    # Save
    output_file = os.path.join(
        args.output,
        f"{args.database}_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    adapter.disconnect()


if __name__ == "__main__":
    main()
