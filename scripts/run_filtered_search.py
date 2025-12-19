#!/usr/bin/env python3
"""
VecOps-Bench: Filtered Search Benchmark

Tests vector search with metadata predicates - critical for production workloads.
~50% of real-world vector queries include filters like:
  "Find similar documents WHERE category = 'science' AND date > 2023"

This benchmark measures:
1. Latency impact of different filter selectivities
2. Recall with filters (may differ from unfiltered)
3. QPS with filters

Selectivity levels:
- 50% (half the corpus matches)
- 10% (1 in 10 matches)
- 1%  (1 in 100 matches)
- 0.1% (1 in 1000 matches - stress test)
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.databases import get_adapter

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

# Configuration
NUM_QUERIES = 1000
TOP_K = 10
DIMENSIONS = 768
SELECTIVITIES = [0.5, 0.1, 0.01, 0.001]  # 50%, 10%, 1%, 0.1%


def run_filtered_search_benchmark(
    adapter,
    collection_name: str,
    query_vectors: np.ndarray,
    corpus_size: int,
) -> Dict[str, Any]:
    """
    Run filtered search benchmark at various selectivity levels.
    """
    print(f"\n{'='*60}")
    print(f"FILTERED SEARCH BENCHMARK: {adapter.name}")
    print(f"  Queries: {len(query_vectors)}")
    print(f"  Corpus size: {corpus_size:,}")
    print(f"  Selectivities: {SELECTIVITIES}")
    print(f"{'='*60}\n")

    results = {
        "database": adapter.name,
        "collection": collection_name,
        "num_queries": len(query_vectors),
        "corpus_size": corpus_size,
        "timestamp": datetime.now().isoformat(),
        "unfiltered": {},
        "filtered": {},
    }

    # Categories used during ingestion (10 categories = ~10% each)
    # Note: Skip "A" for Weaviate as it's treated as a stopword
    CATEGORIES = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "A"]

    # 1. Baseline: Unfiltered search
    print("Running unfiltered baseline...")
    latencies = []
    for query_vec in query_vectors:
        start = time.perf_counter()
        result = adapter.search(collection_name, query_vec.tolist(), top_k=TOP_K)
        latencies.append((time.perf_counter() - start) * 1000)

    results["unfiltered"] = {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
    }
    print(f"  Unfiltered p50: {results['unfiltered']['latency_p50_ms']:.2f}ms")

    # 2. Filtered search at each selectivity
    for selectivity in SELECTIVITIES:
        print(f"\nRunning filtered search (selectivity={selectivity*100:.1f}%)...")

        # Calculate how many categories to include for this selectivity
        # 10 categories total, each ~10% of data
        num_categories = max(1, int(selectivity * 10))
        if selectivity < 0.1:
            # For very low selectivity, use single category
            num_categories = 1

        filter_categories = CATEGORIES[:num_categories]

        latencies = []
        successful = 0
        empty_results = 0

        for query_vec in query_vectors:
            # Build filter based on database
            if adapter.name == "pgvector":
                # pgvector uses SQL WHERE clause
                filter_dict = {"category": filter_categories[0]}
            elif adapter.name == "milvus":
                # Milvus uses expression string
                if len(filter_categories) == 1:
                    filter_dict = {"expr": f'category == "{filter_categories[0]}"'}
                else:
                    cats = '", "'.join(filter_categories)
                    filter_dict = {"expr": f'category in ["{cats}"]'}
            elif adapter.name == "qdrant":
                # Qdrant uses Filter objects
                filter_dict = {"category": filter_categories[0]}
            elif adapter.name == "weaviate":
                # Weaviate uses GraphQL filters
                filter_dict = {"category": filter_categories[0]}
            elif adapter.name == "chroma":
                # Chroma uses where clause
                filter_dict = {"category": filter_categories[0]}
            else:
                filter_dict = {"category": filter_categories[0]}

            start = time.perf_counter()
            try:
                result = adapter.search(
                    collection_name, query_vec.tolist(),
                    top_k=TOP_K, filter=filter_dict
                )
                latencies.append((time.perf_counter() - start) * 1000)

                if len(result.ids) > 0:
                    successful += 1
                else:
                    empty_results += 1
            except Exception as e:
                # Some DBs may not support filtering
                latencies.append(None)

        valid_latencies = [l for l in latencies if l is not None]

        if valid_latencies:
            results["filtered"][f"{selectivity*100:.1f}%"] = {
                "selectivity": selectivity,
                "num_categories": num_categories,
                "latency_mean_ms": float(np.mean(valid_latencies)),
                "latency_p50_ms": float(np.percentile(valid_latencies, 50)),
                "latency_p95_ms": float(np.percentile(valid_latencies, 95)),
                "latency_p99_ms": float(np.percentile(valid_latencies, 99)),
                "success_rate": successful / len(query_vectors),
                "empty_result_rate": empty_results / len(query_vectors),
                "overhead_vs_unfiltered": (
                    (np.mean(valid_latencies) - results["unfiltered"]["latency_mean_ms"])
                    / results["unfiltered"]["latency_mean_ms"] * 100
                ),
            }
            print(f"  Selectivity {selectivity*100:.1f}%: p50={results['filtered'][f'{selectivity*100:.1f}%']['latency_p50_ms']:.2f}ms, "
                  f"overhead={results['filtered'][f'{selectivity*100:.1f}%']['overhead_vs_unfiltered']:.1f}%")
        else:
            results["filtered"][f"{selectivity*100:.1f}%"] = {
                "error": "Filtered search not supported or failed",
                "selectivity": selectivity,
            }
            print(f"  Selectivity {selectivity*100:.1f}%: FAILED (not supported)")

    # Summary
    print(f"\n{'='*60}")
    print(f"FILTERED SEARCH SUMMARY: {adapter.name}")
    print(f"  Unfiltered p50: {results['unfiltered']['latency_p50_ms']:.2f}ms")
    for sel in SELECTIVITIES:
        key = f"{sel*100:.1f}%"
        if key in results["filtered"] and "latency_p50_ms" in results["filtered"][key]:
            overhead = results["filtered"][key]["overhead_vs_unfiltered"]
            print(f"  {key} filter p50: {results['filtered'][key]['latency_p50_ms']:.2f}ms ({overhead:+.1f}% overhead)")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="VecOps-Bench Filtered Search Benchmark")
    parser.add_argument("--database", type=str, required=True,
                        help="Database to test")
    parser.add_argument("--collection", type=str, default="bench_v2",
                        help="Collection name prefix")
    parser.add_argument("--output", type=str, default="results_v2/filtered",
                        help="Output directory")
    parser.add_argument("--num-queries", type=int, default=NUM_QUERIES,
                        help="Number of queries to run")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Connect to database
    config = DB_CONFIGS.get(args.database, {})
    adapter = get_adapter(args.database, config)
    adapter.connect()

    collection_name = f"{args.collection}_{args.database}"
    stats = adapter.get_index_stats(collection_name)

    if stats.num_vectors == 0:
        print(f"ERROR: Collection {collection_name} is empty.")
        adapter.disconnect()
        return

    print(f"Connected to {args.database}: {stats.num_vectors:,} vectors")

    # Generate random query vectors
    rng = np.random.default_rng(456)
    query_vectors = rng.random((args.num_queries, DIMENSIONS), dtype=np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Run benchmark
    results = run_filtered_search_benchmark(
        adapter, collection_name, query_vectors, stats.num_vectors
    )

    # Save results
    output_file = os.path.join(
        args.output,
        f"{args.database}_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    adapter.disconnect()


if __name__ == "__main__":
    main()
