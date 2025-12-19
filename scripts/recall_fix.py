#!/usr/bin/env python3
"""
VecOps-Bench: Proper Recall Computation

Fixes the recall methodology bug:
- OLD (buggy): GT computed on 100K sample, ANN searches 10M index
- NEW (correct): GT computed on FULL corpus, held-out query vectors

Methodology:
1. During ingestion: Save ALL vectors to memory-mapped file
2. Reserve last N vectors as held-out queries (NOT inserted)
3. After ingestion: Compute GT using FAISS brute force on full corpus
4. Measure recall: Compare ANN results vs GT

Memory-efficient: Uses numpy memmap to avoid loading 30GB into RAM at once.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: faiss not available, using slow numpy fallback")

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
HELD_OUT_QUERIES = 10_000  # Queries NOT inserted into index
DIMENSIONS = 768
TOP_K_VALUES = [1, 10, 100]


def save_vectors_memmap(vectors: np.ndarray, path: str):
    """Save vectors to memory-mapped file."""
    fp = np.memmap(path, dtype='float32', mode='w+', shape=vectors.shape)
    fp[:] = vectors[:]
    fp.flush()
    del fp
    # Save shape metadata
    np.save(path + '.shape.npy', np.array(vectors.shape))


def load_vectors_memmap(path: str) -> np.ndarray:
    """Load vectors from memory-mapped file (lazy, doesn't load into RAM)."""
    shape = tuple(np.load(path + '.shape.npy'))
    return np.memmap(path, dtype='float32', mode='r', shape=shape)


def compute_ground_truth_batched(
    corpus_path: str,
    query_vectors: np.ndarray,
    top_k: int = 100,
    batch_size: int = 500_000
) -> List[List[int]]:
    """
    Compute ground truth using batched brute force.

    Memory efficient: processes corpus in batches.
    """
    corpus = load_vectors_memmap(corpus_path)
    n_corpus = corpus.shape[0]
    n_queries = query_vectors.shape[0]

    print(f"  Computing GT: {n_queries} queries vs {n_corpus:,} corpus vectors")

    if faiss is not None:
        # Use FAISS for efficiency
        # For 10M vectors at 768 dims: 10M * 768 * 4 = 30GB
        # We have 123GB RAM, so we can fit this
        print("  Using FAISS IndexFlatIP (brute force exact search)...")

        index = faiss.IndexFlatIP(DIMENSIONS)

        # Add corpus in batches to show progress
        for i in range(0, n_corpus, batch_size):
            end = min(i + batch_size, n_corpus)
            batch = np.array(corpus[i:end], dtype='float32')
            index.add(batch)
            print(f"    Added {end:,} / {n_corpus:,} vectors to GT index")

        print(f"  Index built. Searching for top-{top_k} neighbors...")
        start = time.perf_counter()
        _, indices = index.search(query_vectors.astype('float32'), top_k)
        elapsed = time.perf_counter() - start
        print(f"  Search completed in {elapsed:.1f}s")

        return [list(row) for row in indices]
    else:
        # Numpy fallback (much slower but works)
        print("  Using numpy fallback (this will be slow)...")
        all_results = []

        for q_idx, query in enumerate(query_vectors):
            if q_idx % 100 == 0:
                print(f"    Query {q_idx}/{n_queries}")

            # Compute similarity against all corpus vectors (batched)
            scores = np.zeros(n_corpus, dtype='float32')
            for i in range(0, n_corpus, batch_size):
                end = min(i + batch_size, n_corpus)
                batch = np.array(corpus[i:end], dtype='float32')
                scores[i:end] = np.dot(batch, query)

            top_indices = np.argsort(scores)[::-1][:top_k]
            all_results.append(list(top_indices))

        return all_results


def compute_recall_at_k(retrieved: List[int], ground_truth: List[int], k: int) -> float:
    """Compute Recall@K."""
    retrieved_set = set(retrieved[:k])
    gt_set = set(ground_truth[:k])
    if len(gt_set) == 0:
        return 0.0
    return len(retrieved_set & gt_set) / len(gt_set)


def run_recall_benchmark(
    adapter,
    collection_name: str,
    query_vectors: np.ndarray,
    ground_truth: List[List[int]],
    corpus_ids: List[str],
) -> Dict:
    """
    Run recall benchmark with proper methodology.

    Args:
        adapter: Database adapter
        collection_name: Collection to query
        query_vectors: Held-out query vectors (N x D)
        ground_truth: Pre-computed GT indices for each query
        corpus_ids: ID mapping for corpus vectors (index -> ID)
    """
    print(f"\n{'='*60}")
    print(f"RECALL BENCHMARK: {adapter.name}")
    print(f"  Queries: {len(query_vectors)}")
    print(f"  Top-K values: {TOP_K_VALUES}")
    print(f"{'='*60}\n")

    results = {
        "database": adapter.name,
        "collection": collection_name,
        "num_queries": len(query_vectors),
        "methodology": "held-out queries, full corpus GT",
        "timestamp": datetime.now().isoformat(),
        "recall": {},
        "latency": {},
    }

    # Build reverse lookup: ID -> index
    id_to_idx = {id_: idx for idx, id_ in enumerate(corpus_ids)}

    latencies = []
    recalls = {k: [] for k in TOP_K_VALUES}
    max_k = max(TOP_K_VALUES)

    for i, (query_vec, gt_indices) in enumerate(zip(query_vectors, ground_truth)):
        # Query the database
        start = time.perf_counter()
        result = adapter.search(collection_name, query_vec.tolist(), top_k=max_k)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        # Convert retrieved IDs to indices
        retrieved_indices = []
        for rid in result.ids:
            if rid in id_to_idx:
                retrieved_indices.append(id_to_idx[rid])

        # Compute recall at each K
        for k in TOP_K_VALUES:
            recall = compute_recall_at_k(retrieved_indices, gt_indices, k)
            recalls[k].append(recall)

        if (i + 1) % 1000 == 0:
            avg_recall_10 = np.mean(recalls[10]) if recalls[10] else 0
            print(f"  Progress: {i+1}/{len(query_vectors)} | Recall@10 so far: {avg_recall_10:.4f}")

    # Aggregate results
    print(f"\n  Results:")
    for k in TOP_K_VALUES:
        results["recall"][f"@{k}"] = {
            "mean": float(np.mean(recalls[k])),
            "std": float(np.std(recalls[k])),
            "min": float(np.min(recalls[k])),
            "max": float(np.max(recalls[k])),
        }
        print(f"    Recall@{k}: {results['recall'][f'@{k}']['mean']:.4f} (+/- {results['recall'][f'@{k}']['std']:.4f})")

    results["latency"] = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }
    print(f"    Latency p50: {results['latency']['p50_ms']:.2f}ms, p99: {results['latency']['p99_ms']:.2f}ms")

    return results


def prepare_ground_truth(
    corpus_path: str,
    query_path: str,
    gt_path: str,
    force_recompute: bool = False
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Prepare or load ground truth.

    Returns:
        Tuple of (query_vectors, ground_truth_indices)
    """
    query_vectors = load_vectors_memmap(query_path)
    query_vectors = np.array(query_vectors)  # Load into RAM (only 10K queries = 30MB)

    if os.path.exists(gt_path) and not force_recompute:
        print(f"  Loading cached ground truth from {gt_path}")
        gt_data = np.load(gt_path)
        ground_truth = [list(row) for row in gt_data]
        print(f"  Loaded {len(ground_truth)} ground truth entries")
    else:
        print(f"  Computing ground truth (this takes ~5-10 minutes for 10M vectors)...")
        ground_truth = compute_ground_truth_batched(
            corpus_path, query_vectors, top_k=max(TOP_K_VALUES)
        )
        # Cache for future use
        np.save(gt_path, np.array(ground_truth))
        print(f"  Ground truth cached to {gt_path}")

    return query_vectors, ground_truth


def main():
    parser = argparse.ArgumentParser(description="VecOps-Bench Proper Recall Computation")
    parser.add_argument("--database", type=str, required=True,
                        help="Database to test (milvus, qdrant, pgvector, chroma, weaviate, faiss)")
    parser.add_argument("--collection", type=str, default="bench_v2",
                        help="Collection name prefix")
    parser.add_argument("--data-dir", type=str, default="data/recall_test",
                        help="Directory with corpus/query memmap files")
    parser.add_argument("--output", type=str, default="results_v2/recall",
                        help="Output directory")
    parser.add_argument("--recompute-gt", action="store_true",
                        help="Force recompute ground truth (slow)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # File paths
    corpus_path = os.path.join(args.data_dir, "corpus.memmap")
    query_path = os.path.join(args.data_dir, "queries.memmap")
    gt_path = os.path.join(args.data_dir, "ground_truth.npy")
    ids_path = os.path.join(args.data_dir, "corpus_ids.json")

    # Check if data exists
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus not found at {corpus_path}")
        print("Run: python scripts/prepare_recall_data.py --scale 10000000 first")
        return

    if not os.path.exists(ids_path):
        print(f"ERROR: Corpus IDs not found at {ids_path}")
        return

    # Load corpus IDs
    print(f"Loading corpus IDs from {ids_path}...")
    with open(ids_path, 'r') as f:
        corpus_ids = json.load(f)
    print(f"  Loaded {len(corpus_ids):,} IDs")

    # Prepare ground truth
    print(f"\nPreparing ground truth...")
    query_vectors, ground_truth = prepare_ground_truth(
        corpus_path, query_path, gt_path, args.recompute_gt
    )

    # Connect to database
    print(f"\nConnecting to {args.database}...")
    config = DB_CONFIGS.get(args.database, {})
    adapter = get_adapter(args.database, config)
    adapter.connect()

    collection_name = f"{args.collection}_{args.database}"
    stats = adapter.get_index_stats(collection_name)
    print(f"  Collection {collection_name}: {stats.num_vectors:,} vectors")

    if stats.num_vectors == 0:
        print(f"ERROR: Collection is empty. Run ingestion first.")
        adapter.disconnect()
        return

    # Run recall benchmark
    results = run_recall_benchmark(
        adapter, collection_name,
        query_vectors, ground_truth, corpus_ids
    )

    # Save results
    output_file = os.path.join(
        args.output,
        f"{args.database}_recall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    adapter.disconnect()


if __name__ == "__main__":
    main()
