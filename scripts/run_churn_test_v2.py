#!/usr/bin/env python3
"""
VecOps-Bench: Churn Test v2 (Temporal Drift Analysis) - FIXED RECALL

The "Star Feature" - measuring how performance degrades as corpus evolves.

FIXED: Proper recall computation
- Held-out query vectors (NOT in corpus)
- Ground truth computed on FULL corpus at each cycle
- Fair comparison across all databases

Methodology:
1. Load held-out queries (prepared by prepare_recall_data.py)
2. At each churn cycle:
   a. Delete 100K vectors from DB
   b. Insert 100K new vectors
   c. Compute GT on FULL current corpus (expensive but correct)
   d. Query index and measure recall vs GT
3. Track recall degradation over cycles
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print("WARNING: faiss not available - recall computation will be slow")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.databases import get_adapter

# Configuration
CHURN_SIZE = 100_000      # Vectors per cycle
NUM_CYCLES = 10           # Total cycles (1M churned = 10% of 10M)
TOP_K = 10                # For recall measurement
NUM_QUERIES = 1000        # Queries per measurement (use held-out set)
DIMENSIONS = 768


def load_vectors_memmap(path: str) -> np.ndarray:
    """Load vectors from memory-mapped file."""
    shape = tuple(np.load(path + '.shape.npy'))
    return np.memmap(path, dtype='float32', mode='r', shape=shape)


def compute_ground_truth_full(
    corpus_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int
) -> List[List[int]]:
    """
    Compute exact nearest neighbors on FULL corpus.

    This is the correct way - expensive but accurate.
    """
    n_corpus = len(corpus_vectors)

    if faiss is not None:
        index = faiss.IndexFlatIP(DIMENSIONS)
        index.add(corpus_vectors.astype('float32'))
        _, indices = index.search(query_vectors.astype('float32'), top_k)
        return [list(row) for row in indices]
    else:
        # Numpy fallback
        results = []
        for query in query_vectors:
            scores = np.dot(corpus_vectors, query)
            top_idx = np.argsort(scores)[::-1][:top_k]
            results.append(list(top_idx))
        return results


def compute_recall_at_k(retrieved_indices: List[int], gt_indices: List[int], k: int) -> float:
    """Compute Recall@K."""
    retrieved_set = set(retrieved_indices[:k])
    gt_set = set(gt_indices[:k])
    if len(gt_set) == 0:
        return 0.0
    return len(retrieved_set & gt_set) / len(gt_set)


def measure_performance(
    adapter,
    collection_name: str,
    corpus_vectors: np.ndarray,
    corpus_ids: List[str],
    query_vectors: np.ndarray,
    cycle_num: int,
    delete_time: float = 0,
    insert_time: float = 0
) -> Dict[str, Any]:
    """
    Measure recall and latency at current corpus state.

    FIXED: Computes GT on FULL corpus, not a sample.
    """
    print(f"    Computing ground truth on {len(corpus_ids):,} vectors...")
    gt_start = time.perf_counter()
    ground_truth = compute_ground_truth_full(corpus_vectors, query_vectors, TOP_K)
    gt_time = time.perf_counter() - gt_start
    print(f"    GT computed in {gt_time:.1f}s")

    # Build ID lookup
    id_to_idx = {id_: idx for idx, id_ in enumerate(corpus_ids)}

    # Run queries and measure
    latencies = []
    recalls = []

    print(f"    Running {len(query_vectors)} queries...")
    for i, (query_vec, gt_indices) in enumerate(zip(query_vectors, ground_truth)):
        start = time.perf_counter()
        result = adapter.search(collection_name, query_vec.tolist(), top_k=TOP_K)
        latency = (time.perf_counter() - start) * 1000  # ms
        latencies.append(latency)

        # Convert retrieved IDs to indices
        retrieved_indices = []
        for rid in result.ids:
            if rid in id_to_idx:
                retrieved_indices.append(id_to_idx[rid])

        recall = compute_recall_at_k(retrieved_indices, gt_indices, TOP_K)
        recalls.append(recall)

    return {
        "cycle": cycle_num,
        "corpus_size": len(corpus_ids),
        "recall_at_10": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "recall_min": float(np.min(recalls)),
        "recall_max": float(np.max(recalls)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "latency_mean_ms": float(np.mean(latencies)),
        "gt_computation_time_s": gt_time,
        "delete_time_s": delete_time,
        "insert_time_s": insert_time,
    }


def trigger_compaction(adapter, collection_name: str) -> Dict[str, Any]:
    """
    Trigger index compaction/vacuum for each database.

    This tests whether index maintenance can recover degraded recall.
    Different databases have different maintenance operations:
    - pgvector: VACUUM ANALYZE + REINDEX
    - Qdrant: Optimizer runs automatically, but we can force it
    - Milvus: compact() + flush()
    - Weaviate: No explicit compaction API
    - Chroma: No explicit compaction API
    """
    import subprocess

    result = {
        "database": adapter.name,
        "compaction_supported": False,
        "compaction_time_s": 0,
        "method": "none",
    }

    start_time = time.perf_counter()

    try:
        if adapter.name == "pgvector":
            # PostgreSQL: VACUUM ANALYZE rebuilds statistics, REINDEX rebuilds index
            print("  Running VACUUM ANALYZE and REINDEX...")
            result["method"] = "VACUUM ANALYZE + REINDEX"
            result["compaction_supported"] = True

            # Execute via psql
            table_name = collection_name
            commands = [
                f"VACUUM ANALYZE {table_name};",
                f"REINDEX INDEX {table_name}_embedding_idx;"
            ]
            for cmd in commands:
                try:
                    adapter.execute_sql(cmd)
                    print(f"    Executed: {cmd}")
                except Exception as e:
                    print(f"    Warning: {cmd} - {e}")

        elif adapter.name == "qdrant":
            # Qdrant: Force optimizer to run
            print("  Triggering Qdrant optimizer...")
            result["method"] = "optimizer trigger"
            result["compaction_supported"] = True

            import requests
            # Update collection to trigger optimization
            try:
                resp = requests.post(
                    f"http://localhost:6333/collections/{collection_name}",
                    json={"optimizers_config": {"indexing_threshold": 100}}
                )
                print(f"    Optimizer triggered: {resp.status_code}")
                # Wait for optimization
                time.sleep(30)
            except Exception as e:
                print(f"    Warning: {e}")

        elif adapter.name == "milvus":
            # Milvus: compact() merges segments
            print("  Running Milvus compact()...")
            result["method"] = "compact + flush"
            result["compaction_supported"] = True

            try:
                from pymilvus import Collection
                collection = Collection(collection_name)
                collection.compact()
                collection.wait_for_compaction_completed()
                collection.flush()
                print("    Compaction completed")
            except Exception as e:
                print(f"    Warning: {e}")

        elif adapter.name == "weaviate":
            # Weaviate: No explicit compaction, but we can force a flush
            print("  Weaviate: No explicit compaction API")
            result["method"] = "none (not supported)"
            result["compaction_supported"] = False

        elif adapter.name == "chroma":
            # Chroma: No explicit compaction
            print("  Chroma: No explicit compaction API")
            result["method"] = "none (not supported)"
            result["compaction_supported"] = False

        else:
            print(f"  Unknown database: {adapter.name}")
            result["method"] = "unknown"

    except Exception as e:
        print(f"  Compaction error: {e}")
        result["error"] = str(e)

    result["compaction_time_s"] = time.perf_counter() - start_time
    return result


def run_compaction_experiment(
    adapter,
    collection_name: str,
    corpus_vectors: np.ndarray,
    corpus_ids: List[str],
    query_vectors: np.ndarray,
) -> Dict[str, Any]:
    """
    Run compaction and measure recall recovery.

    This is a key finding for the paper:
    - Shows if degradation is recoverable
    - Quantifies the cost of recovery (time)
    - Provides practical guidance to practitioners
    """
    result = {
        "pre_compaction_recall": None,
        "post_compaction_recall": None,
        "compaction_details": None,
        "recovery_measurement": None,
    }

    # Measure recall before compaction (should match final churn cycle)
    print("  Measuring pre-compaction recall...")
    pre_result = measure_performance(
        adapter, collection_name, corpus_vectors, corpus_ids,
        query_vectors, cycle_num=-1  # -1 indicates pre-compaction
    )
    result["pre_compaction_recall"] = pre_result["recall_at_10"]
    print(f"    Pre-compaction Recall@10: {result['pre_compaction_recall']:.4f}")

    # Trigger compaction
    print("\n  Triggering compaction...")
    compaction_result = trigger_compaction(adapter, collection_name)
    result["compaction_details"] = compaction_result
    print(f"    Compaction time: {compaction_result['compaction_time_s']:.1f}s")

    if not compaction_result["compaction_supported"]:
        print(f"    Compaction not supported for {adapter.name}, skipping post-measurement")
        result["post_compaction_recall"] = result["pre_compaction_recall"]
        return result

    # Wait for index to stabilize
    print("  Waiting for index to stabilize (30s)...")
    time.sleep(30)

    # Measure recall after compaction
    print("  Measuring post-compaction recall...")
    post_result = measure_performance(
        adapter, collection_name, corpus_vectors, corpus_ids,
        query_vectors, cycle_num=-2  # -2 indicates post-compaction
    )
    result["post_compaction_recall"] = post_result["recall_at_10"]
    result["recovery_measurement"] = post_result

    # Calculate recovery
    if result["pre_compaction_recall"] > 0:
        improvement = ((result["post_compaction_recall"] - result["pre_compaction_recall"])
                      / result["pre_compaction_recall"] * 100)
        print(f"    Post-compaction Recall@10: {result['post_compaction_recall']:.4f}")
        print(f"    Recall improvement: {improvement:+.2f}%")

    return result


def run_churn_test(
    adapter,
    collection_name: str,
    corpus_vectors: np.ndarray,
    corpus_ids: List[str],
    query_vectors: np.ndarray,
) -> Dict[str, Any]:
    """
    Run the churn test with proper recall methodology.
    """
    results = {
        "database": adapter.name,
        "collection": collection_name,
        "churn_size": CHURN_SIZE,
        "num_cycles": NUM_CYCLES,
        "num_queries": len(query_vectors),
        "methodology": "held-out queries, full corpus GT per cycle",
        "timestamp": datetime.now().isoformat(),
        "cycles": [],
    }

    # Make mutable copies
    current_vectors = corpus_vectors.copy()
    current_ids = list(corpus_ids)
    next_doc_id = len(current_ids) + 10000  # Offset to avoid ID conflicts
    rng = np.random.default_rng(42)

    print(f"\n{'='*60}")
    print(f"CHURN TEST v2 (FIXED RECALL): {adapter.name}")
    print(f"Initial corpus: {len(current_ids):,} vectors")
    print(f"Query set: {len(query_vectors)} held-out vectors")
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
        churn_size = min(CHURN_SIZE, len(current_ids) // 2)
        delete_indices = rng.choice(len(current_ids), size=churn_size, replace=False)
        delete_ids = [current_ids[i] for i in delete_indices]

        # Delete from database
        print(f"    Deleting {churn_size:,} vectors...")
        delete_start = time.perf_counter()
        for doc_id in delete_ids:
            try:
                adapter.delete_vector(collection_name, doc_id)
            except Exception as e:
                pass  # Some DBs batch delete differently
        delete_time = time.perf_counter() - delete_start

        # Generate new vectors (random, normalized)
        new_vectors = rng.random((churn_size, DIMENSIONS), dtype=np.float32)
        new_vectors = new_vectors / np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_ids = [f"churn_{next_doc_id + i}" for i in range(churn_size)]
        next_doc_id += churn_size

        # Insert new vectors
        print(f"    Inserting {churn_size:,} new vectors...")
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

        print(f"  Delete: {delete_time:.2f}s, Insert: {insert_time:.2f}s")
        print(f"  Recall@{TOP_K}: {cycle_result['recall_at_10']:.4f}")
        print(f"  P99 Latency: {cycle_result['latency_p99_ms']:.2f} ms")

        # Calculate degradation from baseline
        baseline_recall = results["cycles"][0]["recall_at_10"]
        current_recall = cycle_result["recall_at_10"]
        if baseline_recall > 0:
            degradation = ((baseline_recall - current_recall) / baseline_recall) * 100
            print(f"  Recall Degradation from baseline: {degradation:+.1f}%\n")

    # Summary statistics (before compaction)
    recalls = [c["recall_at_10"] for c in results["cycles"]]
    results["summary"] = {
        "initial_recall": recalls[0],
        "final_recall": recalls[-1],
        "recall_degradation_pct": ((recalls[0] - recalls[-1]) / recalls[0]) * 100 if recalls[0] > 0 else 0,
        "min_recall": min(recalls),
        "max_recall": max(recalls),
        "recall_trend": recalls,
    }

    print(f"{'='*60}")
    print(f"SUMMARY (before compaction): {adapter.name}")
    print(f"  Initial Recall@10: {results['summary']['initial_recall']:.4f}")
    print(f"  Final Recall@10: {results['summary']['final_recall']:.4f}")
    print(f"  Total Degradation: {results['summary']['recall_degradation_pct']:.1f}%")
    print(f"{'='*60}\n")

    # ==========================================================================
    # COMPACTION EXPERIMENT - Measure recovery after index maintenance
    # ==========================================================================
    print(f"{'='*60}")
    print(f"COMPACTION EXPERIMENT: {adapter.name}")
    print(f"Testing if index maintenance recovers degraded recall...")
    print(f"{'='*60}\n")

    compaction_result = run_compaction_experiment(
        adapter, collection_name, current_vectors, current_ids, query_vectors
    )
    results["compaction"] = compaction_result

    # Update summary with post-compaction data
    if compaction_result.get("post_compaction_recall"):
        post_recall = compaction_result["post_compaction_recall"]
        recovery_pct = ((post_recall - results["summary"]["final_recall"]) /
                       (results["summary"]["initial_recall"] - results["summary"]["final_recall"]) * 100
                       if results["summary"]["initial_recall"] > results["summary"]["final_recall"] else 0)
        results["summary"]["post_compaction_recall"] = post_recall
        results["summary"]["recovery_pct"] = recovery_pct

        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY: {adapter.name}")
        print(f"  Initial Recall@10: {results['summary']['initial_recall']:.4f}")
        print(f"  Post-Churn Recall@10: {results['summary']['final_recall']:.4f} ({results['summary']['recall_degradation_pct']:.1f}% degradation)")
        print(f"  Post-Compaction Recall@10: {post_recall:.4f} ({recovery_pct:.1f}% recovery)")
        print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="VecOps-Bench Churn Test v2 (Fixed Recall)")
    parser.add_argument("--database", type=str, required=True,
                        help="Database to test (milvus, qdrant, pgvector, chroma, weaviate)")
    parser.add_argument("--collection", type=str, default="bench_v2",
                        help="Collection name prefix")
    parser.add_argument("--data-dir", type=str, default="data/recall_test",
                        help="Directory with corpus/query memmap files")
    parser.add_argument("--output", type=str, default="results_v2/churn",
                        help="Output directory")
    parser.add_argument("--cycles", type=int, default=NUM_CYCLES,
                        help="Number of churn cycles")
    parser.add_argument("--churn-size", type=int, default=CHURN_SIZE,
                        help="Vectors to churn per cycle")
    args = parser.parse_args()

    global NUM_CYCLES, CHURN_SIZE
    NUM_CYCLES = args.cycles
    CHURN_SIZE = args.churn_size

    os.makedirs(args.output, exist_ok=True)

    # Load held-out query vectors
    query_path = os.path.join(args.data_dir, "queries.memmap")
    if not os.path.exists(query_path + '.shape.npy'):
        print(f"ERROR: Query vectors not found at {query_path}")
        print("Run: python scripts/prepare_recall_data.py first")
        return

    query_vectors = load_vectors_memmap(query_path)
    query_vectors = np.array(query_vectors[:NUM_QUERIES])  # Load into RAM
    print(f"Loaded {len(query_vectors)} held-out query vectors")

    # Load corpus vectors (for local GT computation)
    corpus_path = os.path.join(args.data_dir, "corpus.memmap")
    corpus_ids_path = os.path.join(args.data_dir, "corpus_ids.json")

    if not os.path.exists(corpus_path + '.shape.npy'):
        print(f"ERROR: Corpus not found at {corpus_path}")
        return

    print(f"Loading corpus vectors...")
    corpus_vectors = load_vectors_memmap(corpus_path)
    corpus_vectors = np.array(corpus_vectors)  # Load into RAM (needed for GT)
    print(f"  Loaded {len(corpus_vectors):,} corpus vectors ({corpus_vectors.nbytes / 1e9:.1f} GB)")

    with open(corpus_ids_path, 'r') as f:
        corpus_ids = json.load(f)
    print(f"  Loaded {len(corpus_ids):,} corpus IDs")

    # Connect to database
    adapter = get_adapter(args.database, {})
    adapter.connect()

    collection_name = f"{args.collection}_{args.database}"
    stats = adapter.get_index_stats(collection_name)

    if stats.num_vectors == 0:
        print(f"ERROR: Collection {collection_name} is empty.")
        print("Run the baseline benchmark first to load vectors.")
        adapter.disconnect()
        return

    print(f"Connected to {args.database}: {stats.num_vectors:,} vectors in {collection_name}")

    # Verify corpus size matches
    if abs(stats.num_vectors - len(corpus_ids)) > 1000:
        print(f"WARNING: DB has {stats.num_vectors:,} but corpus has {len(corpus_ids):,}")
        print("Results may be inaccurate. Consider re-running ingestion.")

    # Run churn test
    results = run_churn_test(
        adapter, collection_name,
        corpus_vectors, corpus_ids, query_vectors
    )

    # Save results
    output_file = os.path.join(
        args.output,
        f"{args.database}_churn_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")

    adapter.disconnect()


if __name__ == "__main__":
    main()
