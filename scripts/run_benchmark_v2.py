#!/usr/bin/env python3
"""
VectorDB-Bench v2.0 - Production-Scale Benchmark Runner

Key improvements over v1.0:
- 100M vector scale with streaming (no RAM explosion)
- Resilience tests (cold start, crash recovery, memory pressure)
- Statistical rigor (confidence intervals, t-tests, effect sizes)
- Faiss + Elasticsearch baselines
- Query plan capture for root cause analysis

Usage:
    # Phase 1: Validation (10M)
    python scripts/run_benchmark_v2.py --phase validation

    # Phase 2: Production (100M)
    python scripts/run_benchmark_v2.py --phase production

    # Single database test
    python scripts/run_benchmark_v2.py --database pgvector --scale 1000000
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

try:
    import faiss
except ImportError:
    faiss = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.databases import get_adapter
from src.resilience import (
    measure_cold_start,
    measure_crash_recovery,
    run_memory_constrained_benchmark,
    measure_ingestion_speed,
)
from src.resilience.ingestion_speed import (
    vector_batch_generator,
    optimal_batch_size_for_db,
)
from src.stats import (
    compute_stats_for_trials,
    aggregate_trial_results,
    compare_databases,
    generate_latex_table,
)
from src.analysis import capture_pgvector_query_plan, capture_generic_timing


# =============================================================================
# Configuration
# =============================================================================

PHASE_CONFIG = {
    "validation": {
        "scale": 1_000_000,      # 1M vectors
        "num_queries": 1000,
        "trials": 3,
        "qps_duration": 30,
        "cold_start_trials": 5,
        "crash_recovery_trials": 3,
    },
    "production": {
        "scale": 100_000_000,    # 100M vectors
        "num_queries": 5000,
        "trials": 5,
        "qps_duration": 60,
        "cold_start_trials": 10,
        "crash_recovery_trials": 5,
    },
    "quick": {
        "scale": 100_000,        # 100K for testing
        "num_queries": 100,
        "trials": 2,
        "qps_duration": 10,
        "cold_start_trials": 3,
        "crash_recovery_trials": 2,
    }
}

DATABASES = ["faiss", "elasticsearch", "milvus", "qdrant", "pgvector", "chroma", "weaviate"]
DIMENSIONS = 768  # all-mpnet-base-v2


# =============================================================================
# System Checks (Prevent Invalid Experiments)
# =============================================================================

def verify_scale_validity(num_vectors: int, dimensions: int) -> bool:
    """
    Verify dataset is large enough for valid production claims.

    CRITICAL: Prevents the "L3 cache" trap where data fits in CPU cache.
    """
    import psutil

    data_size_gb = (num_vectors * dimensions * 4) / (1024 ** 3)
    available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)

    print(f"Dataset size: {data_size_gb:.1f} GB")
    print(f"Available RAM: {available_ram_gb:.1f} GB")

    if data_size_gb < 1.0:
        print("WARNING: Dataset < 1GB - results may only reflect cache performance!")
        return False

    if data_size_gb < available_ram_gb * 0.5:
        print("WARNING: Dataset fits entirely in RAM - not testing disk I/O!")
        return False

    return True


def check_docker_running(db_name: str) -> bool:
    """Check if database container is running."""
    import subprocess
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={db_name}"],
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())


# =============================================================================
# Ground Truth Computation
# =============================================================================

def compute_ground_truth_brute_force(
    all_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int = 100
) -> List[List[int]]:
    """
    Compute exact nearest neighbors using brute force search.

    This provides ground truth for recall computation.
    Uses Faiss IndexFlatIP for efficiency (still exact, but optimized).

    Args:
        all_vectors: Database vectors (N x D), assumed normalized for cosine
        query_vectors: Query vectors (Q x D), assumed normalized
        top_k: Number of neighbors to find

    Returns:
        List of lists containing ground truth indices for each query
    """
    if faiss is None:
        raise ImportError("faiss-cpu required for ground truth computation")

    # Create exact search index (Inner Product = Cosine for normalized vectors)
    index = faiss.IndexFlatIP(all_vectors.shape[1])
    index.add(all_vectors.astype('float32'))

    # Search
    _, indices = index.search(query_vectors.astype('float32'), top_k)

    return [list(row) for row in indices]


def compute_recall_at_k(
    retrieved_ids: List[str],
    ground_truth_ids: List[str],
    k: int
) -> float:
    """
    Compute Recall@K.

    Recall@K = |retrieved[:k] ∩ ground_truth[:k]| / k
    """
    retrieved_set = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids[:k])

    if len(gt_set) == 0:
        return 0.0

    return len(retrieved_set & gt_set) / len(gt_set)


# =============================================================================
# Benchmark Functions
# =============================================================================

def run_ingestion_benchmark(
    adapter,
    collection_name: str,
    num_vectors: int,
    dimensions: int,
    sample_size: int = 100000
) -> tuple:
    """
    Run ingestion benchmark with streaming.

    CRITICAL: Uses generator to avoid loading all vectors into RAM.
    Also stores a sample of vectors for ground truth computation.

    Returns:
        Tuple of (result_dict, sample_vectors, sample_ids)
    """
    print(f"  Ingesting {num_vectors:,} vectors...")

    batch_size = optimal_batch_size_for_db(adapter.name)

    # Store samples for ground truth computation
    sample_vectors = []
    sample_ids = []
    samples_per_batch = max(1, sample_size // (num_vectors // batch_size + 1))

    # Custom ingestion with sampling
    import psutil
    import time as time_module
    process = psutil.Process()
    mem_before = process.memory_info().rss

    # Create index
    adapter.create_index(collection_name, dimensions, metric="cosine")

    total_time = 0.0
    batches_completed = 0
    rng = np.random.default_rng(42)

    generator = vector_batch_generator(num_vectors, dimensions, batch_size)

    # Categories for filtered search testing
    CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    for ids, vectors in generator:
        # Sample some vectors for ground truth
        if len(sample_vectors) < sample_size:
            sample_idx = rng.choice(
                len(vectors),
                size=min(samples_per_batch, len(vectors), sample_size - len(sample_vectors)),
                replace=False
            )
            for idx in sample_idx:
                sample_vectors.append(vectors[idx])
                sample_ids.append(ids[idx])

        # Generate metadata with categories (uniform distribution)
        metadata = [
            {"category": CATEGORIES[i % len(CATEGORIES)]}
            for i in range(len(ids))
        ]

        batch_start = time_module.perf_counter()
        adapter.insert_vectors(collection_name, ids, vectors, metadata=metadata)
        batch_time = time_module.perf_counter() - batch_start

        total_time += batch_time
        batches_completed += 1

        if batches_completed % 100 == 0:
            progress = (batches_completed * batch_size) / num_vectors * 100
            current_rate = (batches_completed * batch_size) / total_time
            print(f"    Progress: {progress:.1f}% | Rate: {current_rate:.0f} vec/s")

    # Finalize index for databases that need it (e.g., Elasticsearch)
    if hasattr(adapter, 'finalize_index'):
        print(f"    Finalizing index...")
        finalize_time = adapter.finalize_index(collection_name)
        total_time += finalize_time
        print(f"    Finalize time: {finalize_time:.1f}s")

    mem_after = process.memory_info().rss
    stats = adapter.get_index_stats(collection_name)

    vectors_per_second = num_vectors / total_time
    time_to_100m = (100_000_000 / vectors_per_second) / 3600

    result = {
        "database": adapter.name,
        "total_vectors": num_vectors,
        "total_seconds": total_time,
        "vectors_per_second": vectors_per_second,
        "batches": batches_completed,
        "batch_size": batch_size,
        "time_to_100m_estimate_hours": time_to_100m,
        "memory_delta_mb": (mem_after - mem_before) / (1024 * 1024),
        "index_size_bytes": stats.index_size_bytes
    }

    print(f"    Rate: {vectors_per_second:,.0f} vectors/sec")
    print(f"    Time to 100M: {time_to_100m:.1f} hours")
    print(f"    Sampled {len(sample_vectors)} vectors for ground truth")

    return result, np.array(sample_vectors), sample_ids


def run_query_benchmark(
    adapter,
    collection_name: str,
    num_queries: int,
    dimensions: int,
    sample_vectors: np.ndarray,
    sample_ids: List[str],
    top_k_values: List[int] = [1, 10, 100]
) -> Dict[str, Any]:
    """
    Run query benchmark with proper recall computation.

    Uses sample vectors to compute ground truth via brute force,
    then measures recall of the ANN search against this ground truth.
    """
    print(f"  Running {num_queries} queries with recall computation...")

    # Limit queries to sample size
    actual_queries = min(num_queries, len(sample_vectors))
    if actual_queries < num_queries:
        print(f"    (Limited to {actual_queries} queries based on sample size)")

    # Select query vectors from the sample (use first N as queries)
    rng = np.random.default_rng(123)  # Different seed for query selection
    query_indices = rng.choice(len(sample_vectors), size=actual_queries, replace=False)
    query_vectors = sample_vectors[query_indices]

    # Compute ground truth using brute force on the FULL sample
    print(f"    Computing ground truth on {len(sample_vectors)} vectors...")
    max_k = max(top_k_values)

    # Build brute-force index on sample vectors
    if faiss is not None:
        gt_index = faiss.IndexFlatIP(dimensions)
        gt_index.add(sample_vectors.astype('float32'))
        _, gt_indices = gt_index.search(query_vectors.astype('float32'), max_k)
        ground_truth = [[sample_ids[idx] for idx in row if idx >= 0] for row in gt_indices]
    else:
        # Fallback: compute manually (slower)
        print("    WARNING: Faiss not available, using slow ground truth computation")
        ground_truth = []
        for q in query_vectors:
            # Cosine similarity (vectors already normalized)
            similarities = np.dot(sample_vectors, q)
            top_indices = np.argsort(similarities)[::-1][:max_k]
            ground_truth.append([sample_ids[idx] for idx in top_indices])

    print(f"    Ground truth computed. Running ANN queries...")

    # Run ANN queries and compute recall
    latencies = []
    recalls = {f"@{k}": [] for k in top_k_values}

    for i, (query_vec, gt_ids) in enumerate(zip(query_vectors, ground_truth)):
        # Search
        result = adapter.search(collection_name, query_vec.tolist(), top_k=max_k)
        latencies.append(result.latency_ms)

        # Compute recall at each k
        for k in top_k_values:
            recall = compute_recall_at_k(result.ids, gt_ids, k)
            recalls[f"@{k}"].append(recall)

        # Progress
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i+1}/{actual_queries}")

    # Compute statistics
    latencies_np = np.array(latencies)

    recall_stats = {}
    for k in top_k_values:
        recall_values = np.array(recalls[f"@{k}"])
        recall_stats[f"@{k}"] = {
            "mean": float(np.mean(recall_values)),
            "std": float(np.std(recall_values)),
            "min": float(np.min(recall_values)),
            "max": float(np.max(recall_values)),
        }

    print(f"    Recall@10: {recall_stats['@10']['mean']:.3f}")

    return {
        "num_queries": actual_queries,
        "latency": {
            "mean": float(np.mean(latencies_np)),
            "std": float(np.std(latencies_np)),
            "p50": float(np.percentile(latencies_np, 50)),
            "p95": float(np.percentile(latencies_np, 95)),
            "p99": float(np.percentile(latencies_np, 99)),
            "min": float(np.min(latencies_np)),
            "max": float(np.max(latencies_np)),
        },
        "recall": recall_stats,
        "qps_estimate": 1000 / float(np.mean(latencies_np)),
    }


def run_warmup(
    adapter,
    collection_name: str,
    dimensions: int,
    num_warmup_queries: int = 100
) -> None:
    """
    Run warmup queries to populate caches before benchmarking.

    This prevents cold JIT compilation and cache misses from
    skewing early measurements.
    """
    print(f"  Running {num_warmup_queries} warmup queries...")
    rng = np.random.default_rng(999)  # Different seed

    for _ in range(num_warmup_queries):
        query = rng.random(dimensions, dtype=np.float32)
        query = (query / np.linalg.norm(query)).tolist()
        try:
            adapter.search(collection_name, query, top_k=10)
        except Exception:
            pass  # Ignore errors during warmup


def run_qps_benchmark(
    adapter,
    collection_name: str,
    dimensions: int,
    duration_seconds: float = 30.0,
    include_warmup: bool = True
) -> Dict[str, Any]:
    """Run sustained QPS benchmark with warmup."""
    # Warmup phase
    if include_warmup:
        run_warmup(adapter, collection_name, dimensions)

    print(f"  Running QPS benchmark for {duration_seconds}s...")

    # Generate query pool
    rng = np.random.default_rng(42)
    query_pool = []
    for _ in range(1000):
        q = rng.random(dimensions, dtype=np.float32)
        query_pool.append((q / np.linalg.norm(q)).tolist())

    qps, latencies = adapter.benchmark_qps(
        collection_name,
        query_pool,
        top_k=10,
        duration_seconds=duration_seconds
    )

    latencies_np = np.array(latencies)

    print(f"    QPS: {qps:.1f}")
    print(f"    p50: {np.percentile(latencies_np, 50):.2f}ms")

    return {
        "qps": qps,
        "total_queries": len(latencies),
        "duration_seconds": duration_seconds,
        "warmup_included": include_warmup,
        "latency": {
            "mean": float(np.mean(latencies_np)),
            "std": float(np.std(latencies_np)),
            "p50": float(np.percentile(latencies_np, 50)),
            "p95": float(np.percentile(latencies_np, 95)),
            "p99": float(np.percentile(latencies_np, 99)),
        }
    }


def run_concurrent_qps_benchmark(
    adapter,
    collection_name: str,
    dimensions: int,
    num_threads: int = 4,
    duration_seconds: float = 30.0
) -> Dict[str, Any]:
    """
    Run concurrent QPS benchmark simulating production load.

    Uses ThreadPoolExecutor to send concurrent queries.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    print(f"  Running concurrent QPS benchmark ({num_threads} threads)...")

    # Generate query pool
    rng = np.random.default_rng(42)
    query_pool = []
    for _ in range(1000):
        q = rng.random(dimensions, dtype=np.float32)
        query_pool.append((q / np.linalg.norm(q)).tolist())

    # Shared state
    latencies = []
    latencies_lock = threading.Lock()
    stop_flag = threading.Event()

    def worker(thread_id: int):
        """Worker thread that runs queries until stop flag."""
        local_latencies = []
        query_idx = thread_id

        while not stop_flag.is_set():
            query = query_pool[query_idx % len(query_pool)]
            try:
                result = adapter.search(collection_name, query, top_k=10)
                local_latencies.append(result.latency_ms)
            except Exception:
                pass
            query_idx += num_threads

        # Merge results
        with latencies_lock:
            latencies.extend(local_latencies)

    # Run workers
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]

        # Wait for duration
        time.sleep(duration_seconds)
        stop_flag.set()

        # Wait for threads to finish
        for f in futures:
            f.result()

    elapsed = time.time() - start_time
    latencies_np = np.array(latencies) if latencies else np.array([0])

    concurrent_qps = len(latencies) / elapsed

    print(f"    Concurrent QPS: {concurrent_qps:.1f}")
    print(f"    Total queries: {len(latencies)}")

    return {
        "concurrent_qps": concurrent_qps,
        "num_threads": num_threads,
        "total_queries": len(latencies),
        "duration_seconds": elapsed,
        "latency": {
            "mean": float(np.mean(latencies_np)),
            "std": float(np.std(latencies_np)),
            "p50": float(np.percentile(latencies_np, 50)),
            "p95": float(np.percentile(latencies_np, 95)),
            "p99": float(np.percentile(latencies_np, 99)),
        }
    }


def run_filtered_search_benchmark(
    adapter,
    collection_name: str,
    dimensions: int,
    num_queries: int = 100,
    selectivities: List[float] = [0.01, 0.1, 0.5]
) -> Dict[str, Any]:
    """
    Run filtered search benchmark at multiple selectivities.

    Measures overhead for each selectivity level.
    """
    print(f"  Running filtered search benchmark...")

    rng = np.random.default_rng(42)
    results = {"no_filter": [], "selectivities": {}}

    # Unfiltered baseline
    for _ in range(num_queries):
        query = rng.random(dimensions, dtype=np.float32)
        query = (query / np.linalg.norm(query)).tolist()
        result = adapter.search(collection_name, query, top_k=10)
        results["no_filter"].append(result.latency_ms)

    baseline_mean = np.mean(results["no_filter"])

    # Filtered at each selectivity
    categories = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

    for selectivity in selectivities:
        num_cats = max(1, int(len(categories) * selectivity))
        filter_cats = categories[:num_cats]

        latencies = []
        for _ in range(num_queries):
            query = rng.random(dimensions, dtype=np.float32)
            query = (query / np.linalg.norm(query)).tolist()

            filter_dict = {"category": {"$in": filter_cats}}

            try:
                result = adapter.search(collection_name, query, top_k=10, filter=filter_dict)
                latencies.append(result.latency_ms)
            except Exception:
                latencies.append(None)

        valid_latencies = [l for l in latencies if l is not None]

        if valid_latencies:
            filtered_mean = np.mean(valid_latencies)
            overhead = (filtered_mean / baseline_mean - 1) * 100

            results["selectivities"][f"{selectivity:.0%}"] = {
                "mean_ms": float(filtered_mean),
                "overhead_percent": float(overhead),
                "success_rate": len(valid_latencies) / num_queries,
            }

            print(f"    Selectivity {selectivity:.0%}: {overhead:+.1f}% overhead")

    return {
        "no_filter": {
            "mean_ms": float(baseline_mean),
            "std_ms": float(np.std(results["no_filter"])),
        },
        "filtered": results["selectivities"],
    }


def run_resilience_tests(
    adapter,
    db_name: str,
    collection_name: str,
    dimensions: int,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run all resilience tests (cold start, crash recovery)."""
    print(f"  Running resilience tests...")

    # Sample query for tests
    rng = np.random.default_rng(42)
    query = rng.random(dimensions, dtype=np.float32)
    query = (query / np.linalg.norm(query)).tolist()

    results = {}

    # Cold start
    print(f"    Cold start ({config['cold_start_trials']} trials)...")
    try:
        cold_start = measure_cold_start(
            db_name=db_name,
            adapter=adapter,
            collection_name=collection_name,
            query_vector=query,
            num_trials=config["cold_start_trials"]
        )
        results["cold_start"] = cold_start.to_dict()
        print(f"      Mean: {cold_start.mean_ms:.1f}ms")
    except Exception as e:
        results["cold_start"] = {"error": str(e)}

    # Crash recovery
    print(f"    Crash recovery ({config['crash_recovery_trials']} trials)...")
    try:
        stats = adapter.get_index_stats(collection_name)
        crash_recovery = measure_crash_recovery(
            db_name=db_name,
            adapter=adapter,
            collection_name=collection_name,
            query_vector=query,
            expected_vector_count=stats.num_vectors,
            num_trials=config["crash_recovery_trials"]
        )
        results["crash_recovery"] = crash_recovery.to_dict()
        print(f"      Mean: {crash_recovery.mean_ms:.1f}ms")
    except Exception as e:
        results["crash_recovery"] = {"error": str(e)}

    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_full_benchmark(
    db_name: str,
    db_config: Dict[str, Any],
    phase_config: Dict[str, Any],
    output_dir: Path,
    skip_ingestion: bool = False
) -> Dict[str, Any]:
    """Run complete benchmark suite for a single database."""
    collection_name = f"bench_v2_{db_name}"

    print(f"\n{'='*60}")
    print(f"Benchmarking: {db_name.upper()}")
    print(f"Scale: {phase_config['scale']:,} vectors")
    print(f"{'='*60}")

    results = {
        "database": db_name,
        "scale": phase_config["scale"],
        "dimensions": DIMENSIONS,
        "timestamp": datetime.now().isoformat(),
        "phase_config": phase_config,
    }

    # Variables to store sample data for recall computation
    sample_vectors = None
    sample_ids = None

    try:
        # Initialize adapter
        adapter = get_adapter(db_name, db_config)
        adapter.connect()

        # Ingestion
        if not skip_ingestion:
            ingestion_result, sample_vectors, sample_ids = run_ingestion_benchmark(
                adapter, collection_name,
                phase_config["scale"], DIMENSIONS,
                sample_size=min(100000, phase_config["scale"])  # Cap sample size
            )
            results["ingestion"] = ingestion_result
            gc.collect()  # Free memory after ingestion
        else:
            print("  Skipping ingestion (--skip-ingestion)")
            # Generate dummy sample for skip-ingestion mode
            print("  Generating sample vectors for query benchmark...")
            rng = np.random.default_rng(42)
            sample_size = min(10000, phase_config["num_queries"] * 2)
            sample_vectors = rng.random((sample_size, DIMENSIONS), dtype=np.float32)
            sample_vectors = sample_vectors / np.linalg.norm(sample_vectors, axis=1, keepdims=True)
            sample_ids = [f"doc_{i}" for i in range(sample_size)]

        # Query benchmark (with recall computation)
        results["queries"] = run_query_benchmark(
            adapter, collection_name,
            phase_config["num_queries"], DIMENSIONS,
            sample_vectors, sample_ids
        )

        # QPS benchmark (sequential, with warmup)
        results["qps"] = run_qps_benchmark(
            adapter, collection_name, DIMENSIONS,
            phase_config["qps_duration"],
            include_warmup=True
        )

        # Concurrent QPS benchmark (production simulation)
        results["concurrent_qps"] = run_concurrent_qps_benchmark(
            adapter, collection_name, DIMENSIONS,
            num_threads=4,
            duration_seconds=phase_config["qps_duration"]
        )

        # Filtered search
        results["filtered_search"] = run_filtered_search_benchmark(
            adapter, collection_name, DIMENSIONS
        )

        # Resilience tests
        results["resilience"] = run_resilience_tests(
            adapter, db_name, collection_name, DIMENSIONS, phase_config
        )

        # Query plans (for pgvector)
        if db_name == "pgvector":
            print("  Capturing query plans...")
            try:
                import psycopg2
                conn = psycopg2.connect(**db_config)
                cursor = conn.cursor()

                query = np.random.default_rng(42).random(DIMENSIONS, dtype=np.float32)
                query = (query / np.linalg.norm(query)).tolist()

                plan = capture_pgvector_query_plan(
                    cursor, query, collection_name, filter_value="A"
                )
                results["query_plan"] = plan.to_dict()
                cursor.close()
                conn.close()
            except Exception as e:
                results["query_plan"] = {"error": str(e)}

        results["status"] = "success"

    except Exception as e:
        import traceback
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()
        print(f"ERROR: {e}")

    finally:
        try:
            if not skip_ingestion:
                adapter.delete_index(collection_name)
            adapter.disconnect()
        except Exception:
            pass

    # Save individual result
    result_file = output_dir / f"{db_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {result_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VectorDB-Bench v2.0 - Production-Scale Benchmarks"
    )
    parser.add_argument(
        "--phase",
        choices=["quick", "validation", "production"],
        default="quick",
        help="Benchmark phase (determines scale)"
    )
    parser.add_argument(
        "--database",
        nargs="+",
        default=DATABASES,
        help="Specific databases to benchmark"
    )
    parser.add_argument(
        "--scale",
        type=int,
        help="Override vector count"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config_v2.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results_v2",
        help="Output directory"
    )
    parser.add_argument(
        "--skip-ingestion",
        action="store_true",
        help="Skip data ingestion (use existing index)"
    )
    parser.add_argument(
        "--skip-scale-check",
        action="store_true",
        help="Skip scale validity check"
    )

    args = parser.parse_args()

    # Load config
    phase_config = PHASE_CONFIG[args.phase].copy()
    if args.scale:
        phase_config["scale"] = args.scale

    # Output directory
    output_dir = Path(args.output) / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVectorDB-Bench v2.0")
    print(f"Phase: {args.phase}")
    print(f"Scale: {phase_config['scale']:,} vectors")
    print(f"Databases: {args.database}")
    print(f"Output: {output_dir}")

    # Scale check
    if not args.skip_scale_check:
        valid = verify_scale_validity(phase_config["scale"], DIMENSIONS)
        if not valid and args.phase == "production":
            print("\nERROR: Scale too small for production claims!")
            print("Use --skip-scale-check to override.")
            sys.exit(1)

    # Load database configs
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        db_configs = config.get("databases", {})
    except FileNotFoundError:
        print(f"Config not found: {args.config}, using defaults")
        db_configs = {}

    # Run benchmarks
    all_results = []

    for db_name in args.database:
        if db_name not in DATABASES:
            print(f"Unknown database: {db_name}, skipping")
            continue

        db_config = db_configs.get(db_name, {}).get("config", {})

        result = run_full_benchmark(
            db_name, db_config, phase_config, output_dir,
            skip_ingestion=args.skip_ingestion
        )
        all_results.append(result)

        gc.collect()  # Clean up between databases

    # Save combined results
    combined_file = output_dir / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Database':<15} {'QPS':>10} {'p50 (ms)':>12} {'Cold Start':>12} {'Status':>10}")
    print("-" * 80)

    for r in all_results:
        db = r.get("database", "?")
        qps = r.get("qps", {}).get("qps", 0)
        p50 = r.get("queries", {}).get("latency", {}).get("p50", 0)
        cold = r.get("resilience", {}).get("cold_start", {}).get("mean_ms", 0)
        status = r.get("status", "?")

        print(f"{db:<15} {qps:>10.1f} {p50:>12.2f} {cold:>12.1f} {status:>10}")

    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
