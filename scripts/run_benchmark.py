#!/usr/bin/env python3
"""
VectorDB-Bench: Main benchmark runner script.

Production-oriented vector database benchmarking with:
- Standard recall/latency metrics
- Cold-start performance
- Filtered search evaluation
- Temporal drift simulation
- Cost-normalized metrics
- Operational complexity scoring

Usage:
    python scripts/run_benchmark.py --config experiments/config.yaml
    python scripts/run_benchmark.py --config experiments/config.yaml --database chroma qdrant
    python scripts/run_benchmark.py --config experiments/config.yaml --dataset scifact --quick
    python scripts/run_benchmark.py --config experiments/config.yaml --with-drift --with-cost
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import load_dataset, EmbeddedDataset
from src.databases import get_adapter, VectorDBAdapter, QueryResult
from src.workloads import generate_workload, QueryInstance
from src.metrics import recall_at_k, precision_at_k, ndcg_at_k, latency_percentiles

# New production-oriented modules
from src.drift import (
    DriftBenchmark,
    DriftPattern,
    TemporalDriftSimulator,
)
from src.cost import (
    CostTracker,
    CostAnalyzer,
    COST_MODELS,
)
from src.operational import (
    compute_complexity_score,
    DATABASE_OPERATIONAL_PROFILES,
    generate_complexity_report,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_standard_benchmark(
    adapter: VectorDBAdapter,
    dataset: EmbeddedDataset,
    collection_name: str,
    top_k_values: List[int] = [1, 10, 100],
    num_queries: int = 100
) -> Dict[str, Any]:
    """Run standard recall/latency benchmark."""
    results = {
        "recall": {},
        "precision": {},
        "ndcg": {},
        "latency": {},
        "qps": 0
    }

    # Sample queries
    query_indices = np.random.choice(
        len(dataset.queries),
        size=min(num_queries, len(dataset.queries)),
        replace=False
    )

    latencies = []
    all_recalls = {k: [] for k in top_k_values}
    all_precisions = {k: [] for k in top_k_values}
    all_ndcgs = {k: [] for k in top_k_values}

    max_k = max(top_k_values)

    for idx in tqdm(query_indices, desc="Running queries"):
        query = dataset.queries[idx]
        query_vector = dataset.query_embeddings[idx].tolist()

        # Search
        result = adapter.search(collection_name, query_vector, top_k=max_k)
        latencies.append(result.latency_ms)

        # Compute metrics for each k
        for k in top_k_values:
            retrieved_ids = result.ids[:k]
            relevant_ids = query.relevant_docs

            all_recalls[k].append(recall_at_k(retrieved_ids, relevant_ids, k))
            all_precisions[k].append(precision_at_k(retrieved_ids, relevant_ids, k))
            all_ndcgs[k].append(ndcg_at_k(retrieved_ids, relevant_ids, k))

    # Aggregate results
    for k in top_k_values:
        results["recall"][f"@{k}"] = float(np.mean(all_recalls[k]))
        results["precision"][f"@{k}"] = float(np.mean(all_precisions[k]))
        results["ndcg"][f"@{k}"] = float(np.mean(all_ndcgs[k]))

    results["latency"] = latency_percentiles(latencies)
    results["qps"] = 1000 / results["latency"]["mean"]  # queries per second

    return results


def run_cold_start_benchmark(
    adapter: VectorDBAdapter,
    dataset: EmbeddedDataset,
    collection_name: str,
    num_trials: int = 5
) -> Dict[str, Any]:
    """Measure cold-start latency."""
    cold_start_times = []

    query_vector = dataset.query_embeddings[0].tolist()

    for _ in tqdm(range(num_trials), desc="Cold start trials"):
        cold_start_ms = adapter.measure_cold_start(collection_name, query_vector)
        cold_start_times.append(cold_start_ms)

    return {
        "mean_ms": float(np.mean(cold_start_times)),
        "std_ms": float(np.std(cold_start_times)),
        "min_ms": float(np.min(cold_start_times)),
        "max_ms": float(np.max(cold_start_times)),
        "trials": num_trials
    }


def run_filtered_search_benchmark(
    adapter: VectorDBAdapter,
    dataset: EmbeddedDataset,
    collection_name: str,
    num_queries: int = 100
) -> Dict[str, Any]:
    """Benchmark filtered search performance."""
    # Generate synthetic filter values
    filter_values = {
        "category": ["A", "B", "C", "D", "E"],
        "year": [2020, 2021, 2022, 2023, 2024]
    }

    latencies_no_filter = []
    latencies_with_filter = []

    query_indices = np.random.choice(
        len(dataset.queries),
        size=min(num_queries, len(dataset.queries)),
        replace=False
    )

    for idx in tqdm(query_indices, desc="Filtered search"):
        query_vector = dataset.query_embeddings[idx].tolist()

        # Without filter
        result = adapter.search(collection_name, query_vector, top_k=10)
        latencies_no_filter.append(result.latency_ms)

        # With filter
        filter_dict = {
            "category": {"$eq": np.random.choice(filter_values["category"])}
        }
        try:
            result_filtered = adapter.search(
                collection_name, query_vector, top_k=10, filter=filter_dict
            )
            latencies_with_filter.append(result_filtered.latency_ms)
        except Exception:
            # Some databases may not support filtering
            latencies_with_filter.append(None)

    latencies_with_filter = [l for l in latencies_with_filter if l is not None]

    return {
        "no_filter": latency_percentiles(latencies_no_filter) if latencies_no_filter else {},
        "with_filter": latency_percentiles(latencies_with_filter) if latencies_with_filter else {},
        "filter_overhead_percent": (
            (np.mean(latencies_with_filter) / np.mean(latencies_no_filter) - 1) * 100
            if latencies_with_filter and latencies_no_filter else None
        )
    }


def run_qps_benchmark(
    adapter: VectorDBAdapter,
    dataset: EmbeddedDataset,
    collection_name: str,
    duration_seconds: float = 30.0
) -> Dict[str, Any]:
    """Measure sustained queries per second."""
    query_vectors = dataset.query_embeddings.tolist()

    qps, latencies = adapter.benchmark_qps(
        collection_name,
        query_vectors,
        top_k=10,
        duration_seconds=duration_seconds
    )

    return {
        "qps": qps,
        "latency": latency_percentiles(latencies),
        "duration_seconds": duration_seconds,
        "total_queries": len(latencies)
    }


def run_index_build_benchmark(
    adapter: VectorDBAdapter,
    dataset: EmbeddedDataset,
    collection_name: str
) -> Dict[str, Any]:
    """Measure index build time and memory."""
    import psutil

    # Get initial memory
    process = psutil.Process()
    mem_before = process.memory_info().rss

    # Create index and insert
    start_time = time.perf_counter()

    adapter.create_index(
        collection_name,
        dimensions=dataset.embedding_dim,
        metric="cosine"
    )

    # Prepare data
    ids = [doc.doc_id for doc in dataset.documents]
    vectors = dataset.doc_embeddings.tolist()

    insert_time = adapter.insert_vectors(collection_name, ids, vectors)

    total_time = time.perf_counter() - start_time

    # Get final memory
    mem_after = process.memory_info().rss
    mem_delta = mem_after - mem_before

    # Get index stats
    stats = adapter.get_index_stats(collection_name)

    return {
        "total_build_time_seconds": total_time,
        "insert_time_seconds": insert_time,
        "num_vectors": len(ids),
        "vectors_per_second": len(ids) / insert_time if insert_time > 0 else 0,
        "memory_delta_mb": mem_delta / (1024 * 1024),
        "index_size_bytes": stats.index_size_bytes
    }


def run_drift_benchmark(
    adapter: VectorDBAdapter,
    dataset: EmbeddedDataset,
    collection_name: str,
    drift_patterns: List[str] = None,
    num_timestamps: int = 10,
    num_queries: int = 50
) -> Dict[str, Any]:
    """
    Run temporal drift benchmark to measure recall degradation over time.

    This is a KEY NOVEL CONTRIBUTION - no other benchmark measures this.
    """
    if drift_patterns is None:
        drift_patterns = ["moderate", "high_churn"]

    results = {}

    for pattern_name in drift_patterns:
        try:
            pattern = DriftPattern[pattern_name.upper()]
        except KeyError:
            print(f"Unknown drift pattern: {pattern_name}, skipping...")
            continue

        print(f"  Running drift pattern: {pattern_name}")

        # Create drift benchmark
        drift_bench = DriftBenchmark(
            adapter=adapter,
            dataset=dataset,
            collection_name=f"{collection_name}_drift_{pattern_name}",
            drift_pattern=pattern
        )

        try:
            # Run the drift simulation
            drift_result = drift_bench.run(
                num_timestamps=num_timestamps,
                queries_per_timestamp=num_queries
            )

            results[pattern_name] = {
                "recall_degradation_rate": drift_result.recall_degradation_rate,
                "recall_half_life": drift_result.recall_half_life,
                "initial_recall": drift_result.recall_over_time[0] if drift_result.recall_over_time else None,
                "final_recall": drift_result.recall_over_time[-1] if drift_result.recall_over_time else None,
                "recall_curve": drift_result.recall_over_time,
                "latency_curve": drift_result.latency_over_time,
                "corpus_size_curve": drift_result.corpus_size_over_time,
                "num_updates": drift_result.num_updates,
                "num_deletes": drift_result.num_deletes,
                "num_adds": drift_result.num_adds,
            }
        except Exception as e:
            results[pattern_name] = {"error": str(e)}
        finally:
            drift_bench.cleanup()

    return results


def run_cost_tracked_benchmark(
    adapter: VectorDBAdapter,
    dataset: EmbeddedDataset,
    collection_name: str,
    cost_model_name: str,
    num_queries: int = 1000
) -> Dict[str, Any]:
    """
    Run benchmark with cost tracking enabled.

    Returns cost-normalized metrics like $/million queries.
    """
    # Initialize cost tracker
    tracker = CostTracker(
        cost_model_name=cost_model_name,
        database_name=adapter.__class__.__name__
    )

    # Start tracking
    tracker.start()

    # Run queries
    query_indices = np.random.choice(
        len(dataset.queries),
        size=min(num_queries, len(dataset.queries)),
        replace=True
    )

    recalls = []
    latencies = []

    for idx in tqdm(query_indices, desc="Cost-tracked queries"):
        query_vector = dataset.query_embeddings[idx].tolist()
        result = adapter.search(collection_name, query_vector, top_k=10)

        latencies.append(result.latency_ms)

        # Compute recall
        retrieved_ids = result.ids[:10]
        relevant_ids = dataset.queries[idx].relevant_docs
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, 10))

        tracker.record_queries(1)

        # Sample resources periodically
        if len(latencies) % 100 == 0:
            tracker.sample()

    # Stop tracking
    tracker.stop()

    # Compute metrics
    avg_recall = float(np.mean(recalls))
    qps = 1000 / np.mean(latencies)

    # Get cost breakdown
    breakdown = tracker.get_cost_breakdown(recall=avg_recall, qps=qps)

    return {
        "cost_model": cost_model_name,
        "num_queries": num_queries,
        "avg_recall": avg_recall,
        "avg_latency_ms": float(np.mean(latencies)),
        "qps": qps,
        "cost_per_million_queries_usd": breakdown.cost_per_million_queries_usd,
        "cost_per_query_usd": breakdown.cost_per_query_usd,
        "queries_per_dollar": breakdown.queries_per_dollar,
        "cost_per_recall_point": breakdown.cost_per_recall_point,
        "cost_per_qps": breakdown.cost_per_qps,
        "total_cost_usd": breakdown.total_cost_usd,
        "avg_cpu_percent": breakdown.avg_cpu_percent,
        "avg_memory_mb": breakdown.avg_memory_mb,
        "peak_memory_mb": breakdown.peak_memory_mb,
    }


def get_operational_complexity(db_name: str) -> Dict[str, Any]:
    """Get operational complexity score for a database."""
    # Map adapter names to profile names
    name_mapping = {
        "chroma": "chroma",
        "qdrant": "qdrant",
        "milvus": "milvus",
        "weaviate": "weaviate",
        "pgvector": "pgvector",
        "pinecone": "pinecone",
    }

    profile_name = name_mapping.get(db_name.lower())

    if profile_name and profile_name in DATABASE_OPERATIONAL_PROFILES:
        metrics = DATABASE_OPERATIONAL_PROFILES[profile_name]
        score = compute_complexity_score(metrics, profile_name)
        return score.to_dict()
    else:
        return {"error": f"No operational profile for {db_name}"}


def run_full_benchmark(
    db_name: str,
    db_config: Dict[str, Any],
    dataset: EmbeddedDataset,
    benchmark_config: Dict[str, Any],
    with_drift: bool = False,
    with_cost: bool = False,
    cost_model: str = "self_hosted_medium"
) -> Dict[str, Any]:
    """Run full benchmark suite for a single database."""
    collection_name = f"vectordb_bench_{dataset.name}"

    # Determine total number of benchmarks
    total_benchmarks = 5
    if with_drift:
        total_benchmarks += 1
    if with_cost:
        total_benchmarks += 1

    print(f"\n{'='*60}")
    print(f"Benchmarking: {db_name}")
    print(f"{'='*60}")

    results = {
        "database": db_name,
        "dataset": dataset.name,
        "num_documents": len(dataset.documents),
        "num_queries": len(dataset.queries),
        "embedding_dim": dataset.embedding_dim,
        "timestamp": datetime.now().isoformat()
    }

    # Add operational complexity (doesn't require running queries)
    results["operational_complexity"] = get_operational_complexity(db_name)

    current_benchmark = 0

    try:
        # Get adapter and connect
        adapter = get_adapter(db_name, db_config)
        adapter.connect()

        # Index build benchmark
        current_benchmark += 1
        print(f"\n[{current_benchmark}/{total_benchmarks}] Index Build Benchmark...")
        results["index_build"] = run_index_build_benchmark(
            adapter, dataset, collection_name
        )

        # Standard recall/latency benchmark
        current_benchmark += 1
        print(f"\n[{current_benchmark}/{total_benchmarks}] Standard Benchmark (Recall/Latency)...")
        results["standard"] = run_standard_benchmark(
            adapter,
            dataset,
            collection_name,
            top_k_values=benchmark_config.get("recall_k", [1, 10, 100]),
            num_queries=benchmark_config.get("num_queries", 100)
        )

        # Cold start benchmark
        current_benchmark += 1
        print(f"\n[{current_benchmark}/{total_benchmarks}] Cold Start Benchmark...")
        results["cold_start"] = run_cold_start_benchmark(
            adapter,
            dataset,
            collection_name,
            num_trials=benchmark_config.get("cold_start_trials", 5)
        )

        # Filtered search benchmark
        current_benchmark += 1
        if benchmark_config.get("filtered_search_enabled", True):
            print(f"\n[{current_benchmark}/{total_benchmarks}] Filtered Search Benchmark...")
            results["filtered_search"] = run_filtered_search_benchmark(
                adapter, dataset, collection_name
            )
        else:
            results["filtered_search"] = None

        # QPS benchmark
        current_benchmark += 1
        print(f"\n[{current_benchmark}/{total_benchmarks}] QPS Benchmark...")
        results["qps"] = run_qps_benchmark(
            adapter,
            dataset,
            collection_name,
            duration_seconds=benchmark_config.get("qps_duration_seconds", 30)
        )

        # Temporal Drift Benchmark (NOVEL)
        if with_drift:
            current_benchmark += 1
            print(f"\n[{current_benchmark}/{total_benchmarks}] Temporal Drift Benchmark (NOVEL)...")
            results["temporal_drift"] = run_drift_benchmark(
                adapter,
                dataset,
                collection_name,
                drift_patterns=benchmark_config.get("drift_patterns", ["moderate", "high_churn"]),
                num_timestamps=benchmark_config.get("drift_timestamps", 10),
                num_queries=benchmark_config.get("drift_queries_per_timestamp", 50)
            )

        # Cost-Tracked Benchmark (NOVEL)
        if with_cost:
            current_benchmark += 1
            print(f"\n[{current_benchmark}/{total_benchmarks}] Cost-Tracked Benchmark (NOVEL)...")
            results["cost_analysis"] = run_cost_tracked_benchmark(
                adapter,
                dataset,
                collection_name,
                cost_model_name=cost_model,
                num_queries=benchmark_config.get("cost_queries", 1000)
            )

        results["status"] = "success"

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error benchmarking {db_name}: {e}")

    finally:
        try:
            adapter.delete_index(collection_name)
            adapter.disconnect()
        except Exception:
            pass

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VectorDB-Bench: Production-Oriented Vector Database Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard benchmark
  python scripts/run_benchmark.py --config experiments/config.yaml

  # Full benchmark with temporal drift and cost analysis
  python scripts/run_benchmark.py --config experiments/config.yaml --with-drift --with-cost

  # Quick benchmark for testing
  python scripts/run_benchmark.py --config experiments/config.yaml --quick

  # Specific databases only
  python scripts/run_benchmark.py --config experiments/config.yaml --database chroma qdrant
        """
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--database", type=str, nargs="+", help="Specific databases to benchmark")
    parser.add_argument("--dataset", type=str, help="Specific dataset to use")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick run with fewer queries")

    # Novel benchmark options
    parser.add_argument(
        "--with-drift",
        action="store_true",
        help="Enable temporal drift benchmark (measures recall degradation over corpus evolution)"
    )
    parser.add_argument(
        "--with-cost",
        action="store_true",
        help="Enable cost-tracked benchmark (measures $/million queries)"
    )
    parser.add_argument(
        "--cost-model",
        type=str,
        default="self_hosted_medium",
        choices=list(COST_MODELS.keys()),
        help="Cost model to use for cost analysis"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all benchmarks including drift and cost (equivalent to --with-drift --with-cost)"
    )
    args = parser.parse_args()

    # Handle --full flag
    if args.full:
        args.with_drift = True
        args.with_cost = True

    # Load config
    config = load_config(args.config)

    # Override with quick settings
    if args.quick:
        config["benchmark"]["num_queries"] = 50
        config["benchmark"]["qps_duration_seconds"] = 10
        config["benchmark"]["cold_start_trials"] = 2

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which databases to benchmark
    if args.database:
        databases = {
            name: cfg for name, cfg in config["databases"].items()
            if name in args.database and cfg.get("enabled", True)
        }
    else:
        databases = {
            name: cfg for name, cfg in config["databases"].items()
            if cfg.get("enabled", True)
        }

    if not databases:
        print("No databases enabled for benchmarking!")
        return

    print(f"Databases to benchmark: {list(databases.keys())}")

    # Load dataset
    dataset_config = config["datasets"][0]  # Use first dataset
    if args.dataset:
        dataset_config = next(
            (d for d in config["datasets"] if d["name"] == args.dataset),
            dataset_config
        )

    print(f"\nLoading dataset: {dataset_config['name']}...")
    dataset = load_dataset(
        name=dataset_config["name"],
        source=dataset_config.get("source", "beir"),
        max_docs=dataset_config.get("num_docs"),
        embedding_model=config["embedding_model"]["name"],
        cache_dir="./data/cache"
    )

    print(f"Loaded {len(dataset.documents)} documents, {len(dataset.queries)} queries")

    # Run benchmarks
    all_results = []

    print(f"\nBenchmark Configuration:")
    print(f"  - Temporal Drift: {'ENABLED' if args.with_drift else 'disabled'}")
    print(f"  - Cost Tracking: {'ENABLED' if args.with_cost else 'disabled'}")
    if args.with_cost:
        print(f"  - Cost Model: {args.cost_model}")

    for db_name, db_config in databases.items():
        if not db_config.get("enabled", True):
            continue

        results = run_full_benchmark(
            db_name,
            db_config.get("config", {}),
            dataset,
            config["benchmark"],
            with_drift=args.with_drift,
            with_cost=args.with_cost,
            cost_model=args.cost_model
        )
        all_results.append(results)

        # Save individual results
        result_file = output_dir / f"{db_name}_{dataset.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to: {result_file}")

    # Save combined results
    combined_file = output_dir / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved combined results to: {combined_file}")

    # Print summary table
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)

    # Header depends on what benchmarks were run
    header = f"{'Database':<12} {'Recall@10':<10} {'Latency p50':<12} {'QPS':<8}"
    if args.with_cost:
        header += f" {'$/M Queries':<12}"
    header += f" {'Ops Score':<10} {'Status':<8}"
    print(header)
    print("-"*100)

    for result in all_results:
        db = result.get("database", "unknown")
        recall = result.get("standard", {}).get("recall", {}).get("@10", 0)
        latency = result.get("standard", {}).get("latency", {}).get("p50", 0)
        qps = result.get("qps", {}).get("qps", 0)
        status = result.get("status", "unknown")
        ops_score = result.get("operational_complexity", {}).get("overall_score", "N/A")

        row = f"{db:<12} {recall:<10.4f} {latency:<12.2f} {qps:<8.1f}"

        if args.with_cost:
            cost = result.get("cost_analysis", {}).get("cost_per_million_queries_usd", 0)
            row += f" ${cost:<11.2f}"

        if isinstance(ops_score, (int, float)):
            row += f" {ops_score:<10.0f}"
        else:
            row += f" {ops_score:<10}"

        row += f" {status:<8}"
        print(row)

    print("="*100)

    # Print drift summary if enabled
    if args.with_drift:
        print("\n" + "="*100)
        print("TEMPORAL DRIFT SUMMARY")
        print("="*100)
        print(f"{'Database':<12} {'Pattern':<12} {'Initial':<10} {'Final':<10} {'Degradation':<12} {'Half-Life':<10}")
        print("-"*100)

        for result in all_results:
            db = result.get("database", "unknown")
            drift_results = result.get("temporal_drift", {})

            for pattern, data in drift_results.items():
                if "error" in data:
                    print(f"{db:<12} {pattern:<12} ERROR: {data['error']}")
                else:
                    initial = data.get("initial_recall", 0)
                    final = data.get("final_recall", 0)
                    degradation = data.get("recall_degradation_rate", 0)
                    half_life = data.get("recall_half_life", "N/A")
                    if half_life is None:
                        half_life = "N/A"

                    print(f"{db:<12} {pattern:<12} {initial:<10.4f} {final:<10.4f} {degradation:<12.4f} {half_life!s:<10}")

        print("="*100)

    # Print operational complexity report
    print("\n" + "="*100)
    print("OPERATIONAL COMPLEXITY SCORES (Lower = Simpler)")
    print("="*100)
    print(f"{'Database':<12} {'Deploy':<8} {'Config':<8} {'Monitor':<8} {'Maint':<8} {'Docs':<8} {'Overall':<8} {'Team Size':<10}")
    print("-"*100)

    for result in all_results:
        db = result.get("database", "unknown")
        ops = result.get("operational_complexity", {})

        if "error" not in ops:
            print(
                f"{db:<12} "
                f"{ops.get('deployment_score', 0):<8.0f} "
                f"{ops.get('configuration_score', 0):<8.0f} "
                f"{ops.get('monitoring_score', 0):<8.0f} "
                f"{ops.get('maintenance_score', 0):<8.0f} "
                f"{ops.get('documentation_score', 0):<8.0f} "
                f"{ops.get('overall_score', 0):<8.0f} "
                f"{ops.get('recommended_team_size', 'N/A'):<10}"
            )

    print("="*100)


if __name__ == "__main__":
    main()
