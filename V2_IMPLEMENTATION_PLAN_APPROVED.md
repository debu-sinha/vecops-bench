# VectorDB-Bench v2.0 Implementation Plan - APPROVED

**Status:** APPROVED WITH MODIFICATIONS
**Expert Review Date:** December 2025
**Target Scale:** 100M vectors (upgraded from 10M)

---

## Executive Summary: Key Changes from Expert Review

| Original Plan | Expert Verdict | Updated Plan |
|---------------|----------------|--------------|
| 10M vectors | REJECTED | **100M vectors** |
| "Operational Complexity" (config counting) | REJECTED | **Resilience Tests** (TTR, TTFT) |
| Single RAM config | MODIFIED | **Memory-constrained mode** (8GB/16GB) |
| Publish v2.0 over v1.0 | MODIFIED | **New record: "VectorDB-Bench-XL"** |

---

## Part 1: 100M Vector Requirements

### 1.1 Data Size Calculation

```
Vector Data:
  100,000,000 vectors × 768 dimensions × 4 bytes = 307.2 GB

HNSW Index Overhead:
  ~1.5x vector size = 460 GB

Total Working Set:
  ~500-800 GB (depending on DB implementation)
```

### 1.2 Hardware Options

| Instance | vCPU | RAM | Storage | $/hr | Strategy |
|----------|------|-----|---------|------|----------|
| r6i.4xlarge | 16 | 128 GB | EBS | $1.01 | Memory-constrained mode |
| i4i.4xlarge | 16 | 128 GB | 3.75TB NVMe | $1.25 | Disk-based indexing |
| r6i.8xlarge | 32 | 256 GB | EBS | $2.02 | Partial RAM fit |
| r6i.16xlarge | 64 | 512 GB | EBS | $4.03 | Most data in RAM |
| r6i.24xlarge | 96 | 768 GB | EBS | $6.05 | Full RAM fit |

### 1.3 Recommended Strategy: Hybrid Approach

**Primary Instance: i4i.4xlarge** ($1.25/hr)
- 128 GB RAM (insufficient for 100M - forces disk I/O)
- 3.75 TB NVMe SSD (fast local storage)
- Tests TRUE production behavior under memory pressure

**Memory-Constrained Experiments:**
```bash
# Run each DB with artificial RAM limits
docker run --memory=16g --memory-swap=16g milvus/milvus
docker run --memory=8g --memory-swap=8g chromadb/chroma

# This reveals which DBs gracefully degrade vs crash
```

### 1.4 Revised Cost Estimate

```
Phase 1: Validation (10M vectors, i4i.4xlarge)
├── Runtime: 8 hours
├── Instance: $1.25/hr × 8 = $10
├── Storage: Included (NVMe)
└── Subtotal: ~$15

Phase 2: Production (100M vectors, i4i.4xlarge)
├── Runtime: 72-96 hours (longer due to disk I/O)
├── Instance: $1.25/hr × 96 = $120
├── EBS snapshot storage: $20
└── Subtotal: ~$150

Phase 3: Memory-Constrained Experiments
├── Runtime: 24 hours (re-run with limits)
├── Instance: $1.25/hr × 24 = $30
└── Subtotal: ~$35

Buffer (re-runs, debugging): $50

═══════════════════════════════════════
TOTAL BUDGET: ~$250-300
═══════════════════════════════════════
```

**Note:** This is LESS than the expert's $400 estimate because we're using
the memory-constraint approach instead of massive RAM instances.

---

## Part 2: Resilience Tests (Replaces "Operational Complexity")

### 2.1 The 3 Pillars of Operations

**OLD (Rejected):**
```python
# Counting config files - scientifically weak
complexity = num_docker_images * 10 + num_config_params * 0.5
```

**NEW (Approved):**
```python
# Time-based resilience measurements - scientifically rigorous
resilience = {
    "cold_start_ms": measure_cold_start(),        # TTFT
    "crash_recovery_ms": measure_crash_recovery(), # TTR
    "ingestion_rate": measure_ingestion_speed(),   # Throughput
}
```

### 2.2 Resilience Test Methodology

#### Pillar 1: Cold Start Latency (TTFT)
```python
def measure_cold_start(db_name: str, num_trials: int = 10) -> dict:
    """Time from container start to first successful query."""
    times = []
    for _ in range(num_trials):
        # Stop container
        subprocess.run(["docker", "stop", db_name])

        # Start and time to first query
        start = time.perf_counter()
        subprocess.run(["docker", "start", db_name])
        wait_for_healthy(db_name)
        result = execute_query(db_name, sample_vector)
        elapsed = time.perf_counter() - start

        times.append(elapsed * 1000)  # ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "p99_ms": np.percentile(times, 99),
        "trials": times
    }
```

#### Pillar 2: Crash Recovery Time (TTR)
```python
def measure_crash_recovery(db_name: str, num_trials: int = 5) -> dict:
    """Time from kill -9 to API returning 200 OK."""
    times = []
    for _ in range(num_trials):
        # Simulate crash (not graceful shutdown)
        container_id = get_container_id(db_name)
        subprocess.run(["docker", "kill", "--signal=KILL", container_id])

        # Time to recovery
        start = time.perf_counter()
        subprocess.run(["docker", "start", container_id])
        wait_for_api_ready(db_name)  # Poll until 200 OK
        elapsed = time.perf_counter() - start

        times.append(elapsed * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "max_ms": np.max(times),  # Worst case matters for SLAs
        "trials": times
    }
```

#### Pillar 3: Ingestion Latency
```python
def measure_ingestion_speed(db_name: str, num_vectors: int) -> dict:
    """Vectors per second during bulk load."""
    start = time.perf_counter()
    insert_vectors(db_name, vectors, batch_size=10000)
    elapsed = time.perf_counter() - start

    return {
        "total_seconds": elapsed,
        "vectors_per_second": num_vectors / elapsed,
        "time_to_100m_hours": (100_000_000 / (num_vectors / elapsed)) / 3600
    }
```

### 2.3 Memory Pressure Experiment

```python
def run_memory_constrained_benchmark(
    db_name: str,
    memory_limits: List[str] = ["64g", "32g", "16g", "8g"]
) -> dict:
    """Test behavior under increasing memory pressure."""
    results = {}

    for mem_limit in memory_limits:
        print(f"Testing {db_name} with --memory={mem_limit}")

        try:
            # Restart with memory limit
            restart_with_memory_limit(db_name, mem_limit)

            # Run standard benchmark
            qps = measure_qps(db_name, duration=60)
            latency_p99 = measure_latency_p99(db_name)

            results[mem_limit] = {
                "status": "success",
                "qps": qps,
                "latency_p99_ms": latency_p99
            }

        except Exception as e:
            results[mem_limit] = {
                "status": "crashed" if "OOM" in str(e) else "error",
                "error": str(e)
            }

    return results
```

**Expected Findings:**
- Milvus/Qdrant: Graceful degradation (designed for production)
- Chroma: OOM crash at 8-16GB (embedded-first design)
- pgvector: Uses PostgreSQL's buffer management (graceful)

---

## Part 3: Query Plan Analysis

### 3.1 pgvector: EXPLAIN ANALYZE

```python
def capture_pgvector_query_plan(cursor, query_vector, filter_value):
    """Capture PostgreSQL execution plan for filtered search."""

    query = """
    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
    SELECT id, embedding <=> %s::vector AS distance
    FROM documents
    WHERE category = %s
    ORDER BY distance
    LIMIT 10
    """

    cursor.execute(query, (query_vector, filter_value))
    plan = cursor.fetchone()[0]

    return {
        "plan": plan,
        "planning_time_ms": plan[0]["Planning Time"],
        "execution_time_ms": plan[0]["Execution Time"],
        "index_used": extract_index_name(plan),
        "rows_scanned": extract_rows_scanned(plan),
        "strategy": "index_scan" if "Index" in str(plan) else "seq_scan"
    }
```

**Expected Finding for pgvector -31% overhead:**
```
"Filtered search is FASTER because PostgreSQL's query optimizer
recognizes that category='science' reduces the candidate set to
10K rows. It scans the B-tree index on 'category' first, then
performs vector distance calculation only on the filtered subset.
This is 10x fewer distance computations than unfiltered search."
```

### 3.2 Chroma: Python Profiling

```python
import cProfile
import pstats

def profile_chroma_filtered_search(collection, query_vector, filter_dict):
    """Profile Chroma's execution to find bottleneck."""

    profiler = cProfile.Profile()
    profiler.enable()

    # Execute filtered search
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=10,
        where=filter_dict
    )

    profiler.disable()

    # Analyze
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')

    # Extract top time consumers
    top_functions = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        top_functions.append({
            "function": f"{func[0]}:{func[2]}",
            "cumulative_time_ms": ct * 1000,
            "calls": nc
        })

    return sorted(top_functions, key=lambda x: -x["cumulative_time_ms"])[:10]
```

**Expected Finding for Chroma +2978% overhead:**
```
"Chroma's filtered search performs the full HNSW traversal first,
returning 10K candidates, then applies the Python filter function
in a loop. Profiling shows 94% of execution time in
'_filter_results()' post-processing. This negates HNSW's O(log n)
advantage, resulting in O(n) behavior."
```

---

## Part 4: TCO (Total Cost of Ownership) Metric

### 4.1 Monthly TCO Calculation

```python
def calculate_monthly_tco(
    db_name: str,
    target_qps: float,
    dataset_size_gb: float,
    region: str = "us-east-1"
) -> dict:
    """Calculate monthly cost to sustain target QPS."""

    # Get benchmark results
    single_instance_qps = benchmark_results[db_name]["qps"]["mean"]

    # Calculate required instances
    instances_needed = math.ceil(target_qps / single_instance_qps)

    # Instance costs (r6i.4xlarge @ $1.008/hr)
    compute_cost = instances_needed * 1.008 * 24 * 30

    # Storage costs (EBS gp3 @ $0.08/GB/month)
    storage_cost = dataset_size_gb * 0.08 * instances_needed

    # Data transfer (assume 1KB per query response)
    data_transfer_gb = (target_qps * 3600 * 24 * 30 * 1024) / (1024**3)
    transfer_cost = data_transfer_gb * 0.09  # $0.09/GB outbound

    return {
        "instances_needed": instances_needed,
        "compute_cost_usd": compute_cost,
        "storage_cost_usd": storage_cost,
        "transfer_cost_usd": transfer_cost,
        "total_monthly_usd": compute_cost + storage_cost + transfer_cost,
        "cost_per_million_queries": (compute_cost + storage_cost) / (target_qps * 3600 * 24 * 30 / 1_000_000)
    }
```

### 4.2 Example TCO Comparison (100 QPS target)

| Database | QPS (single) | Instances | Monthly TCO | $/M queries |
|----------|--------------|-----------|-------------|-------------|
| pgvector | 398 | 1 | $750 | $0.29 |
| Weaviate | 436 | 1 | $750 | $0.26 |
| Qdrant | 309 | 1 | $750 | $0.37 |
| Milvus | 101 | 1 | $750 | $1.14 |
| Chroma | 324 | 1 | $750 | $0.36 |

*Note: These are estimates; actual results from 100M benchmark will differ.*

---

## Part 5: Zenodo Publication Strategy

### 5.1 v2.0 as NEW Record (Not Replacement)

**Repository Name:** `VectorDB-Bench-XL`

```
Title: VectorDB-Bench-XL: Large-Scale Production Benchmarks
       for Vector Database Systems

Description:
This dataset contains benchmark results for 7 vector databases
(Milvus, Qdrant, pgvector, Weaviate, Chroma, Faiss, Elasticsearch)
at 100M vector scale. Includes resilience tests, memory-constrained
experiments, and query execution plans.

Supersedes: VectorDB-Bench v1.0 (10.5281/zenodo.17924957)
```

### 5.2 Linking Strategy

In v2.0 README:
```markdown
## Relationship to v1.0

This benchmark supersedes [VectorDB-Bench v1.0](https://zenodo.org/records/17924957),
which evaluated systems at 100K scale. v2.0 (XL) provides:

- 1000x larger scale (100M vs 100K vectors)
- Resilience testing (crash recovery, memory pressure)
- Statistical rigor (5 trials, 95% CI, effect sizes)
- Additional baselines (Faiss, Elasticsearch)

v1.0 remains available as a reference for small-scale/embedded use cases.
```

---

## Part 6: Updated File Structure

```
vectordb-bench/
├── experiments/
│   ├── config_v2.yaml           # Updated config (100M)
│   └── config_memory_test.yaml  # Memory-constrained experiments
│
├── src/
│   ├── databases/
│   │   ├── faiss_adapter.py     # NEW: Speed of light baseline
│   │   └── elasticsearch_adapter.py  # NEW: Enterprise baseline
│   │
│   ├── resilience/              # NEW: Replaces operational/
│   │   ├── cold_start.py        # TTFT measurement
│   │   ├── crash_recovery.py    # TTR measurement
│   │   ├── memory_pressure.py   # Constrained experiments
│   │   └── ingestion_speed.py   # Bulk load performance
│   │
│   ├── analysis/                # NEW: Root cause analysis
│   │   ├── query_plans.py       # pgvector EXPLAIN capture
│   │   ├── profiler.py          # Chroma cProfile analysis
│   │   └── tco_calculator.py    # Cost modeling
│   │
│   └── stats/                   # NEW: Statistical rigor
│       ├── confidence_intervals.py
│       ├── hypothesis_tests.py  # t-tests, ANOVA
│       └── effect_sizes.py      # Cohen's d
│
├── scripts/
│   ├── run_benchmark_v2.py      # Main runner
│   ├── run_resilience_tests.py  # 3 Pillars
│   ├── run_memory_experiments.py
│   └── generate_figures_v2.py
│
├── infrastructure/
│   ├── setup_ec2_v2.sh          # i4i.4xlarge setup
│   ├── docker-compose-100m.yaml # 100M scale config
│   └── memory_limits.sh         # Apply RAM constraints
│
└── results_v2/
    ├── 10m_validation/
    ├── 100m_production/
    ├── memory_constrained/
    ├── query_plans/
    └── figures/
```

---

## Part 7: Execution Timeline

```
Week 1: Implementation
├── Day 1: Create Faiss adapter + Elasticsearch adapter
├── Day 2: Create resilience tests module
├── Day 3: Create memory pressure experiments
├── Day 4: Create query plan capture (pgvector + profiler)
├── Day 5: Create statistical analysis module
├── Day 6: Local testing with 100K subset
├── Day 7: Fix bugs, prepare for EC2

Week 2: Phase 1 (10M Validation)
├── Day 1: Launch i4i.4xlarge, run setup
├── Day 2-3: Run 10M benchmark (all DBs)
├── Day 4: Run resilience tests
├── Day 5: Analyze, fix issues
├── Day 6-7: Buffer

Week 3: Phase 2 (100M Production)
├── Day 1-4: Run 100M benchmark (72-96 hours)
├── Day 5: Run memory-constrained experiments
├── Day 6: Capture query plans
├── Day 7: Run resilience tests at 100M scale

Week 4: Analysis & Paper
├── Day 1-2: Statistical analysis, generate figures
├── Day 3-4: Rewrite paper with new results
├── Day 5: Publish Zenodo v2.0 (VectorDB-Bench-XL)
├── Day 6-7: Final review, prepare conference submission
```

---

## Part 8: Approval Checklist

- [x] Scale: 100M vectors (APPROVED)
- [x] Budget: ~$250-300 (APPROVED)
- [x] Baselines: Faiss + Elasticsearch (APPROVED)
- [x] Methodology: Resilience tests replace config counting (APPROVED)
- [x] Memory experiments: Docker --memory limits (APPROVED)
- [x] Zenodo: New record "VectorDB-Bench-XL" (APPROVED)
- [x] Timeline: 4 weeks (APPROVED)

---

## READY FOR IMPLEMENTATION

All expert feedback incorporated. Proceeding with code implementation.

---

*Plan approved: December 2025*
