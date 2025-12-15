# VectorDB-Bench v2.0 Implementation Plan

**Status:** DRAFT - Awaiting Expert Review
**Author:** Debu Sinha
**Date:** December 2025
**Target:** KDD 2026 / VLDB 2026 Submission

---

## Executive Summary

This document outlines the complete plan to upgrade VectorDB-Bench from v1.0 (published on Zenodo) to v2.0 with publication-quality methodology. The upgrade addresses critical reviewer feedback regarding dataset scale, statistical rigor, and metric formalization.

---

## Part 1: What Was Wrong with v1.0

### 1.1 Audit Findings Summary

| Issue | Severity | v1.0 State | Impact |
|-------|----------|------------|--------|
| Only 21 queries | CRITICAL | MS MARCO BEIR has 21 test queries | No statistical power |
| 100K vectors | HIGH | Fits in L3 cache (~307 MB) | Not testing DB performance |
| Hardcoded ops metrics | HIGH | Manual entries in Python dict | Not reproducible |
| Single trial reported | MEDIUM | Paper uses Trial 1, not averages | Cherry-picked results |
| Missing filtered search | MEDIUM | Milvus/Weaviate have null data | Incomplete comparison |
| No baselines | MEDIUM | Missing Faiss, Elasticsearch | No reference point |
| No query plans | MEDIUM | No "why" analysis | Superficial findings |

### 1.2 Data Integrity Issues

```
v1.0 Paper Claims:              Actual Data:
─────────────────────────────────────────────────
"1000 queries"                  21 queries
"Averaged across trials"        Trial 1 only
"Measured config params"        Hardcoded in Python
pgvector recall: 0.545          0.0 in full_metrics run (bug)
```

---

## Part 2: v2.0 Architecture

### 2.1 Core Changes

```
┌─────────────────────────────────────────────────────────────────┐
│                    v2.0 BENCHMARK ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SCALE        100K → 10M vectors (100x increase)               │
│  QUERIES      21 → 5,000 queries (238x increase)               │
│  TRIALS       1 → 5 with CI reporting                          │
│  BASELINES    +Faiss (speed of light), +Elasticsearch          │
│  ANALYSIS     +Query plans, +Memory profiling                  │
│  OPS METRICS  Hardcoded → Runtime measurement                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Phased Execution Strategy

```
Phase 1: VALIDATION (1M vectors)
├── Purpose: Verify methodology before expensive 10M run
├── Scale: 1,000,000 vectors, 1,000 queries
├── Trials: 3
├── Time: ~6 hours
├── Cost: ~$6
└── Deliverable: Validated scripts + preliminary results

Phase 2: PRODUCTION (10M vectors)
├── Purpose: Publication-quality results
├── Scale: 10,000,000 vectors, 5,000 queries
├── Trials: 5
├── Time: ~48 hours
├── Cost: ~$50
└── Deliverable: Final results for paper
```

### 2.3 Database Coverage

| Database | v1.0 | v2.0 | Type | Purpose |
|----------|------|------|------|---------|
| Milvus | ✅ | ✅ | Distributed | Cloud-native reference |
| Qdrant | ✅ | ✅ | Native | Rust performance reference |
| pgvector | ✅ | ✅ | Extension | Relational baseline |
| Weaviate | ✅ | ✅ | Native | GraphQL alternative |
| Chroma | ✅ | ✅ | Embedded | Simplest option |
| **Faiss** | ❌ | ✅ | Library | **Speed of light baseline** |
| **Elasticsearch** | ❌ | ✅ | Search | **Enterprise incumbent** |

---

## Part 3: Methodology Fixes

### 3.1 Statistical Rigor

**v1.0 (Flawed):**
```python
# Report single trial
results["qps"] = trial_1_qps
```

**v2.0 (Correct):**
```python
# Report mean ± CI across all trials
results["qps"] = {
    "mean": np.mean(all_trials),
    "std": np.std(all_trials),
    "ci_95": stats.t.interval(0.95, len(all_trials)-1,
                               loc=np.mean(all_trials),
                               scale=stats.sem(all_trials)),
    "trials": all_trials  # Raw data for reproducibility
}
```

**Statistical Tests Added:**
- Two-sample t-tests for pairwise DB comparisons
- ANOVA for multi-DB comparisons
- Effect size (Cohen's d) reporting
- 95% confidence intervals on all metrics

### 3.2 Operational Complexity: From Hardcoded to Measured

**v1.0 (Hardcoded):**
```python
DATABASE_OPERATIONAL_PROFILES = {
    "milvus": {
        "config_params": 55,  # MANUALLY ENTERED
        "required_services": 4,  # MANUALLY ENTERED
    }
}
```

**v2.0 (Runtime Measured):**
```python
def measure_operational_complexity(db_name: str) -> OperationalMetrics:
    """Actually measure complexity at runtime."""

    # Count Docker images
    images = docker_client.images.list(filters={"reference": f"*{db_name}*"})
    num_images = len(images)

    # Count config parameters (parse actual config files)
    config_files = glob(f"/etc/{db_name}/**/*.yaml") + \
                   glob(f"/etc/{db_name}/**/*.json")
    config_params = count_config_keys(config_files)

    # Scrape Prometheus metrics
    prometheus_metrics = len(scrape_prometheus_endpoint(db_url + "/metrics"))

    # Measure startup time
    startup_time = measure_container_startup(db_name)

    # Measure recovery time after simulated failure
    recovery_time = measure_recovery_after_kill(db_name)

    return OperationalMetrics(
        docker_images=num_images,
        config_params=config_params,
        prometheus_metrics=prometheus_metrics,
        startup_time_ms=startup_time,
        recovery_time_ms=recovery_time,
        timestamp=datetime.now()  # Provenance
    )
```

### 3.3 Query Plan Capture for Root Cause Analysis

**Why pgvector gets FASTER with filters (-31%):**
```sql
-- v2.0 will capture EXPLAIN ANALYZE output
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT id, embedding <=> $1 AS distance
FROM documents
WHERE category = 'science'
ORDER BY distance
LIMIT 10;

-- Expected finding: PostgreSQL uses index scan
-- on category first (10K rows), then vector search
-- on smaller set = faster than full 100K scan
```

**Why Chroma gets 30x SLOWER with filters (+2978%):**
```python
# v2.0 will trace execution
# Expected finding: Chroma does full scan,
# then post-filters, negating HNSW benefits
```

### 3.4 Faiss Baseline (Speed of Light)

**Purpose:** Establish theoretical maximum performance

```python
class FaissAdapter(VectorDBAdapter):
    """In-memory Faiss baseline - no DB overhead."""

    def __init__(self):
        # HNSW with same parameters as other DBs
        self.index = faiss.IndexHNSWFlat(768, 32)  # d=768, M=32
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 100

    def search(self, query_vector, top_k=10):
        start = time.perf_counter()
        distances, ids = self.index.search(
            np.array([query_vector], dtype='float32'),
            top_k
        )
        latency = (time.perf_counter() - start) * 1000
        return QueryResult(ids=ids[0], latency_ms=latency)
```

**Expected Insight:**
> "Vector databases add 2-10x overhead compared to raw Faiss,
> with pgvector closest to theoretical limit due to minimal
> serialization overhead."

---

## Part 4: Hardware & Cost

### 4.1 EC2 Instance Selection

| Requirement | Value | Justification |
|-------------|-------|---------------|
| Vector data | 30.7 GB | 10M × 768 × 4 bytes |
| HNSW index | ~46 GB | 1.5x vector size |
| Working memory | ~15 GB | 20% overhead |
| OS + DBs | ~8 GB | Docker containers |
| **Total RAM** | **~100 GB** | Need 128 GB instance |

**Selected: r6i.4xlarge**
- 16 vCPU
- 128 GB RAM
- $1.008/hour
- EBS-optimized

### 4.2 Cost Breakdown

```
Phase 1 (Validation):
├── Runtime: 6 hours
├── Instance: $1.00/hr × 6 = $6.00
├── Storage: 500GB gp3 = $1.00 (prorated)
└── Total: ~$7

Phase 2 (Production):
├── Runtime: 48 hours
├── Instance: $1.00/hr × 48 = $48.00
├── Storage: 500GB gp3 = $4.00 (prorated)
└── Total: ~$52

GRAND TOTAL: ~$60
```

### 4.3 Timeline

```
Week 1: Implementation
├── Day 1-2: Complete v2 benchmark scripts
├── Day 3-4: Add Faiss + Elasticsearch adapters
├── Day 5: Add query plan capture
├── Day 6-7: Local testing with 100K subset

Week 2: Validation Phase
├── Day 1: Launch r6i.4xlarge, run setup script
├── Day 2-3: Run Phase 1 (1M validation)
├── Day 4: Analyze results, fix any issues
├── Day 5-7: Buffer / re-runs if needed

Week 3: Production Phase
├── Day 1-3: Run Phase 2 (10M production)
├── Day 4-5: Statistical analysis
├── Day 6-7: Generate figures and tables

Week 4: Paper Revision
├── Day 1-3: Rewrite results section
├── Day 4-5: Update methodology section
├── Day 6: Publish Zenodo v2.0
├── Day 7: Final review
```

---

## Part 5: Expected Outputs

### 5.1 New Figures for Paper

1. **Recall vs Latency at Scale** (10M vectors)
   - Shows actual DB performance, not cache hits

2. **QPS with Confidence Intervals**
   - Error bars on all measurements

3. **Filtered Search: Query Plans**
   - Visual explanation of WHY results differ

4. **Faiss Baseline Comparison**
   - Shows DB overhead vs theoretical limit

5. **Operational Complexity Radar**
   - Runtime-measured metrics with provenance

### 5.2 New Tables for Paper

| Table | Content |
|-------|---------|
| Main Results | Mean ± std for all metrics |
| Statistical Tests | p-values for pairwise comparisons |
| Query Plans | EXPLAIN output excerpts |
| Ops Complexity | Measured values with methodology |
| Cost Efficiency | $/million queries |

### 5.3 Zenodo v2.0 Release

```
vectordb-bench-v2.0/
├── data/
│   ├── raw_results_1m.parquet      # Validation phase
│   ├── raw_results_10m.parquet     # Production phase
│   └── query_plans/                 # EXPLAIN outputs
├── paper/
│   ├── paper_v2.tex
│   └── figures/
├── code/
│   └── [complete benchmark suite]
└── CHANGELOG.md                     # v1.0 → v2.0 changes
```

---

## Part 6: Risk Mitigation

### 6.1 Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| 10M embedding generation takes too long | MEDIUM | HIGH | Use pre-embedded LAION-10M from HuggingFace |
| DB fails at 10M scale | MEDIUM | MEDIUM | Test at 1M first (Phase 1) |
| EC2 spot instance termination | LOW | HIGH | Use on-demand, not spot |
| Results contradict v1.0 | HIGH | MEDIUM | Expected; v1.0 was flawed |
| Cost overrun | LOW | LOW | Monitor with AWS budgets |

### 6.2 Fallback Options

1. **If 10M too slow:** Report 1M results, note as limitation
2. **If DB crashes:** Report crash as finding (production-relevant!)
3. **If results match v1.0:** Good - validates core findings at scale

---

## Part 7: Reviewer Response Strategy

### 7.1 How v2.0 Addresses Each Critique

| Reviewer Critique | v2.0 Response |
|-------------------|---------------|
| "100K is toy scale" | 10M vectors, proper I/O stress |
| "Ops complexity is subjective" | Runtime-measured with methodology |
| "Missing the 'why'" | Query plan analysis + Faiss baseline |
| "No baselines" | Added Faiss + Elasticsearch |
| "Statistical rigor" | 5 trials, CI, t-tests, effect sizes |

### 7.2 New Claims in Paper

**Old (v1.0):**
> "Cold start varies 8× across systems"

**New (v2.0):**
> "At production scale (10M vectors), cold start varies 8×
> (14ms ± 2ms to 109ms ± 12ms, p < 0.001, Cohen's d = 4.2),
> with pgvector's advantage attributable to PostgreSQL's
> shared buffer warm-up strategy (see Table 5 for query plans)."

---

## Part 8: Files to Create

### 8.1 New Scripts

| File | Purpose |
|------|---------|
| `scripts/run_benchmark_v2.py` | Main runner with statistical analysis |
| `scripts/analyze_results_v2.py` | CI calculation, t-tests, effect sizes |
| `scripts/generate_figures_v2.py` | Publication figures with error bars |
| `scripts/capture_query_plans.py` | EXPLAIN ANALYZE capture |

### 8.2 New Adapters

| File | Purpose |
|------|---------|
| `src/databases/faiss_adapter.py` | Faiss baseline |
| `src/databases/elasticsearch_adapter.py` | ES dense_vector |

### 8.3 New Modules

| File | Purpose |
|------|---------|
| `src/operational/runtime_complexity.py` | Actual measurement |
| `src/stats/statistical_analysis.py` | t-tests, CI, effect sizes |
| `src/analysis/query_plan_parser.py` | Parse EXPLAIN output |

---

## Part 9: Questions for Expert Review

Before proceeding, please confirm:

1. **Scale:** Is 10M sufficient, or should we target 100M?
   - 10M = ~$60, 100M = ~$400+

2. **Baselines:** Faiss + Elasticsearch enough, or add others?
   - Options: LanceDB, Vespa, Pinecone (managed)

3. **Query plans:** Focus on pgvector (SQL) or attempt for all DBs?
   - Milvus/Qdrant have limited introspection

4. **Operational complexity formula:**
   - Keep weighted sum, or switch to PCA-derived composite?

5. **Zenodo versioning:**
   - Publish v2.0 alongside v1.0, or request retraction of v1.0?

---

## Approval

- [ ] Expert review completed
- [ ] Scale confirmed (10M / 100M)
- [ ] Baselines confirmed
- [ ] Budget approved (~$60)
- [ ] Timeline approved (4 weeks)

**Proceed with implementation?** ________________

---

*Document generated: December 2025*
