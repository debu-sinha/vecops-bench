# VecOps-Bench v2.0 - Industrial Strength Testing Plan

**Target Venues**: SIGMOD, VLDB, ICDE
**Novel Contribution**: First benchmark to measure temporal drift in vector databases
**Core Thesis**: "Current benchmarks measure Day 1 speed; VecOps measures Day 2 survival."
**Last Updated**: 2025-12-15

---

## Executive Summary

This benchmark differentiates from existing work (ann-benchmarks, VectorDBBench) by introducing metrics that matter for **production deployments**:

1. **Temporal Drift Analysis** - How performance degrades over time (the "Day 2" problem)
2. **Cost Efficiency** - Recall-per-Dollar-Hour for business decisions
3. **Operational Resilience** - Recovery time, crash survival, memory pressure

---

## Part 1: Infrastructure Requirements

### Hardware Specification

| Component | Minimum | Recommended | Current |
|-----------|---------|-------------|---------|
| RAM | 64 GB | 128 GB | **123 GB** ✓ |
| vCPUs | 8 | 16 | **16** ✓ |
| Storage | 100 GB SSD | 500 GB SSD | 3.8 TB ✓ |
| Instance | r6i.2xlarge | r6i.4xlarge | Equivalent ✓ |

**The Math**:
- 10M vectors × 768 dimensions × 4 bytes = **30 GB raw data**
- With HNSW index overhead: **45-50 GB required**
- Current 123 GB provides comfortable headroom

---

## Part 2: Databases Under Test

| Database | Version | Index Type | Architecture |
|----------|---------|------------|--------------|
| FAISS | 1.13.1 | HNSW | In-memory library (baseline) |
| Milvus | 2.3.4 | HNSW | Distributed, production-grade |
| Qdrant | 1.16.0 | HNSW | Rust-based, single-node |
| pgvector | 0.5+ | HNSW | PostgreSQL extension |
| Weaviate | 1.27.0 | HNSW | GraphQL-native |
| Chroma | 0.6.3 | HNSW | AI-native, simple API |

### Standardized HNSW Parameters (Critical for Fair Comparison)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M | 16 | Industry standard |
| ef_construction | 128 | Balanced build time vs quality |
| ef_search | 100 | Balanced recall vs latency |
| Metric | Cosine | Standard for text embeddings |

---

## Part 3: Phase 1 - Scale Baseline (10M Vectors)

**Goal**: Establish credibility. Prove these systems work at a size that matters.

### Dataset
- **Source**: Cohere/Wikipedia-22-12-en-embeddings (HuggingFace)
- **Size**: 10,000,000 vectors
- **Dimensions**: 768
- **Type**: Real embeddings (NOT random)

### Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| Index Build Time | Time to ingest 10M vectors | seconds |
| Peak Memory Usage | Max RAM during build | GB |
| Ingestion Rate | Vectors per second | vec/s |
| Query Latency p50 | Median latency | ms |
| Query Latency p95 | 95th percentile | ms |
| Query Latency p99 | 99th percentile | ms |
| QPS | Maximum sustained throughput | queries/s |
| Recall@10 | Accuracy vs brute force | ratio |
| Recall@100 | Extended accuracy | ratio |
| Filtered Search | Latency with metadata filter | ms |

---

## Part 4: Phase 2 - Temporal Drift (The "Star" Feature)

**Goal**: The Novelty. Show how performance rots over time.

### Methodology

1. **Start** with the 10M loaded database (from Phase 1)
2. **Churn Cycle**: Delete 100,000 vectors + Insert 100,000 new vectors
3. **Repeat**: 10 cycles (1M total churned = 10% turnover)
4. **Measure**: Recall@10 and P99 Latency **after each cycle**

### Drift Scenarios

| Scenario | Name | Delete Rate | Add Rate | Use Case |
|----------|------|-------------|----------|----------|
| Archivist | STABLE | 0% | 1% | Append-only archives |
| Newsroom | HIGH_CHURN | 8% | 10% | Fast-moving content |
| Catastrophe | CATASTROPHIC | 30% | 30% | Major corpus overhaul |

### Expected Output

**Line Graph**: X-axis = Churn Cycles (0-10), Y-axis = Recall@10

**Hypothesis to Prove**:
> "HNSW graphs degrade under churn. Systems with background vacuuming (pgvector)
> may remain stable, while others see recall drop from 95% to 80%."

**This finding is the key novel contribution of the paper.**

---

## Part 5: Phase 3 - Cost Efficiency (The "Manager" Metric)

**Goal**: Business relevance for enterprise architects.

### Primary Metric: Recall-per-Dollar-Hour

```
Score = Recall@10 / Hourly Infrastructure Cost
```

### Example Comparison

| System | Recall@10 | Hourly Cost | Score | Verdict |
|--------|-----------|-------------|-------|---------|
| System A | 0.95 | $2.00/hr | 0.475 | High accuracy, high cost |
| System B | 0.90 | $0.50/hr | 1.800 | "Production Winner" |

### Additional Cost Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| QPS per Dollar | QPS / Cost | Throughput efficiency |
| Latency-Cost Product | p50 × cost | Lower = better value |
| Pareto Frontier | Multi-objective | Optimal tradeoffs |

---

## Part 6: Phase 4 - Operational Resilience

**Goal**: Quantify the "pain" of production operations.

### Test 1: Hard Kill Recovery (TTR)

**Methodology**:
1. Run DB at 500 QPS sustained load
2. Issue `docker kill -9 <container>`
3. Restart container
4. Measure: Time until API returns successful search (TTR)

**Hypothesis**:
> "In-memory DBs (Chroma/Milvus) need to reload 30GB from disk (slow).
> Postgres (pgvector) recovers instantly due to disk-based page buffers."

### Test 2: Memory Pressure (OOM)

**Methodology**:
1. Restrict container to 80% of index size
2. Run query workload
3. Measure: Does it crash, swap, or degrade gracefully?

### Test 3: Traffic Spike

**Methodology**:
1. Run at baseline 100 QPS
2. Spike to 1000 QPS (10x)
3. Measure: Latency jitter and error rate

---

## Part 7: Statistical Rigor

| Requirement | Implementation |
|-------------|----------------|
| Trials | Minimum 3 per experiment |
| Variance | Report σ (standard deviation) |
| Significance | Cohen's d for comparisons |
| Confidence | 95% CI on all metrics |

---

## Part 8: Paper Structure

### Suggested Title
**"VecOps: Benchmarking the Long-Term Operational Stability of Vector Databases"**

### Section Outline

1. **Abstract**
   - Hook: "Current benchmarks focus on Day 1 speed. We introduce VecOps for Day 2 survival."
   - Key stat: "Under 10% churn, Recall in System X degrades by 15%..."

2. **Introduction**
   - The "Production Gap": Why 100k tests are dangerous
   - Introduce VecOps Framework: Drift, Cost, Resilience

3. **Experimental Setup**
   - Hardware: "Memory-optimized instances (128GB RAM)"
   - Dataset: "10M vectors from Cohere/Wikipedia"

4. **The Hidden Cost of Churn (Drift Results)** ← Main Section
   - Line graph: Recall degradation over churn cycles
   - Discovery: Which DBs degrade vs maintain stability

5. **Economics of Scale (Cost Results)**
   - Pareto Frontier chart
   - "Best Buy" for different budgets

6. **Failure Modes (Resilience)**
   - TTR table
   - Tradeoffs: Disk-backed vs RAM-only

7. **Conclusion**
   - Practitioner guidance: "Use X for speed, Y for stability, Z for cost"

---

## Part 9: Execution Timeline

| Day | Task | Status |
|-----|------|--------|
| Day 1 | Phase 1: 10M baseline load + standard metrics | 🔄 Running |
| Day 2 | Phase 2: Churn test on pgvector (validate harness) | Pending |
| Day 3 | Phase 2: Full drift suite (all DBs) | Pending |
| Day 4 | Phase 3: Cost analysis | Pending |
| Day 5 | Phase 4: Resilience tests | Pending |
| Day 6 | Analysis + figure generation | Pending |

---

## Part 10: Files and Scripts

| File | Purpose |
|------|---------|
| `scripts/run_benchmark_v2.py` | Phase 1 baseline benchmark |
| `scripts/run_churn_test.py` | Phase 2 drift/churn test |
| `src/drift/temporal_drift.py` | Drift simulation engine |
| `src/cost/analyzer.py` | Cost modeling |
| `src/resilience/crash_recovery.py` | TTR measurement |
| `src/resilience/memory_pressure.py` | OOM testing |

---

## Part 11: Success Criteria

Success Criteria:

- [ ] Demonstrate drift degradation that no prior paper has shown
- [ ] Publish at SIGMOD/VLDB/ICDE (acceptance = peer validation)
- [ ] Provide actionable guidance for practitioners
- [ ] Open-source benchmark for community adoption

---

*Internal document - VecOps-Bench v2.0*
