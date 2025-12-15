# VectorDB-Bench: Standardized Benchmarking Protocol

This document defines the reproducible benchmarking protocol for VectorDB-Bench, following best practices from systems research papers at MLSys, VLDB, and SIGMOD.

## Hardware Specification

### Primary Test Environment

| Component | Specification |
|-----------|---------------|
| **Instance Type** | AWS EC2 c5.2xlarge |
| **CPU** | Intel Xeon Platinum 8275CL @ 3.0GHz |
| **vCPUs** | 8 |
| **RAM** | 16 GB |
| **Storage** | 100 GB gp3 SSD (3000 IOPS) |
| **Network** | Up to 10 Gbps |
| **Region** | us-east-1 |
| **OS** | Ubuntu 22.04 LTS |

### Why c5.2xlarge?

1. **Compute-optimized**: Vector operations are CPU-intensive
2. **Sufficient RAM**: 16GB handles datasets up to 1M vectors
3. **Reproducible**: Widely available, consistent performance
4. **Cost-effective**: ~$0.34/hour, full experiments in ~$30-50

### Alternative Configurations

| Budget | Instance | vCPUs | RAM | Cost/hr |
|--------|----------|-------|-----|---------|
| Minimum | c5.xlarge | 4 | 8 GB | $0.17 |
| **Recommended** | **c5.2xlarge** | **8** | **16 GB** | **$0.34** |
| Extended | c5.4xlarge | 16 | 32 GB | $0.68 |

## Software Versions (Pinned)

All software versions are pinned for reproducibility:

```yaml
# Docker Images (pinned tags)
databases:
  qdrant: qdrant/qdrant:v1.7.4
  milvus: milvusdb/milvus:v2.3.4
  weaviate: semitechnologies/weaviate:1.23.0
  postgres: pgvector/pgvector:pg16
  chroma: chromadb/chroma:0.4.22

# Supporting Services
infrastructure:
  etcd: quay.io/coreos/etcd:v3.5.11
  minio: minio/minio:RELEASE.2024-01-01T16-36-33Z
```

```
# Python Dependencies (see requirements.txt)
python==3.11.x
numpy==1.26.3
sentence-transformers==2.2.2
torch==2.1.2
# ... (full list in requirements.txt)
```

## Experiment Protocol

### Pre-Benchmark Checklist

- [ ] Fresh EC2 instance launched
- [ ] Setup script executed successfully
- [ ] All Docker containers healthy
- [ ] System idle (no background processes)
- [ ] Sufficient disk space (>50GB free)

### Warm-up Procedure

Before each benchmark run:

1. **Start databases**: `docker-compose up -d`
2. **Wait for healthy**: 60 seconds minimum
3. **Verify connectivity**: Run health checks
4. **Clear caches**: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`

### Benchmark Execution

```bash
# Standard benchmark (5 runs for statistical significance)
for run in {1..5}; do
    echo "Run $run of 5"
    python scripts/run_benchmark.py \
        --config experiments/config.yaml \
        --output results/run_$run \
        --full

    # Cool-down between runs
    sleep 60
done

# Aggregate results
python scripts/aggregate_runs.py --input results/ --output results/aggregated/
```

### Between Database Tests

1. Stop current database: `docker-compose stop <db>`
2. Clear Docker volumes: `docker volume prune -f`
3. Drop filesystem caches
4. Wait 30 seconds
5. Start next database

## Statistical Rigor

### Number of Runs

| Metric Type | Minimum Runs | Report |
|-------------|--------------|--------|
| Latency (p50, p95, p99) | 5 | Mean ± std |
| Recall/Precision | 3 | Mean ± std |
| QPS | 5 | Mean ± std |
| Cold Start | 10 | Mean ± std |
| Index Build | 3 | Mean ± std |

### Outlier Handling

- Remove runs with >3σ deviation
- Document any removed outliers
- Report both with and without outliers if significant

### Significance Testing

For comparing databases:
- Use Wilcoxon signed-rank test (non-parametric)
- Report p-values for key claims
- Threshold: p < 0.05

## Managed Services (Pinecone)

Pinecone is evaluated separately due to architectural differences:

### Configuration

```yaml
pinecone:
  tier: serverless
  region: us-east-1  # Same region as EC2
  cloud: aws
  metric: cosine
```

### Caveats Documented in Paper

1. **Network latency**: Results include ~5-20ms network RTT
2. **Shared infrastructure**: Performance may vary
3. **Not directly comparable**: Latency vs self-hosted

### Fair Comparison Metrics

| Metric | Fair to Compare? | Notes |
|--------|------------------|-------|
| Recall@k | Yes | Algorithm-dependent |
| Latency | No* | Network overhead |
| QPS | Partial | Client-side measurement |
| Cost | Yes | Primary advantage |
| Cold Start | No | Different model |

*Reported separately with caveats

## Datasets

### Standard Datasets

| Dataset | Documents | Queries | Domain |
|---------|-----------|---------|--------|
| SciFact | 5,183 | 300 | Scientific claims |
| NFCorpus | 3,633 | 323 | Medical/nutrition |
| MS MARCO (subset) | 100,000 | 6,980 | Web passages |

### Embedding Model

```
Model: sentence-transformers/all-mpnet-base-v2
Dimensions: 768
Pooling: Mean
Normalization: L2
```

### Data Preparation

1. Download from BEIR/MTEB
2. Generate embeddings (cached)
3. Verify checksums
4. Document any preprocessing

## Reproducibility Checklist

For paper submission:

- [ ] Hardware specs documented
- [ ] All software versions pinned
- [ ] Docker images use specific tags (not :latest)
- [ ] Random seeds fixed (42)
- [ ] Setup script tested on fresh instance
- [ ] Results include error bars
- [ ] Outliers documented
- [ ] Caveats for managed services noted
- [ ] Code repository public
- [ ] Data download scripts included

## Cost Estimation

| Phase | Duration | Cost |
|-------|----------|------|
| Setup & debugging | 2 hours | $0.68 |
| Full benchmark (5 runs) | 8 hours | $2.72 |
| Extended tests | 4 hours | $1.36 |
| Buffer | 4 hours | $1.36 |
| **Total EC2** | **18 hours** | **~$6** |
| Pinecone API | - | $0-20 |
| **Grand Total** | | **$10-30** |

*Costs are estimates. Actual may vary.*

## Reporting Results

### Required Tables

1. **Table 1**: Hardware & software environment
2. **Table 2**: Standard metrics (recall, latency, QPS)
3. **Table 3**: Production metrics (cold start, filtered search)
4. **Table 4**: Novel metrics (drift, cost, operational complexity)

### Required Figures

1. Recall vs Latency Pareto frontier
2. Cold start comparison (box plots)
3. Temporal drift curves
4. Cost-efficiency Pareto
5. Operational complexity radar

### LaTeX Template

```latex
\begin{table}[t]
\caption{Experimental Environment}
\label{tab:environment}
\centering
\begin{tabular}{ll}
\toprule
Component & Specification \\
\midrule
Instance & AWS EC2 c5.2xlarge \\
CPU & Intel Xeon Platinum 8275CL (8 vCPU) \\
Memory & 16 GB \\
Storage & 100 GB gp3 SSD \\
OS & Ubuntu 22.04 LTS \\
Region & us-east-1 \\
\bottomrule
\end{tabular}
\end{table}
```

## References

- [SIGMOD Reproducibility Guidelines](https://reproducibility.sigmod.org/)
- [MLSys Artifact Evaluation](https://mlsys.org/Conferences/2024/CallForPapers)
- [VLDB Experiments Guidelines](https://vldb.org/pvldb/vol14/experiments.html)
