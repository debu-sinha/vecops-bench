# VectorDB-Bench

**Production-Oriented Benchmarking for Vector Database Systems**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17924957.svg)](https://doi.org/10.5281/zenodo.17924957)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> A comprehensive benchmark suite for evaluating vector databases beyond recall-latency trade-offs. Introduces novel production-relevant metrics: **cold start latency**, **operational complexity**, and **filtered search overhead**.

---

## Key Findings

Our evaluation of 5 vector databases on MS MARCO 100K reveals:

| Database | Recall@10 | p50 Latency | QPS | Cold Start | Insert/s |
|----------|-----------|-------------|-----|------------|----------|
| **Milvus** | 0.537 | 3.86ms | 101 | 17ms | 10,279 |
| **Qdrant** | 0.537 | 5.27ms | 309 | 70ms | 1,411 |
| **pgvector** | 0.545 | 3.74ms | 398 | **14ms** | 164 |
| **Chroma** | 0.537 | 4.42ms | 324 | 65ms | 1,744 |
| **Weaviate** | 0.537 | 4.49ms | **436** | 109ms | 2,911 |

### Surprising Results

- **8x cold start variation**: pgvector (14ms) vs Weaviate (109ms)
- **100x filtered search overhead variation**: pgvector (-31%) vs Chroma (+2,978%)
- **No universal winner**: Each database excels in different dimensions

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/debu-sinha/vectordb-bench.git
cd vectordb-bench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run benchmark (single database)
python scripts/run_benchmark.py --config experiments/config.yaml --database pgvector

# Run all databases
python scripts/run_benchmark.py --config experiments/config.yaml
```

---

## Databases Evaluated

| Database | Version | Index Type | Architecture |
|----------|---------|------------|--------------|
| [Milvus](https://milvus.io/) | 2.3.x | IVF_FLAT, HNSW | Distributed, cloud-native |
| [Qdrant](https://qdrant.tech/) | 1.7.x | HNSW | Rust-based, single/cluster |
| [pgvector](https://github.com/pgvector/pgvector) | 0.5.x | IVFFlat, HNSW | PostgreSQL extension |
| [Weaviate](https://weaviate.io/) | 1.22.x | HNSW | GraphQL-native, modular |
| [Chroma](https://www.trychroma.com/) | 0.4.x | HNSW | Embedded-first, Python-native |

---

## Metrics

### Standard Metrics
- **Recall@k**: Fraction of relevant documents in top-k results
- **Latency**: p50, p95, p99 query latency
- **QPS**: Sustained queries per second under load

### Novel Production Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Cold Start Latency** | Time from container start to first successful query | Critical for serverless/Lambda deployments |
| **Operational Complexity** | Measurable deployment artifacts (services, config params, metrics) | Impacts maintenance burden and TCO |
| **Filtered Search Overhead** | Latency change when adding metadata filters | Most real queries combine vectors + filters |
| **Insert Throughput** | Vectors indexed per second during bulk load | Important for data ingestion pipelines |

---

## Results Deep Dive

### Cold Start Performance

```
pgvector   ████████████████ 14.3ms (fastest)
Milvus     █████████████████ 17.0ms
Chroma     ██████████████████████████████████████████████████████████████████ 65.2ms
Qdrant     ███████████████████████████████████████████████████████████████████████ 69.5ms
Weaviate   ████████████████████████████████████████████████████████████████████████████████████████████████████████████ 109.2ms (slowest)
```

### Operational Complexity

| Database | Required Services | Config Params | Docker Images | Prometheus Metrics | Score |
|----------|-------------------|---------------|---------------|-------------------|-------|
| Qdrant | 1 | 12 | 1 | 42 | **8.9** (simplest) |
| Weaviate | 1 | 23 | 1 | 67 | 24.5 |
| pgvector | 1 | 8 | 1 | 156 | 27.5 |
| Milvus | 4 | 47 | 3 | 89 | 40.3 (most complex) |
| Chroma | 1 | 6 | 1 | 12 | 43.8 |

### Filtered Search Overhead

| Database | Overhead | Notes |
|----------|----------|-------|
| pgvector | **-31%** | *Faster with filters* (PostgreSQL optimizer) |
| Qdrant | +347% | Pre-filtering approach |
| Chroma | +2,978% | Falls back to full scan |

---

## Recommendations by Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Serverless/Lambda** | pgvector | Fastest cold start (14ms) |
| **High Throughput** | Weaviate | Highest QPS (436) |
| **Minimal Ops** | Qdrant | Lowest complexity score (8.9) |
| **Bulk Ingestion** | Milvus | Fastest insert (10K vec/s) |
| **Existing PostgreSQL** | pgvector | Seamless integration |
| **Rapid Prototyping** | Chroma | Simplest API |

---

## Reproducibility

### Hardware Configuration
- **Instance**: AWS c5.2xlarge (8 vCPU, 16GB RAM)
- **Deployment**: Docker containers with pinned versions
- **Trials**: 5 runs per configuration
- **Warm-up**: 100 queries before measurement

### Dataset
- **MS MARCO Passage Ranking** (100K subset)
- **Embeddings**: sentence-transformers/all-mpnet-base-v2 (768 dimensions)

### Run Full Benchmark

```bash
# Start all databases
docker-compose up -d

# Run complete benchmark suite
python scripts/run_benchmark.py --config experiments/full_config.yaml

# Generate figures
python scripts/generate_figures.py --results results/

# Results saved to results/ directory
```

---

## Project Structure

```
vectordb-bench/
├── src/
│   ├── databases/          # Database adapters (Milvus, Qdrant, etc.)
│   ├── metrics/            # Metric computation modules
│   ├── workloads/          # Query workload generators
│   └── utils/              # Utilities and helpers
├── experiments/            # YAML experiment configurations
├── scripts/                # Benchmark runner scripts
├── results/                # Raw benchmark results (JSON)
├── paper/                  # LaTeX paper source
│   ├── paper_outline.tex   # Main paper
│   ├── references.bib      # Bibliography
│   └── figures/            # Publication figures
└── paper_figures/          # Generated visualizations
```

---

## Citation

If you use VectorDB-Bench in your research, please cite:

```bibtex
@software{sinha2025vectordbbench,
  author       = {Sinha, Debu},
  title        = {{VectorDB-Bench: A Production-Oriented Benchmark
                   Suite for Vector Database Systems}},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17924957},
  url          = {https://doi.org/10.5281/zenodo.17924957}
}
```

**Paper:** [VectorDB-Bench on Zenodo](https://zenodo.org/records/17924957)

---

## Contributing

We welcome contributions! Areas of interest:

- [ ] Additional database adapters (Pinecone, Elasticsearch, OpenSearch)
- [ ] Distributed/cluster mode benchmarks
- [ ] Larger scale experiments (1M+ vectors)
- [ ] Cost-per-query modeling for cloud deployments
- [ ] Hybrid search (dense + sparse) evaluation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- MS MARCO dataset from Microsoft Research
- BEIR benchmark framework
- sentence-transformers library

---

**Questions?** Open an [issue](https://github.com/debu-sinha/vectordb-bench/issues) or start a [discussion](https://github.com/debu-sinha/vectordb-bench/discussions).
