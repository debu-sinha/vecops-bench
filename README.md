# VecOps-Bench

**Beyond QPS: Benchmarking Vector Databases for Production Resilience**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17924957.svg)](https://doi.org/10.5281/zenodo.17924957)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/debu-sinha/vecops-bench?style=social)](https://github.com/debu-sinha/vecops-bench)
[![GitHub forks](https://img.shields.io/github/forks/debu-sinha/vecops-bench?style=social)](https://github.com/debu-sinha/vecops-bench/fork)
[![Downloads](https://img.shields.io/github/downloads/debu-sinha/vecops-bench/total)](https://github.com/debu-sinha/vecops-bench/releases)
[![CI](https://github.com/debu-sinha/vecops-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/debu-sinha/vecops-bench/actions)

> A production-oriented benchmark suite for vector databases focusing on **Day-2 Operations**: cold start latency, crash recovery, operational complexity, and filtered search overhead.

---

## What Makes VecOps-Bench Different?

| Metric | ANN-Benchmarks | VectorDBBench (Zilliz) | **VecOps-Bench** |
|--------|----------------|------------------------|------------------|
| Recall/QPS | Yes | Yes | Yes |
| Cold Start (TTFQ) | No | No | **Yes** |
| Crash Recovery | No | No | **Yes** |
| Config Friction | No | No | **Yes** |
| Filtered Search | Limited | Limited | **Yes (1%, 10%, 50%)** |

---

## Quick Start

### Reproduce Published Results (Zenodo Paper)

```bash
# Clone repository
git clone https://github.com/debu-sinha/vecops-bench.git
cd vecops-bench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start databases
docker-compose up -d

# Wait for healthy status
docker-compose ps

# Run quick validation (100K vectors, ~30 min)
python scripts/run_benchmark_v2.py --phase quick

# Results saved to results_v2/quick/
```

### Run Production Scale (10M vectors)

```bash
# Run 10M scale benchmark (~5-6 hours)
python scripts/run_benchmark_v2.py --scale 10000000

# Or run specific database only
python scripts/run_benchmark_v2.py --scale 10000000 --database qdrant pgvector
```

---

## Databases Evaluated

| Database | Version | Index | Notes |
|----------|---------|-------|-------|
| [FAISS](https://github.com/facebookresearch/faiss) | CPU | HNSW | In-memory baseline |
| [Elasticsearch](https://www.elastic.co/) | 8.11.0 | HNSW (Lucene) | Full-text + vector |
| [Milvus](https://milvus.io/) | 2.3.4 | HNSW | Distributed |
| [Qdrant](https://qdrant.tech/) | 1.16.0 | HNSW | Rust-based |
| [pgvector](https://github.com/pgvector/pgvector) | **0.8.0+** | HNSW | PostgreSQL extension |
| [Chroma](https://www.trychroma.com/) | 0.6.3 | HNSW | Lightweight |
| [Weaviate](https://weaviate.io/) | 1.27.0 | HNSW | GraphQL-native |

---

## Metrics

### Standard Metrics
- **Recall@10**: Accuracy vs brute-force ground truth
- **QPS**: Queries per second (single & concurrent)
- **Latency**: p50, p95, p99 response times

### Novel Production Metrics

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| **Cold Start (TTFQ)** | Container restart to first query | Serverless/auto-scaling |
| **Crash Recovery** | Kill -9 to data integrity | SRE reliability |
| **Config Friction** | YAML lines + Image size | Operational complexity |
| **Filtered Search** | Overhead at 1%, 10%, 50% selectivity | Real-world queries |

---

## Infrastructure

### Recommended Hardware
- **Instance**: AWS i4i.4xlarge (16 vCPU, 128GB RAM, NVMe)
- **Cost**: ~$1.37/hour
- **OS**: Ubuntu 22.04+

### Docker Configuration

All databases run in Docker with:
- Pinned versions for reproducibility
- Health checks for readiness detection
- Memory limits for fair comparison

```bash
# Start all databases
docker-compose up -d

# Check health
docker-compose ps

# View logs
docker-compose logs -f qdrant
```

---

## Project Structure

```
vecops-bench/
├── src/
│   ├── databases/          # Database adapters
│   ├── metrics/            # Metric computation
│   ├── resilience/         # Cold start, crash recovery
│   ├── operational/        # Runtime complexity prober
│   └── stats/              # Statistical analysis
├── scripts/
│   ├── run_benchmark_v2.py # Main benchmark runner
│   └── generate_figures.py # Visualization
├── docker-compose.yaml     # Database containers
├── results_v2/             # Benchmark results (JSON)
└── paper/                  # LaTeX paper source
```

---

## Results Format

Each database produces a JSON file with:

```json
{
  "database": "qdrant",
  "scale": 10000000,
  "recall_at_10": 0.892,
  "qps": 1250.5,
  "p50_ms": 0.78,
  "p95_ms": 1.23,
  "cold_start_ms": 2450.3,
  "crash_recovery_ms": 3200.1,
  "filtered_search": {
    "selectivity_1pct": 1.15,
    "selectivity_10pct": 1.08,
    "selectivity_50pct": 0.95
  }
}
```

---

## Citation

If you use VecOps-Bench in your research, please cite:

```bibtex
@software{sinha2025vecopsbench,
  author       = {Sinha, Debu},
  title        = {{VecOps-Bench: A Day-2 Operations Benchmark
                   for Vector Database Systems}},
  month        = dec,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17924957},
  url          = {https://doi.org/10.5281/zenodo.17924957}
}
```

---

## Related Work

- [ANN-Benchmarks](https://ann-benchmarks.com/) - Algorithm-level recall/QPS
- [VectorDBBench](https://github.com/zilliztech/VectorDBBench) - Zilliz benchmark
- [BEIR](https://github.com/beir-cellar/beir) - Information retrieval benchmark

---

## Contributing

We welcome contributions:
- Additional database adapters
- New operational metrics
- Larger scale experiments
- Cloud cost modeling

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Questions?** Open an [issue](https://github.com/debu-sinha/vecops-bench/issues)
