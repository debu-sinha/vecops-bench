# VecOps-Bench

**Measuring the Hidden Cost of Data Churn in Production Vector Databases**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Key Finding**: All tested HNSW indexes degrade 3.4%–9.7% after just 10% data churn, with 38× speed differences in churn operations.

---

## The Production Gap

Current benchmarks (ANN-Benchmarks, VectorDBBench) evaluate **Day 1 performance**:
- Fresh index, static corpus, synthetic queries

Real production systems face **Day 2 challenges**:
- Continuous updates (documents added/removed)
- Index degradation from data churn
- Compaction overhead and timing
- Memory pressure during operations

**VecOps-Bench measures what happens when production reality meets HNSW theory.**

---

## Key Results (December 2025)

### Temporal Drift Under 10% Data Churn

| Database | Initial Recall@10 | After 10 Cycles | Degradation |
|----------|-------------------|-----------------|-------------|
| pgvector | 83.13% | 75.06% | **9.71%** |
| Milvus | 97.84% | 89.28% | **8.75%** |
| Chroma | 88.99% | 81.61% | **8.29%** |
| Weaviate | 81.80% | 79.06% | **3.35%** |

### Churn Operation Speed

| Database | Avg Delete (100K) | Avg Insert (100K) | Full Cycle |
|----------|-------------------|-------------------|------------|
| Milvus | 22.9s | 14.3s | **37s** |
| Weaviate | 94.9s | 94.9s | 190s |
| pgvector | 87.5s | 1307.8s | **1395s** |
| Chroma | 671.5s | 781.8s | 1453s |

**38× speed difference** between fastest (Milvus) and slowest (pgvector) churn operations.

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- 64GB+ RAM recommended for 10M scale

### Run Benchmark

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

# Run baseline benchmark
python scripts/run_benchmark_v2.py --scale 1000000

# Run temporal drift analysis (THE KEY EXPERIMENT)
python scripts/run_churn_test_v2.py --database milvus --cycles 10
```

---

## Methodology

### Dataset
- **Primary**: Cohere Wikipedia embeddings (9.99M × 768 dimensions)
- **Validation**: SIFT1M (1M × 128 dimensions)

### HNSW Parameters (Standardized)
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M | 16 | Industry standard |
| ef_construction | 128 | Balanced build/quality |
| ef_search | 100 | Balanced recall/latency |

### Churn Protocol
1. Load 9.99M vectors + 10K held-out queries
2. Each cycle: DELETE 100K oldest + INSERT 100K new
3. Measure Recall@10 after each cycle
4. 10 cycles = 10% total corpus turnover

---

## Databases Tested

| Database | Version | Status |
|----------|---------|--------|
| [Milvus](https://milvus.io/) | 2.3.4 | ✅ Complete |
| [pgvector](https://github.com/pgvector/pgvector) | 0.8.0 | ✅ Complete |
| [Weaviate](https://weaviate.io/) | 1.27.0 | ✅ Complete |
| [Chroma](https://www.trychroma.com/) | 0.6.3 | ✅ Complete |
| [Qdrant](https://qdrant.tech/) | 1.7.4 | ⚠️ Delete timeouts |

---

## Project Structure

```
vecops-bench/
├── src/
│   ├── databases/          # Database adapters
│   ├── metrics/            # Recall computation
│   └── resilience/         # Cold start, crash recovery
├── scripts/
│   ├── run_benchmark_v2.py # Baseline benchmark
│   ├── run_churn_test_v2.py # Temporal drift analysis
│   ├── run_sift1m_*.py     # SIFT1M validation
│   └── generate_vldb_figures.py # Publication figures
├── paper/
│   ├── vecops_bench_vldb2026.tex # VLDB 2026 submission
│   ├── figures/            # Publication figures
│   └── CLAIMS_VALIDATION.md # Data verification
├── results_v2/             # Experimental results (JSON)
└── docker-compose.yaml     # Database containers
```

---

## Reproducing Results

### Generate Publication Figures

```bash
python scripts/generate_vldb_figures.py
# Output: paper/figures/fig1_recall_degradation.pdf (etc.)
```

### Validate Paper Claims

All claims in the paper are validated against experimental JSON files:
- See `paper/CLAIMS_VALIDATION.md` for full verification

---

## Hardware

**AWS EC2 Instance**:
- Type: Memory-optimized (i4i.4xlarge or similar)
- RAM: 128GB
- vCPUs: 16
- Storage: NVMe SSD

---

## Citation

```bibtex
@article{sinha2025vecops,
  title={{VecOps-Bench}: Measuring the Hidden Cost of Data Churn
         in Production Vector Databases},
  author={Sinha, Debu},
  journal={Proceedings of the VLDB Endowment},
  year={2026},
  note={Under review}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

- **Author**: Debu Sinha
- **GitHub**: [@debu-sinha](https://github.com/debu-sinha)
- **Issues**: [GitHub Issues](https://github.com/debu-sinha/vecops-bench/issues)
