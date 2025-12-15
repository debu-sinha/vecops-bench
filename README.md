# VectorDB-Bench: Production-Oriented Vector Database Evaluation

**Target Venue:** MLSys 2026 / SIGMOD 2026
**Status:** Planning
**Author:** Debu Sinha

## Abstract

A comprehensive benchmark for vector databases focusing on production-realistic evaluation dimensions beyond recall/latency: hybrid search accuracy, metadata filtering, cold-start performance, operational complexity, and cost modeling.

## Databases Under Evaluation

| Database | Deployment | Free Tier |
|----------|------------|-----------|
| Pinecone | Cloud | Yes (starter) |
| Milvus | Self-hosted (Docker) | Yes |
| Qdrant | Cloud + Self-hosted | Yes |
| pgvector | Self-hosted (PostgreSQL) | Yes |
| Chroma | Self-hosted | Yes |
| Weaviate | Cloud + Self-hosted | Yes |

## Evaluation Dimensions

### Standard Metrics (Baseline)
- Recall@k (k=1, 10, 100)
- Queries per second (QPS)
- Latency (p50, p95, p99)

### Production Metrics (Novel)
- **Hybrid Search Accuracy**: Dense + sparse retrieval performance
- **Filtered Search Latency**: Metadata filtering impact on performance
- **Cold-Start Time**: Time to first query after index load
- **Index Build Cost**: Time and memory for index construction
- **Operational Complexity Score**: Deployment, monitoring, backup difficulty
- **Cost per 1M Queries**: Normalized cost comparison

## Datasets

- MTEB Retrieval Tasks
- BEIR Benchmark
- MS MARCO Passages
- Custom synthetic workloads

## Project Structure

```
vectordb-bench/
├── src/                    # Core benchmarking code
│   ├── databases/          # Database-specific adapters
│   ├── metrics/            # Metric computation
│   ├── workloads/          # Workload generators
│   └── utils/              # Utilities
├── experiments/            # Experiment configurations
├── data/                   # Downloaded datasets
├── results/                # Benchmark results
├── paper/                  # LaTeX paper draft
└── scripts/                # Setup and run scripts
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run benchmarks
python scripts/run_benchmark.py --config experiments/config.yaml
```

## Timeline

- [ ] Set up database adapters
- [ ] Implement standard metrics
- [ ] Implement production metrics
- [ ] Run baseline experiments
- [ ] Analyze results
- [ ] Write paper draft

## Budget

Estimated cost: $100-200 (cloud overflow only)

## License

MIT
