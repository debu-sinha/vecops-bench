#!/bin/bash
# =============================================================================
# VectorDB-Bench: Complete Paper Experiments
# =============================================================================
# This script runs all experiments needed for a rigorous academic paper:
# 1. Multiple trials (5 runs) for statistical rigor
# 2. Multiple dataset sizes (100K, 1M vectors)
# 3. Multiple datasets (MS MARCO, NFCorpus, SciFact)
# 4. All 5 databases (Milvus, Qdrant, pgvector, Weaviate, Chroma)
#
# Estimated runtime: 8-10 hours on c5.2xlarge
# =============================================================================

set +e  # Continue on error (individual benchmark failures are logged)

# Configuration
NUM_TRIALS=5
DATABASES="milvus qdrant pgvector weaviate chroma"
RESULTS_DIR="./results/paper_experiments"
LOG_DIR="./logs"
CONFIG_FILE="./experiments/config.yaml"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Activate virtual environment
source venv/bin/activate

echo "=============================================="
echo "VectorDB-Bench: Paper Experiments"
echo "=============================================="
echo "Start time: $(date)"
echo "Trials per experiment: $NUM_TRIALS"
echo "Databases: $DATABASES"
echo "=============================================="

# =============================================================================
# PHASE 1: Multiple Trials on 100K vectors (Statistical Rigor)
# =============================================================================
echo ""
echo "PHASE 1: Running $NUM_TRIALS trials on 100K vectors..."
echo "=============================================="

for trial in $(seq 1 $NUM_TRIALS); do
    echo ""
    echo "=== Trial $trial of $NUM_TRIALS ==="
    TRIAL_DIR="$RESULTS_DIR/100k_trial_$trial"
    mkdir -p "$TRIAL_DIR"

    # Run benchmark for all databases
    python scripts/run_benchmark.py \
        --config "$CONFIG_FILE" \
        --database $DATABASES \
        --output "$TRIAL_DIR" \
        2>&1 | tee "$LOG_DIR/100k_trial_${trial}.log"

    echo "Trial $trial complete. Results in $TRIAL_DIR"
done

echo ""
echo "PHASE 1 Complete: $NUM_TRIALS trials on 100K vectors"

# =============================================================================
# PHASE 2: Scale Testing on 1M vectors
# =============================================================================
echo ""
echo "PHASE 2: Running benchmarks on 1M vectors..."
echo "=============================================="

# Create 1M config
cat > ./experiments/config_1m.yaml << 'EOF'
experiment:
  name: "vectordb-bench-1m"
  description: "Scale testing with 1M vectors"
  random_seed: 42

databases:
  milvus:
    enabled: true
    config:
      host: "localhost"
      port: 19530
      deployment: "docker"

  qdrant:
    enabled: true
    config:
      host: "localhost"
      port: 6333
      deployment: "docker"

  pgvector:
    enabled: true
    config:
      host: "localhost"
      port: 5432
      database: "vectordb_bench"
      user: "postgres"
      password: "postgres"

  chroma:
    enabled: true
    config:
      host: "localhost"
      port: 8000
      deployment: "server"

  weaviate:
    enabled: true
    config:
      host: "localhost"
      port: 8080
      deployment: "docker"

datasets:
  - name: "msmarco-passage"
    source: "beir"
    num_docs: 1000000
    dimensions: 768

embedding_model:
  name: "sentence-transformers/all-mpnet-base-v2"
  dimensions: 768
  batch_size: 32

benchmark:
  recall_k: [1, 10, 100]
  num_queries: 1000
  qps_duration_seconds: 60
  cold_start_trials: 5
  filtered_search_enabled: true
  hybrid_search_enabled: true

output:
  results_dir: "./results"
  save_latencies: true
  save_plots: true
  export_formats: ["json", "csv"]
EOF

SCALE_DIR="$RESULTS_DIR/1m_scale"
mkdir -p "$SCALE_DIR"

for trial in $(seq 1 3); do
    echo ""
    echo "=== 1M Scale Trial $trial of 3 ==="

    python scripts/run_benchmark.py \
        --config ./experiments/config_1m.yaml \
        --database $DATABASES \
        --output "$SCALE_DIR/trial_$trial" \
        2>&1 | tee "$LOG_DIR/1m_trial_${trial}.log"
done

echo ""
echo "PHASE 2 Complete: Scale testing on 1M vectors"

# =============================================================================
# PHASE 3: Additional Datasets (NFCorpus, SciFact)
# =============================================================================
echo ""
echo "PHASE 3: Running benchmarks on additional datasets..."
echo "=============================================="

# NFCorpus config
cat > ./experiments/config_nfcorpus.yaml << 'EOF'
experiment:
  name: "vectordb-bench-nfcorpus"
  description: "NFCorpus dataset benchmarks"
  random_seed: 42

databases:
  milvus:
    enabled: true
    config:
      host: "localhost"
      port: 19530
      deployment: "docker"

  qdrant:
    enabled: true
    config:
      host: "localhost"
      port: 6333
      deployment: "docker"

  pgvector:
    enabled: true
    config:
      host: "localhost"
      port: 5432
      database: "vectordb_bench"
      user: "postgres"
      password: "postgres"

  chroma:
    enabled: true
    config:
      host: "localhost"
      port: 8000
      deployment: "server"

  weaviate:
    enabled: true
    config:
      host: "localhost"
      port: 8080
      deployment: "docker"

datasets:
  - name: "nfcorpus"
    source: "beir"
    num_docs: null
    dimensions: 768

embedding_model:
  name: "sentence-transformers/all-mpnet-base-v2"
  dimensions: 768
  batch_size: 32

benchmark:
  recall_k: [1, 10, 100]
  num_queries: 500
  qps_duration_seconds: 30
  cold_start_trials: 5
  filtered_search_enabled: true

output:
  results_dir: "./results"
  save_latencies: true
  export_formats: ["json"]
EOF

# SciFact config
cat > ./experiments/config_scifact.yaml << 'EOF'
experiment:
  name: "vectordb-bench-scifact"
  description: "SciFact dataset benchmarks"
  random_seed: 42

databases:
  milvus:
    enabled: true
    config:
      host: "localhost"
      port: 19530
      deployment: "docker"

  qdrant:
    enabled: true
    config:
      host: "localhost"
      port: 6333
      deployment: "docker"

  pgvector:
    enabled: true
    config:
      host: "localhost"
      port: 5432
      database: "vectordb_bench"
      user: "postgres"
      password: "postgres"

  chroma:
    enabled: true
    config:
      host: "localhost"
      port: 8000
      deployment: "server"

  weaviate:
    enabled: true
    config:
      host: "localhost"
      port: 8080
      deployment: "docker"

datasets:
  - name: "scifact"
    source: "beir"
    num_docs: null
    dimensions: 768

embedding_model:
  name: "sentence-transformers/all-mpnet-base-v2"
  dimensions: 768
  batch_size: 32

benchmark:
  recall_k: [1, 10, 100]
  num_queries: 300
  qps_duration_seconds: 30
  cold_start_trials: 5
  filtered_search_enabled: true

output:
  results_dir: "./results"
  save_latencies: true
  export_formats: ["json"]
EOF

# Run NFCorpus
echo ""
echo "=== NFCorpus Dataset ==="
NFCORPUS_DIR="$RESULTS_DIR/nfcorpus"
mkdir -p "$NFCORPUS_DIR"

for trial in $(seq 1 3); do
    echo "NFCorpus Trial $trial of 3"
    python scripts/run_benchmark.py \
        --config ./experiments/config_nfcorpus.yaml \
        --database $DATABASES \
        --output "$NFCORPUS_DIR/trial_$trial" \
        2>&1 | tee "$LOG_DIR/nfcorpus_trial_${trial}.log"
done

# Run SciFact
echo ""
echo "=== SciFact Dataset ==="
SCIFACT_DIR="$RESULTS_DIR/scifact"
mkdir -p "$SCIFACT_DIR"

for trial in $(seq 1 3); do
    echo "SciFact Trial $trial of 3"
    python scripts/run_benchmark.py \
        --config ./experiments/config_scifact.yaml \
        --database $DATABASES \
        --output "$SCIFACT_DIR/trial_$trial" \
        2>&1 | tee "$LOG_DIR/scifact_trial_${trial}.log"
done

echo ""
echo "PHASE 3 Complete: Additional datasets"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Logs saved to: $LOG_DIR"
echo ""
echo "Directory structure:"
find "$RESULTS_DIR" -type d | head -20
echo ""
echo "Total result files:"
find "$RESULTS_DIR" -name "*.json" | wc -l
echo ""
echo "Next steps:"
echo "1. Run: python scripts/analyze_results.py --input $RESULTS_DIR"
echo "2. Generate visualizations for paper"
echo "=============================================="
