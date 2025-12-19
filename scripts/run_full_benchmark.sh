#!/bin/bash
# =============================================================================
# VecOps-Bench: Full Sequential Benchmark Runner
# =============================================================================
#
# Runs all benchmarks sequentially to avoid OOM on 128GB RAM instance.
# Tests one DB at a time, stopping containers between tests.
#
# Usage:
#   ./scripts/run_full_benchmark.sh
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
SCALE=9990000
NUM_QUERIES=5000
DATA_DIR="data/recall_test"
RESULTS_DIR="results_v2"
LOG_FILE="benchmark_full_$(date +%Y%m%d_%H%M%S).log"

# Databases to test (order: fastest to slowest expected)
DATABASES=("pgvector" "qdrant" "milvus" "weaviate" "chroma")

echo "=============================================="
echo "VecOps-Bench Full Benchmark"
echo "Scale: $SCALE vectors"
echo "Queries: $NUM_QUERIES"
echo "Log: $LOG_FILE"
echo "=============================================="

# Activate virtual environment
source venv/bin/activate

# Step 1: Prepare data (held-out queries)
echo ""
echo "[Step 1/4] Preparing recall test data..."
if [ ! -f "$DATA_DIR/corpus.memmap.shape.npy" ]; then
    python scripts/prepare_recall_data.py \
        --scale $SCALE \
        --held-out 10000 \
        --output-dir "$DATA_DIR" \
        2>&1 | tee -a "$LOG_FILE"
else
    echo "  Data already prepared, skipping..."
fi

# Step 2: Stop all containers and clear old data
echo ""
echo "[Step 2/4] Cleaning up containers..."
docker-compose down -v 2>/dev/null || true
sleep 5

# Step 3: Run benchmark for each database
echo ""
echo "[Step 3/4] Running benchmarks..."

for db in "${DATABASES[@]}"; do
    echo ""
    echo "=============================================="
    echo "TESTING: $db"
    echo "Time: $(date)"
    echo "=============================================="

    # Start required containers
    case $db in
        milvus)
            echo "  Starting Milvus stack..."
            docker-compose up -d milvus-etcd milvus-minio milvus
            ;;
        pgvector)
            echo "  Starting PostgreSQL..."
            docker-compose up -d postgres
            ;;
        *)
            echo "  Starting $db..."
            docker-compose up -d $db
            ;;
    esac

    # Wait for container to be healthy
    echo "  Waiting for container to be ready..."
    sleep 30

    # Run baseline benchmark (ingestion + latency + QPS)
    echo "  Running baseline benchmark..."
    python scripts/run_benchmark_v2.py \
        --database $db \
        --scale $SCALE \
        --phase validation \
        --use-real-embeddings \
        --output "$RESULTS_DIR/baseline" \
        2>&1 | tee -a "$LOG_FILE"

    # Run recall benchmark (with proper methodology)
    echo "  Running recall benchmark..."
    python scripts/recall_fix.py \
        --database $db \
        --data-dir "$DATA_DIR" \
        --output "$RESULTS_DIR/recall" \
        2>&1 | tee -a "$LOG_FILE"

    # Run filtered search benchmark
    echo "  Running filtered search..."
    python scripts/run_filtered_search.py \
        --database $db \
        --output "$RESULTS_DIR/filtered" \
        2>&1 | tee -a "$LOG_FILE"

    # Run churn test (drift analysis)
    echo "  Running churn/drift test..."
    python scripts/run_churn_test_v2.py \
        --database $db \
        --data-dir "$DATA_DIR" \
        --output "$RESULTS_DIR/churn" \
        --cycles 10 \
        --churn-size 100000 \
        2>&1 | tee -a "$LOG_FILE"

    # Stop containers to free memory
    echo "  Stopping containers..."
    case $db in
        milvus)
            docker-compose stop milvus milvus-minio milvus-etcd
            ;;
        pgvector)
            docker-compose stop postgres
            ;;
        *)
            docker-compose stop $db
            ;;
    esac

    # Clear Docker volumes for this DB to free disk space
    echo "  Clearing volumes..."
    docker volume rm $(docker volume ls -q | grep $db) 2>/dev/null || true

    echo "  $db complete!"
    sleep 10
done

# Step 4: Run FAISS baseline (in-memory, no container)
echo ""
echo "=============================================="
echo "TESTING: FAISS (speed of light baseline)"
echo "=============================================="

# Baseline benchmark
echo "  Running baseline benchmark..."
python scripts/run_benchmark_v2.py \
    --database faiss \
    --scale $SCALE \
    --phase validation \
    --use-real-embeddings \
    --output "$RESULTS_DIR/baseline" \
    2>&1 | tee -a "$LOG_FILE"

# Recall benchmark (provides theoretical maximum recall reference)
echo "  Running recall benchmark..."
python scripts/recall_fix.py \
    --database faiss \
    --data-dir "$DATA_DIR" \
    --output "$RESULTS_DIR/recall" \
    2>&1 | tee -a "$LOG_FILE"

echo "  FAISS complete!"

# Summary
echo ""
echo "=============================================="
echo "BENCHMARK COMPLETE"
echo "Results in: $RESULTS_DIR/"
echo "Log: $LOG_FILE"
echo "=============================================="

ls -la "$RESULTS_DIR"/*/*.json 2>/dev/null || echo "No results found"
