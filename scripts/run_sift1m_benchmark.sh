#!/bin/bash
# =============================================================================
# VecOps-Bench: SIFT1M Benchmark Runner
# =============================================================================
#
# Runs SIFT1M benchmarks for cross-domain validation (image descriptors).
# This complements the Cohere Wikipedia benchmark (text embeddings).
#
# SIFT1M Dataset:
#   - 1,000,000 base vectors (128 dimensions)
#   - 10,000 query vectors (held out)
#   - 100 ground truth neighbors per query (pre-computed by INRIA)
#
# Why SIFT1M?
#   - Classic benchmark, enables direct comparison with ann-benchmarks
#   - Different domain (image vs text) for generalization
#   - Different dimensionality (128 vs 768) tests scalability
#   - Pre-computed ground truth (gold standard)
#
# Usage:
#   ./scripts/run_sift1m_benchmark.sh
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
SCALE=1000000  # SIFT1M has exactly 1M vectors
NUM_QUERIES=10000
DATA_DIR="data/sift1m"
RESULTS_DIR="results_v2/sift1m"
LOG_FILE="benchmark_sift1m_$(date +%Y%m%d_%H%M%S).log"

# Databases to test (same as Cohere benchmark)
DATABASES=("pgvector" "qdrant" "milvus" "weaviate" "chroma")

echo "=============================================="
echo "VecOps-Bench SIFT1M Benchmark"
echo "Scale: $SCALE vectors (128 dimensions)"
echo "Queries: $NUM_QUERIES (held out)"
echo "Log: $LOG_FILE"
echo "=============================================="

# Activate virtual environment
source venv/bin/activate

# Step 1: Download and prepare SIFT1M data
echo ""
echo "[Step 1/4] Preparing SIFT1M data..."
if [ ! -f "$DATA_DIR/corpus.memmap.shape.npy" ]; then
    python scripts/prepare_sift1m.py \
        2>&1 | tee -a "$LOG_FILE"
else
    echo "  SIFT1M data already prepared, skipping..."
fi

# Verify data files exist
if [ ! -f "$DATA_DIR/corpus.memmap" ] || [ ! -f "$DATA_DIR/queries.memmap" ] || [ ! -f "$DATA_DIR/ground_truth.npy" ]; then
    echo "ERROR: SIFT1M data files missing. Run prepare_sift1m.py first."
    exit 1
fi

echo "  SIFT1M data ready:"
echo "    - Corpus: $(ls -lh $DATA_DIR/corpus.memmap | awk '{print $5}')"
echo "    - Queries: $(ls -lh $DATA_DIR/queries.memmap | awk '{print $5}')"
echo "    - Ground truth: $(ls -lh $DATA_DIR/ground_truth.npy | awk '{print $5}')"

# Step 2: Stop all containers and clear old data
echo ""
echo "[Step 2/4] Cleaning up containers..."
docker-compose down -v 2>/dev/null || true
sleep 5

# Step 3: Run benchmark for each database
echo ""
echo "[Step 3/4] Running SIFT1M benchmarks..."
mkdir -p "$RESULTS_DIR/baseline" "$RESULTS_DIR/recall"

for db in "${DATABASES[@]}"; do
    echo ""
    echo "=============================================="
    echo "TESTING: $db (SIFT1M)"
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

    # Run SIFT1M ingestion + baseline metrics
    echo "  Running SIFT1M baseline benchmark..."
    python scripts/run_sift1m_baseline.py \
        --database $db \
        --data-dir "$DATA_DIR" \
        --output "$RESULTS_DIR/baseline" \
        2>&1 | tee -a "$LOG_FILE"

    # Run SIFT1M recall benchmark (uses pre-computed ground truth)
    echo "  Running SIFT1M recall benchmark..."
    python scripts/run_sift1m_recall.py \
        --database $db \
        --data-dir "$DATA_DIR" \
        --output "$RESULTS_DIR/recall" \
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

    echo "  $db SIFT1M complete!"
    sleep 10
done

# Step 4: Run FAISS baseline (in-memory, no container)
echo ""
echo "=============================================="
echo "TESTING: FAISS (SIFT1M speed of light baseline)"
echo "=============================================="

# FAISS baseline + recall
echo "  Running FAISS SIFT1M benchmark..."
python scripts/run_sift1m_baseline.py \
    --database faiss \
    --data-dir "$DATA_DIR" \
    --output "$RESULTS_DIR/baseline" \
    2>&1 | tee -a "$LOG_FILE"

python scripts/run_sift1m_recall.py \
    --database faiss \
    --data-dir "$DATA_DIR" \
    --output "$RESULTS_DIR/recall" \
    2>&1 | tee -a "$LOG_FILE"

echo "  FAISS SIFT1M complete!"

# Summary
echo ""
echo "=============================================="
echo "SIFT1M BENCHMARK COMPLETE"
echo "Results in: $RESULTS_DIR/"
echo "Log: $LOG_FILE"
echo "=============================================="

ls -la "$RESULTS_DIR"/*/*.json 2>/dev/null || echo "No results found"

echo ""
echo "Next steps:"
echo "  1. Compare with ann-benchmarks results"
echo "  2. Generate cross-dataset analysis figures"
echo "  3. Verify generalization across domains"
