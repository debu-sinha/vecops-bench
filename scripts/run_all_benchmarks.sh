#!/bin/bash
# =============================================================================
# VectorDB-Bench: Complete Benchmark Automation Script
# =============================================================================
#
# This script runs the entire benchmark suite with proper methodology:
# - 5 runs for statistical significance
# - Cache clearing between runs
# - All novel metrics (drift, cost, operational complexity)
# - Automatic analysis and report generation
#
# Usage:
#   chmod +x scripts/run_all_benchmarks.sh
#   ./scripts/run_all_benchmarks.sh
#
# Estimated time: 4-6 hours
# Estimated cost: ~$1 (Spot) / ~$2 (On-Demand)
# =============================================================================

set -e  # Exit on error

# Configuration
NUM_RUNS=5
RESULTS_DIR="./results"
ANALYSIS_DIR="./analysis"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "=============================================="
echo "  VectorDB-Bench: Full Benchmark Suite"
echo "=============================================="
echo -e "${NC}"

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
echo -e "${YELLOW}[Pre-flight] Checking environment...${NC}"

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate || { echo -e "${RED}Failed to activate venv${NC}"; exit 1; }
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose not found.${NC}"
    exit 1
fi

# Record system info
echo -e "${YELLOW}[Pre-flight] Recording system information...${NC}"
mkdir -p "$RESULTS_DIR/metadata"

cat > "$RESULTS_DIR/metadata/system_info_$TIMESTAMP.txt" << EOF
=== VectorDB-Bench System Information ===
Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Hostname: $(hostname)

=== Hardware ===
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs 2>/dev/null || echo "Unknown")
CPU Cores: $(nproc 2>/dev/null || echo "Unknown")
RAM: $(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "Unknown")

=== Software ===
OS: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d'"' -f2 || echo "Unknown")
Kernel: $(uname -r)
Docker: $(docker --version 2>/dev/null || echo "Unknown")
Python: $(python --version 2>&1)

=== Docker Images ===
$(docker images --format "{{.Repository}}:{{.Tag}}" | grep -E "qdrant|milvus|weaviate|pgvector|chroma" || echo "None found")
EOF

echo "System info saved to: $RESULTS_DIR/metadata/system_info_$TIMESTAMP.txt"

# -----------------------------------------------------------------------------
# Start databases
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[Setup] Starting database containers...${NC}"

docker-compose up -d

echo "Waiting 90 seconds for databases to initialize..."
sleep 90

# Check health
echo -e "${YELLOW}[Setup] Checking database health...${NC}"
docker-compose ps

# Verify each database is responding
echo "Verifying database connectivity..."
HEALTHY=true

# Qdrant
if curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Qdrant: OK"
else
    echo -e "  ${RED}✗${NC} Qdrant: FAILED"
    HEALTHY=false
fi

# Weaviate
if curl -s http://localhost:8080/v1/.well-known/ready > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Weaviate: OK"
else
    echo -e "  ${RED}✗${NC} Weaviate: FAILED"
    HEALTHY=false
fi

# Chroma
if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Chroma: OK"
else
    echo -e "  ${RED}✗${NC} Chroma: FAILED"
    HEALTHY=false
fi

# PostgreSQL
if docker exec vectordb-bench-postgres pg_isready -U postgres > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} PostgreSQL: OK"
else
    echo -e "  ${RED}✗${NC} PostgreSQL: FAILED"
    HEALTHY=false
fi

# Milvus
if curl -s http://localhost:9091/healthz > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Milvus: OK"
else
    echo -e "  ${RED}✗${NC} Milvus: FAILED (may need more time)"
fi

if [ "$HEALTHY" = false ]; then
    echo -e "${RED}Some databases failed health checks. Continue anyway? (y/n)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=============================================="
echo "  Starting Benchmark Runs ($NUM_RUNS total)"
echo "==============================================${NC}"

START_TIME=$(date +%s)

for run in $(seq 1 $NUM_RUNS); do
    RUN_DIR="$RESULTS_DIR/run_$run"
    mkdir -p "$RUN_DIR"

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Run $run of $NUM_RUNS${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Clear filesystem caches (requires sudo)
    echo "Clearing filesystem caches..."
    sync
    if sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        echo "  Cache cleared"
    else
        echo "  Cache clear skipped (no sudo)"
    fi

    # Restart containers for fair cold-start measurement
    echo "Restarting containers for cold-start fairness..."
    docker-compose restart
    sleep 60

    # Run the benchmark
    echo "Running benchmark..."
    RUN_START=$(date +%s)

    python scripts/run_benchmark.py \
        --config experiments/config.yaml \
        --full \
        --output "$RUN_DIR" \
        2>&1 | tee "$RUN_DIR/benchmark_log.txt"

    RUN_END=$(date +%s)
    RUN_DURATION=$((RUN_END - RUN_START))

    echo ""
    echo -e "${GREEN}Run $run completed in $((RUN_DURATION / 60)) minutes${NC}"

    # Brief pause between runs
    if [ $run -lt $NUM_RUNS ]; then
        echo "Pausing 30 seconds before next run..."
        sleep 30
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}=============================================="
echo "  All $NUM_RUNS runs completed!"
echo "  Total time: $((TOTAL_DURATION / 60)) minutes"
echo "==============================================${NC}"

# -----------------------------------------------------------------------------
# Aggregate and analyze results
# -----------------------------------------------------------------------------
echo ""
echo -e "${YELLOW}[Analysis] Generating reports and visualizations...${NC}"

mkdir -p "$ANALYSIS_DIR"

python scripts/analyze_results.py \
    --results "$RESULTS_DIR" \
    --output "$ANALYSIS_DIR"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=============================================="
echo "  BENCHMARK COMPLETE"
echo "==============================================${NC}"
echo ""
echo "Results saved to:"
echo "  - Raw data: $RESULTS_DIR/"
echo "  - Analysis: $ANALYSIS_DIR/"
echo ""
echo "Key files:"
ls -la "$ANALYSIS_DIR"/*.png 2>/dev/null | head -10
echo ""
echo "To view the summary report:"
echo "  cat $ANALYSIS_DIR/summary_report.md"
echo ""
echo -e "${YELLOW}REMINDER: Stop your EC2 instance to avoid charges!${NC}"
echo ""

# Record completion
echo "Benchmark completed at $(date)" >> "$RESULTS_DIR/metadata/completion_$TIMESTAMP.txt"
echo "Total duration: $((TOTAL_DURATION / 60)) minutes" >> "$RESULTS_DIR/metadata/completion_$TIMESTAMP.txt"
