#!/bin/bash
# =============================================================================
# VectorDB-Bench v2.0 - EC2 Setup Script for r6i.4xlarge
# Instance: r6i.4xlarge (16 vCPU, 128 GB RAM, ~$1.00/hour)
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "VectorDB-Bench v2.0 - EC2 Setup"
echo "Instance Type: r6i.4xlarge"
echo "Target Scale: 10M vectors"
echo "=============================================="

# Record system info
echo "Recording system information..."
cat > /tmp/system_info.txt << EOF
Hostname: $(hostname)
Date: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
Region: $(curl -s http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "unknown")
CPU: $(nproc) cores
Memory: $(free -h | awk '/^Mem:/ {print $2}')
Disk: $(df -h / | awk 'NR==2 {print $2}')
Kernel: $(uname -r)
EOF
cat /tmp/system_info.txt

# =============================================================================
# System Updates
# =============================================================================
echo ""
echo "[1/8] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# =============================================================================
# Docker Installation
# =============================================================================
echo ""
echo "[2/8] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    rm get-docker.sh
fi

# Install Docker Compose v2
echo "Installing Docker Compose v2..."
sudo apt-get install -y docker-compose-plugin

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# =============================================================================
# Python Environment
# =============================================================================
echo ""
echo "[3/8] Setting up Python 3.11..."
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
cd /home/ubuntu
python3.11 -m venv vectordb-bench-env
source vectordb-bench-env/bin/activate

# =============================================================================
# Clone Repository
# =============================================================================
echo ""
echo "[4/8] Cloning VectorDB-Bench repository..."
if [ ! -d "vectordb-bench" ]; then
    git clone https://github.com/debu-sinha/vectordb-bench.git
fi
cd vectordb-bench

# =============================================================================
# Python Dependencies
# =============================================================================
echo ""
echo "[5/8] Installing Python dependencies..."
pip install --upgrade pip wheel setuptools

# Core dependencies
pip install \
    numpy>=1.26.0 \
    pandas>=2.1.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.66.0 \
    pyyaml>=6.0 \
    psutil>=5.9.0 \
    docker>=7.0.0

# ML/Embeddings (CPU only for cost efficiency)
pip install \
    torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    sentence-transformers>=2.2.0 \
    datasets>=2.16.0

# Vector DB clients
pip install \
    pymilvus>=2.4.0 \
    qdrant-client>=1.7.0 \
    chromadb>=0.4.0 \
    weaviate-client>=4.4.0 \
    pgvector>=0.2.4 \
    psycopg2-binary>=2.9.9 \
    elasticsearch>=8.12.0

# NEW: Faiss for baseline (CPU version for r6i)
pip install faiss-cpu>=1.7.4

# Visualization
pip install \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0

# Statistics
pip install \
    statsmodels>=0.14.0 \
    scipy>=1.11.0

# =============================================================================
# Docker Images (Pre-pull to save time during benchmark)
# =============================================================================
echo ""
echo "[6/8] Pre-pulling Docker images..."

# Use specific versions for reproducibility
sudo docker pull milvusdb/milvus:v2.3.4 &
sudo docker pull qdrant/qdrant:v1.7.4 &
sudo docker pull pgvector/pgvector:pg16 &
sudo docker pull chromadb/chroma:0.4.22 &
sudo docker pull semitechnologies/weaviate:1.23.7 &
sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.0 &
sudo docker pull quay.io/coreos/etcd:v3.5.11 &
sudo docker pull minio/minio:RELEASE.2024-01-01T16-36-33Z &

wait  # Wait for all pulls to complete
echo "Docker images pulled successfully."

# =============================================================================
# Storage Setup (for 10M vectors)
# =============================================================================
echo ""
echo "[7/8] Setting up storage..."

# Create data directories
mkdir -p /home/ubuntu/vectordb-bench/data/{milvus,qdrant,pgvector,chroma,weaviate,elasticsearch,cache}
mkdir -p /home/ubuntu/vectordb-bench/results_v2

# Set permissions
sudo chown -R ubuntu:ubuntu /home/ubuntu/vectordb-bench

# Check available disk space
echo "Available disk space:"
df -h /home/ubuntu

# =============================================================================
# Verify Installation
# =============================================================================
echo ""
echo "[8/8] Verifying installation..."

# Check Python
python --version

# Check key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import faiss; print(f'Faiss: OK')"
python -c "import pymilvus; print(f'PyMilvus: {pymilvus.__version__}')"
python -c "import qdrant_client; print(f'Qdrant: OK')"
python -c "import chromadb; print(f'Chroma: {chromadb.__version__}')"
python -c "import weaviate; print(f'Weaviate: OK')"
python -c "import elasticsearch; print(f'Elasticsearch: {elasticsearch.__version__}')"

# Check Docker
docker --version
docker compose version

# =============================================================================
# Create tmux session
# =============================================================================
echo ""
echo "Creating tmux session 'benchmark'..."
tmux new-session -d -s benchmark -c /home/ubuntu/vectordb-bench

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Quick Start:"
echo "  1. tmux attach -t benchmark"
echo "  2. source ~/vectordb-bench-env/bin/activate"
echo "  3. docker compose up -d"
echo "  4. python scripts/run_benchmark_v2.py --phase validation"
echo ""
echo "Estimated benchmark time:"
echo "  - Validation (1M): ~4-6 hours"
echo "  - Production (10M): ~24-48 hours"
echo ""
echo "Estimated cost at \$1.00/hour:"
echo "  - Validation: ~\$6"
echo "  - Production: ~\$48"
echo ""
echo "=============================================="

# Save setup completion marker
echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > /home/ubuntu/.vectordb-bench-v2-setup-complete
