#!/bin/bash
# =============================================================================
# VectorDB-Bench v2.0 - EC2 Setup for 100M Scale
# Instance: i4i.4xlarge (16 vCPU, 128 GB RAM, 3.75 TB NVMe)
# Cost: ~$1.25/hour
# =============================================================================

set -e

echo "=============================================="
echo "VectorDB-Bench v2.0 - 100M Scale Setup"
echo "Instance: i4i.4xlarge"
echo "=============================================="

# =============================================================================
# NVMe Setup (CRITICAL for 100M scale)
# =============================================================================
echo ""
echo "[1/9] Setting up NVMe storage..."

# Find NVMe devices
NVME_DEVICE=$(lsblk -d -o NAME,SIZE | grep nvme | grep -v nvme0n1 | head -1 | awk '{print $1}')

if [ -n "$NVME_DEVICE" ]; then
    echo "Found NVMe: /dev/$NVME_DEVICE"

    # Format if not already formatted
    if ! blkid /dev/$NVME_DEVICE > /dev/null 2>&1; then
        echo "Formatting NVMe..."
        sudo mkfs.ext4 -F /dev/$NVME_DEVICE
    fi

    # Mount to /data
    sudo mkdir -p /data
    sudo mount /dev/$NVME_DEVICE /data
    sudo chown ubuntu:ubuntu /data

    # Add to fstab for persistence
    echo "/dev/$NVME_DEVICE /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

    echo "NVMe mounted at /data"
    df -h /data
else
    echo "WARNING: No NVMe found, using EBS (slower)"
    sudo mkdir -p /data
    sudo chown ubuntu:ubuntu /data
fi

# Create data directories
mkdir -p /data/{milvus,qdrant,pgvector,chroma,weaviate,elasticsearch,etcd,minio}

# =============================================================================
# System Tuning for High Performance
# =============================================================================
echo ""
echo "[2/9] Tuning system parameters..."

# Increase file descriptors
echo "* soft nofile 65535" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65535" | sudo tee -a /etc/security/limits.conf

# Increase max memory map count (required for Elasticsearch)
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf

# Disable swap (prefer OOM kill over slow swap)
sudo swapoff -a

# Apply sysctl changes
sudo sysctl -p

# =============================================================================
# Docker Installation
# =============================================================================
echo ""
echo "[3/9] Installing Docker..."

if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker ubuntu
    rm get-docker.sh
fi

# Install Docker Compose v2
sudo apt-get update -qq
sudo apt-get install -y docker-compose-plugin

# Configure Docker to use NVMe
sudo mkdir -p /etc/docker
cat << EOF | sudo tee /etc/docker/daemon.json
{
    "data-root": "/data/docker",
    "storage-driver": "overlay2",
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "5"
    }
}
EOF

sudo systemctl restart docker
sudo systemctl enable docker

# =============================================================================
# Python Environment
# =============================================================================
echo ""
echo "[4/9] Setting up Python 3.11..."

sudo apt-get install -y python3.11 python3.11-venv python3-pip

cd /home/ubuntu
python3.11 -m venv vectordb-bench-env
source vectordb-bench-env/bin/activate

# =============================================================================
# Clone Repository
# =============================================================================
echo ""
echo "[5/9] Cloning repository..."

if [ ! -d "vectordb-bench" ]; then
    git clone https://github.com/debu-sinha/vectordb-bench.git
fi
cd vectordb-bench
git pull origin main

# =============================================================================
# Python Dependencies (Optimized for CPU)
# =============================================================================
echo ""
echo "[6/9] Installing Python dependencies..."

pip install --upgrade pip wheel setuptools

# Core (minimal memory footprint)
pip install \
    numpy>=1.26.0 \
    scipy>=1.11.0 \
    pyyaml>=6.0 \
    tqdm>=4.66.0 \
    psutil>=5.9.0 \
    requests>=2.31.0

# Stats
pip install \
    pandas>=2.1.0 \
    scikit-learn>=1.3.0 \
    statsmodels>=0.14.0

# DB clients
pip install \
    pymilvus>=2.4.0 \
    qdrant-client>=1.7.0 \
    chromadb>=0.4.0 \
    weaviate-client>=4.4.0 \
    pgvector>=0.2.4 \
    psycopg2-binary>=2.9.9 \
    elasticsearch>=8.12.0

# Faiss (CPU only - no GPU needed)
pip install faiss-cpu>=1.7.4

# Streaming datasets (for 100M scale)
pip install \
    datasets>=2.16.0 \
    pyarrow>=14.0.0

# PyTorch CPU (for embeddings if needed)
pip install torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers>=2.2.0

# Docker SDK
pip install docker>=7.0.0

# Visualization (lightweight)
pip install matplotlib>=3.8.0 seaborn>=0.13.0

# =============================================================================
# Pull Docker Images
# =============================================================================
echo ""
echo "[7/9] Pulling Docker images (this takes a while)..."

docker pull milvusdb/milvus:v2.3.4 &
docker pull qdrant/qdrant:v1.7.4 &
docker pull pgvector/pgvector:pg16 &
docker pull chromadb/chroma:0.4.22 &
docker pull semitechnologies/weaviate:1.23.7 &
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.0 &
docker pull quay.io/coreos/etcd:v3.5.11 &
docker pull minio/minio:RELEASE.2024-01-01T16-36-33Z &

wait
echo "All images pulled."

# =============================================================================
# Start Services
# =============================================================================
echo ""
echo "[8/9] Starting database services..."

cd /home/ubuntu/vectordb-bench
docker compose -f infrastructure/docker-compose-100m.yaml up -d

# Wait for services to be healthy
echo "Waiting for services to initialize (this may take 2-3 minutes)..."
sleep 120

# Check health
docker ps --format "table {{.Names}}\t{{.Status}}"

# =============================================================================
# Create tmux session
# =============================================================================
echo ""
echo "[9/9] Creating tmux session..."

tmux new-session -d -s benchmark -c /home/ubuntu/vectordb-bench
tmux send-keys -t benchmark "source ~/vectordb-bench-env/bin/activate" Enter

# =============================================================================
# Verification
# =============================================================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "System Info:"
echo "  CPU: $(nproc) cores"
echo "  RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "  NVMe: $(df -h /data | awk 'NR==2 {print $2}')"
echo ""
echo "Docker Services:"
docker ps --format "  {{.Names}}: {{.Status}}"
echo ""
echo "Quick Start:"
echo "  1. tmux attach -t benchmark"
echo "  2. python scripts/run_benchmark_v2.py --phase validation"
echo ""
echo "Estimated times:"
echo "  - Validation (10M): ~8 hours"
echo "  - Production (100M): ~72-96 hours"
echo ""
echo "Cost tracker:"
echo "  Instance: \$1.25/hour"
echo "  Validation: ~\$10"
echo "  Production: ~\$120"
echo ""
echo "IMPORTANT: Stop instance when done to save costs!"
echo "  aws ec2 stop-instances --instance-ids \$(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo ""
echo "=============================================="

# Save completion marker
date -u +"%Y-%m-%dT%H:%M:%SZ" > /home/ubuntu/.vectordb-bench-v2-100m-setup-complete
