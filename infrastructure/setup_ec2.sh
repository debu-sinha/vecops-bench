#!/bin/bash
# =============================================================================
# VectorDB-Bench: EC2 Instance Setup Script
# =============================================================================
#
# Recommended Instance: c5.2xlarge (compute-optimized)
#   - 8 vCPUs (Intel Xeon Platinum 8275CL @ 3.0GHz)
#   - 16 GB RAM
#   - Up to 10 Gbps network
#   - Cost: ~$0.34/hour (~$8/day)
#
# Alternative (budget): c5.xlarge
#   - 4 vCPUs, 8 GB RAM
#   - Cost: ~$0.17/hour (~$4/day)
#
# Estimated total cost for full experiments: $30-50
#
# Usage:
#   1. Launch EC2 instance with Ubuntu 22.04 LTS AMI
#   2. SSH into instance
#   3. Run: curl -sSL <this-script-url> | bash
#   OR
#   4. Copy this script and run: chmod +x setup_ec2.sh && ./setup_ec2.sh
#
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "VectorDB-Bench EC2 Setup"
echo "=============================================="

# Record system info for reproducibility
echo ""
echo "[1/8] Recording system information..."
mkdir -p ~/vectordb-bench/logs
cat > ~/vectordb-bench/logs/system_info.txt << EOF
=== System Information ===
Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Hostname: $(hostname)
Instance ID: $(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "N/A")
Instance Type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "N/A")
Region: $(curl -s http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo "N/A")
AMI ID: $(curl -s http://169.254.169.254/latest/meta-data/ami-id 2>/dev/null || echo "N/A")

=== Hardware ===
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
CPU Cores: $(nproc)
RAM: $(free -h | grep Mem | awk '{print $2}')
Disk: $(df -h / | tail -1 | awk '{print $2}')

=== OS ===
$(cat /etc/os-release | grep -E "^(NAME|VERSION)=")
Kernel: $(uname -r)
EOF
cat ~/vectordb-bench/logs/system_info.txt

# Update system
echo ""
echo "[2/8] Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq

# Install Docker
echo ""
echo "[3/8] Installing Docker..."
sudo apt-get install -y -qq apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose standalone (for compatibility)
echo ""
echo "[4/8] Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python 3.11
echo ""
echo "[5/8] Installing Python 3.11..."
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip

# Clone repository
echo ""
echo "[6/8] Setting up VectorDB-Bench..."
cd ~
if [ -d "vectordb-bench" ]; then
    cd vectordb-bench
    git pull
else
    git clone https://github.com/debu-sinha/vectordb-bench.git
    cd vectordb-bench
fi

# Create virtual environment
echo ""
echo "[7/8] Creating Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies with pinned versions
pip install --upgrade pip
pip install -r requirements.txt

# Record versions
pip freeze > ~/vectordb-bench/logs/pip_freeze.txt

# Docker version info
echo ""
echo "[8/8] Recording Docker versions..."
cat >> ~/vectordb-bench/logs/system_info.txt << EOF

=== Docker ===
$(docker --version)
$(docker-compose --version)

=== Python ===
$(python --version)
Pip packages: See pip_freeze.txt
EOF

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Log out and back in (for Docker group)"
echo "  2. cd ~/vectordb-bench"
echo "  3. source venv/bin/activate"
echo "  4. docker-compose up -d"
echo "  5. python scripts/run_benchmark.py --config experiments/config.yaml --full"
echo ""
echo "System info saved to: ~/vectordb-bench/logs/system_info.txt"
echo ""
