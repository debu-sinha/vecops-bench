# VectorDB-Bench: Step-by-Step Execution Guide

## Quick Start (3 Commands)

```powershell
# Step 1: Setup (one time)
cd C:\Users\dsinh\research\vectordb-bench
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# Step 2: Test locally (5 minutes)
python scripts/test_local.py

# Step 3: If tests pass, proceed to EC2
```

---

## Phase 1: Local Testing (Your PC, ~10 min, $0)

### 1.1 Open PowerShell and setup environment

```powershell
cd C:\Users\dsinh\research\vectordb-bench

# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate

# Install dependencies (may take 5-10 min)
pip install -r requirements.txt
```

### 1.2 Run local tests

```powershell
python scripts/test_local.py
```

**Expected output:**
```
╔════════════════════════════════════════════════════════════╗
║  ✓ ALL TESTS PASSED - Ready for EC2 deployment!           ║
╚════════════════════════════════════════════════════════════╝
```

If any test fails, fix the issue before proceeding.

---

## Phase 2: AWS EC2 Setup (~30 min, ~$0.15)

### 2.1 Launch EC2 Spot Instance

1. Go to: https://console.aws.amazon.com/ec2
2. Click **Launch Instance**
3. Fill in:

| Setting | Value |
|---------|-------|
| Name | `vectordb-bench` |
| AMI | Ubuntu 22.04 LTS (64-bit x86) |
| Instance type | `c5.2xlarge` |
| Key pair | Create new or select existing |

4. **Enable Spot Instance** (saves 55%):
   - Expand **Advanced details**
   - Scroll to **Purchasing option**
   - Check **Request Spot Instances**

5. **Storage**: Change to 100 GB

6. Click **Launch instance**

### 2.2 Connect via SSH

```powershell
# Replace with your key file and EC2 public IP
ssh -i "C:\path\to\your-key.pem" ubuntu@<EC2-PUBLIC-IP>
```

### 2.3 Run setup script

```bash
# On the EC2 instance:
sudo apt-get update && sudo apt-get install -y git curl

# Clone the repo (or copy from your machine)
git clone https://github.com/YOUR_USERNAME/vectordb-bench.git
cd vectordb-bench

# Run setup
chmod +x infrastructure/setup_ec2.sh
./infrastructure/setup_ec2.sh

# IMPORTANT: Log out and back in for Docker permissions
exit
```

Then reconnect:
```powershell
ssh -i "your-key.pem" ubuntu@<EC2-PUBLIC-IP>
```

---

## Phase 3: Run Benchmarks (~5 hrs, ~$1)

### 3.1 Start the benchmark

```bash
cd ~/vectordb-bench
source venv/bin/activate

# Make script executable
chmod +x scripts/run_all_benchmarks.sh

# Run everything (will take 4-6 hours)
./scripts/run_all_benchmarks.sh
```

### 3.2 Monitor progress (optional, in another terminal)

```bash
# Watch the logs
tail -f results/run_1/benchmark_log.txt
```

### 3.3 Check Docker containers

```bash
docker-compose ps
docker stats
```

---

## Phase 4: Get Results (~5 min)

### 4.1 Download results to your PC

Open a NEW PowerShell window on your local machine:

```powershell
# Create local results folder
mkdir C:\Users\dsinh\research\vectordb-bench\final_results

# Download everything
scp -i "C:\path\to\your-key.pem" -r ubuntu@<EC2-IP>:~/vectordb-bench/results C:\Users\dsinh\research\vectordb-bench\final_results\
scp -i "C:\path\to\your-key.pem" -r ubuntu@<EC2-IP>:~/vectordb-bench/analysis C:\Users\dsinh\research\vectordb-bench\final_results\
```

### 4.2 STOP THE EC2 INSTANCE!

**IMPORTANT: Stop immediately to avoid charges!**

1. Go to: https://console.aws.amazon.com/ec2
2. Select your instance
3. **Instance State** → **Stop instance**

(Or terminate if you're done forever)

---

## Phase 5: Review Results

Your results are now in:

```
C:\Users\dsinh\research\vectordb-bench\final_results\
├── results\
│   ├── run_1\
│   │   ├── chroma_scifact_*.json
│   │   ├── qdrant_scifact_*.json
│   │   └── combined_*.json
│   ├── run_2\
│   ├── run_3\
│   ├── run_4\
│   └── run_5\
└── analysis\
    ├── recall_comparison.png
    ├── latency_comparison.png
    ├── temporal_drift_curves.png      ← NOVEL
    ├── cost_performance_pareto.png    ← NOVEL
    ├── operational_complexity_radar.png ← NOVEL
    ├── summary_report.md
    └── results_table.tex              ← For paper
```

### View the summary:

```powershell
cat C:\Users\dsinh\research\vectordb-bench\final_results\analysis\summary_report.md
```

---

## Cost Summary

| Phase | Time | Cost |
|-------|------|------|
| Local testing | 10 min | $0 |
| EC2 setup | 30 min | $0.08 |
| Full benchmarks | 5 hrs | $0.77 |
| Download results | 5 min | $0.01 |
| **TOTAL** | **~6 hrs** | **~$1** |

---

## Troubleshooting

### "Permission denied" on SSH
```powershell
# Fix key permissions (PowerShell)
icacls "your-key.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

### Docker permission denied
```bash
# On EC2, make sure you logged out and back in after setup
# Or run:
sudo usermod -aG docker $USER
newgrp docker
```

### Milvus won't start
```bash
# Milvus needs more memory - check Docker resources
docker logs vectordb-bench-milvus
# Give it more time (up to 3 min to initialize)
```

### Out of disk space
```bash
# Clean up Docker
docker system prune -a
```

---

## Next Steps After Benchmarking

1. Copy figures from `analysis/` to `paper/figures/`
2. Copy LaTeX tables from `analysis/results_table.tex` to paper
3. Update paper with actual numbers
4. Write analysis of results
