#!/bin/bash
#SBATCH --job-name=dcpt_pretrain_rowB
#SBATCH --partition=phd
#SBATCH --account=p24cs0203
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pretrain_rowB_%j.out
#SBATCH --error=logs/pretrain_rowB_%j.err

# ── GPU selection ──────────────────────────────────────────────────────────────
# cn07 GPU layout:
#   0,3,4,5 → NVIDIA RTX A6000 (49 GB)  ← CUDA 11.8-compatible, use these
#   1,2     → NVIDIA RTX PRO 6000 Blackwell (98 GB) ← CUDA 12.1, causes
#              "PyTorch and torchvision compiled with different CUDA major versions"
# We select GPU 0 (first A6000). To use a different A6000, change to 3, 4, or 5.
export CUDA_VISIBLE_DEVICES=0

module purge
module load python/3.10.pytorch

cd ~/DL_Project
mkdir -p logs

echo "Running on node: $(hostname)"
echo "GPU selected: ${CUDA_VISIBLE_DEVICES} (RTX A6000)"
echo "Job started: $(date)"

echo "=== STARTING ROW B PRE-TRAINING (Focal Loss, 100k iters, batch=32) ==="
./run_experiments.sh row_b_pretrain

echo "=== ROW B PRE-TRAINING COMPLETE: $(date) ==="
