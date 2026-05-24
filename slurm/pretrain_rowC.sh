#!/bin/bash
#SBATCH --job-name=dcpt_pretrain_rowC
#SBATCH --partition=phd
#SBATCH --account=p24cs0203
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pretrain_rowC_%j.out
#SBATCH --error=logs/pretrain_rowC_%j.err

# GPU 0 = RTX A6000 (CUDA 11.8-compatible). Do NOT use GPU 1 or 2 (Blackwell / CUDA 12.1).
export CUDA_VISIBLE_DEVICES=0

module purge
module load python/3.10.pytorch

cd ~/DL_Project
mkdir -p logs

echo "Running on node: $(hostname)"
echo "GPU selected: ${CUDA_VISIBLE_DEVICES} (RTX A6000)"
echo "Job started: $(date)"

echo "=== STARTING ROW C PRE-TRAINING (Multi-Label BCE, 100k iters, batch=32) ==="
./run_experiments.sh row_c_pretrain

echo "=== ROW C PRE-TRAINING COMPLETE: $(date) ==="
