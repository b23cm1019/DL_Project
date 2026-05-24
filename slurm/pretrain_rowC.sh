#!/bin/bash
#SBATCH --job-name=rowC_pretrain
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/pretrain_rowC_%j.out

set -euo pipefail

PROJECT_ROOT="/scratch/p24cs0203/krish/projects/DL_Project"

cd "${PROJECT_ROOT}"

# -----------------------------------------------------------------------------
# Centralized environment setup
# -----------------------------------------------------------------------------
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
echo "Node    : $(hostname)"
echo "GPU(s)  : $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Started : $(date)"

echo "=== STARTING ROW C PRE-TRAINING ==="

# -----------------------------------------------------------------------------
# Run experiment
# -----------------------------------------------------------------------------
bash run_experiments.sh row_c_pretrain

echo "=== ROW C PRE-TRAINING COMPLETE: $(date) ==="