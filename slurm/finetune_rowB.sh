#!/bin/bash
#SBATCH --job-name=dcpt_finetune_rowB
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/finetune_rowB_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/finetune_rowB_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/p24cs0203/krish/projects/DL_Project}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo "Node    : $(hostname)"
echo "GPU(s)  : ${CUDA_VISIBLE_DEVICES:-0}"
echo "Started : $(date)"

echo "=== STARTING ROW B FINE-TUNING ==="
bash run_experiments.sh row_b_finetune
echo "=== ROW B FINE-TUNING COMPLETE: $(date) ==="
