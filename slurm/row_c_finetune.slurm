#!/bin/bash
#SBATCH --job-name=row_c_finetune
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/%x_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/%x_%j.err

set -euo pipefail

mkdir -p /csehome/p24cs0203/krish/logs/slurm

PROJECT_ROOT="${PROJECT_ROOT:-/csehome/p24cs0203/krish/projects/DL_Project}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo "Node    : $(hostname)"
echo "GPU     : ${CUDA_VISIBLE_DEVICES:-0}"
echo "Started : $(date)"

echo "=== STARTING ROW C FINE-TUNING ==="
bash "${PROJECT_ROOT}/run_experiments.sh" row_c_finetune
echo "=== ROW C FINE-TUNING COMPLETE : $(date) ==="
