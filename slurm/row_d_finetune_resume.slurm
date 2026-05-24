#!/bin/bash
#SBATCH --job-name=row_d_fin_resume
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/%x_%j.err

set -euo pipefail

mkdir -p /scratch/p24cs0203/krish/logs/slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/scratch/p24cs0203/krish/projects/DL_Project}}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo "Node    : $(hostname)"
echo "GPU     : ${CUDA_VISIBLE_DEVICES:-0}"
echo "Started : $(date)"

echo "=== RESUMING ROW D FINE-TUNING ==="
bash "${PROJECT_ROOT}/run_experiments.sh" row_d_finetune_resume
echo "=== ROW D FINE-TUNING RESUME COMPLETE : $(date) ==="
