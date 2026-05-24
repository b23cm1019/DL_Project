#!/bin/bash
#SBATCH --job-name=dcpt_pretrain_rowC
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/%x_%j.err

# =============================================================================
# pretrain.slurm  —  Row C pre-training (multi-label BCE, 25k iters)
#
# Works on ANY GPU assigned by SLURM on cn07:
#   A6000   (GPUs 0,3,4,5): 49 GB, cu118, dl_project venv
#   Blackwell(GPUs 1,2)   : 98 GB, cu121, dl_project_cu121 venv
# _cluster_env.sh handles venv selection automatically.
# =============================================================================

set -euo pipefail

mkdir -p /scratch/p24cs0203/krish/logs/slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/scratch/p24cs0203/krish/projects/DL_Project}}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo "Node    : $(hostname)"
echo "GPU     : ${CUDA_VISIBLE_DEVICES:-0}"
echo "Started : $(date)"

echo "=== STARTING ROW C PRE-TRAINING ==="
bash "${PROJECT_ROOT}/run_experiments.sh" row_c_pretrain
echo "=== ROW C PRE-TRAINING COMPLETE : $(date) ==="
