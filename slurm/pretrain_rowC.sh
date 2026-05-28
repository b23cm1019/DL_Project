#!/bin/bash
#SBATCH --job-name=rowC_pretrain
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/pretrain_rowC_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/pretrain_rowC_%j.err
#SBATCH --time=08:00:00

set -euo pipefail

mkdir -p /csehome/p24cs0203/krish/logs/slurm

PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo "Node    : $(hostname)"
echo "GPU     : ${DCPT_GPU_NAME}"
echo "Started : $(date)"

echo "=== STARTING ROW C PRE-TRAINING ==="
bash "${PROJECT_ROOT}/run_experiments.sh" row_c_pretrain
echo "=== ROW C PRE-TRAINING COMPLETE : $(date) ==="
