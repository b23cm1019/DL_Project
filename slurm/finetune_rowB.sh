#!/bin/bash
# Legacy alias kept for convenience. Prefer slurm/row_b_finetune.sh if you add one later.
#SBATCH --job-name=row_b_finetune
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/row_b_finetune_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/row_b_finetune_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p /csehome/p24cs0203/krish/logs/slurm
PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"
echo "Node: $(hostname) | GPU: ${DCPT_GPU_NAME} | Started: $(date)"
echo "=== STARTING row_b_finetune ==="
bash "${PROJECT_ROOT}/run_experiments.sh" row_b_finetune
echo "=== row_b_finetune COMPLETE : $(date) ==="
