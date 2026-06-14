#!/bin/bash
# Legacy alias kept for convenience. Prefer slurm/row_c_finetune.sh.
#SBATCH --job-name=row_c_finetune
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/row_c_finetune_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/row_c_finetune_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
export MASTER_PORT=29500
mkdir -p /csehome/p24cs0203/krish/logs/slurm
PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"
echo "Node: $(hostname) | GPU: ${DCPT_GPU_NAME} | Started: $(date)"
echo "=== STARTING row_c_finetune ==="
bash "${PROJECT_ROOT}/run_experiments.sh" row_c_finetune
echo "=== row_c_finetune COMPLETE : $(date) ==="
