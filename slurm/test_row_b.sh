#!/bin/bash
#SBATCH --job-name=test_row_b
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/test_row_b_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/test_row_b_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
mkdir -p /csehome/p24cs0203/krish/logs/slurm
PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"
echo "Node: $(hostname) | GPU: ${DCPT_GPU_NAME} | Started: $(date)"
echo "=== STARTING test_row_b ==="
bash "${PROJECT_ROOT}/run_experiments.sh" test_row_b
echo "=== test_row_b COMPLETE : $(date) ==="
