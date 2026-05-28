#!/bin/bash
#SBATCH --job-name=sanity_row_c
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/sanity_row_c_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/sanity_row_c_%j.err
#SBATCH --time=00:30:00
set -euo pipefail
mkdir -p /csehome/p24cs0203/krish/logs/slurm
PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"
echo "Node: $(hostname) | GPU: ${DCPT_GPU_NAME} | Started: $(date)"
echo "=== SANITY CHECK ROW C (5 iters) ==="
bash "${PROJECT_ROOT}/run_experiments.sh" sanity_row_c
echo "=== SANITY OK : $(date) ==="
