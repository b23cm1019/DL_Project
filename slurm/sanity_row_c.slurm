#!/bin/bash
#SBATCH --job-name=sanity_row_c
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/%x_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/%x_%j.err

# Runs 5 training iterations to verify the full stack works end-to-end.
# Run this on EVERY venv rebuild before submitting long jobs.

set -euo pipefail

mkdir -p /csehome/p24cs0203/krish/logs/slurm

PROJECT_ROOT="${PROJECT_ROOT:-/csehome/p24cs0203/krish/projects/DL_Project}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo "=== SANITY CHECK: Row C (5 iters) ==="
bash "${PROJECT_ROOT}/run_experiments.sh" sanity_row_c
echo "=== SANITY OK ==="
