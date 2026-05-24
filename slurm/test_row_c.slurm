#!/bin/bash
#SBATCH --job-name=test_row_c
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/%x_%j.err

set -euo pipefail

mkdir -p /scratch/p24cs0203/krish/logs/slurm

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/scratch/p24cs0203/krish/projects/DL_Project}}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo "=== TEST ROW C ==="
bash "${PROJECT_ROOT}/run_experiments.sh" test_row_c
echo "=== TEST ROW C COMPLETE : $(date) ==="
