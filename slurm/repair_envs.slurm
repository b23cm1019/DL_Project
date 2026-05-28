#!/bin/bash
#SBATCH --job-name=repair_envs
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/%x_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/%x_%j.err

set -euo pipefail

mkdir -p /csehome/p24cs0203/krish/logs/slurm

PROJECT_ROOT="${PROJECT_ROOT:-/csehome/p24cs0203/krish/projects/DL_Project}"
cd "${PROJECT_ROOT}"

echo "Node    : $(hostname)"
echo "Started : $(date)"
echo "=== REPAIRING / RESUMING ENVIRONMENTS ==="
bash "${PROJECT_ROOT}/install_venv_packages.sh"
echo "=== ENVIRONMENT REPAIR COMPLETE : $(date) ==="
