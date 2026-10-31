#!/bin/bash
# Cache full-image severity logits from the frozen Stage-2 analysis copy (~30-60 min).
#SBATCH --job-name=sevfull_cache
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/p26cs0003/krish/logs/sevfull_cache_%j.out
#SBATCH --error=/scratch/p26cs0003/krish/logs/sevfull_cache_%j.err
set -euo pipefail

PROJECT_ROOT=/csehome/p26cs0003/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
python scripts/cache_severity_full_conditions.py --temperature_fit "$@"
