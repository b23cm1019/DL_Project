#!/bin/bash
#SBATCH --job-name=rowd_2pass
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/rowd_2pass_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/rowd_2pass_%j.err
#SBATCH --time=24:00:00

set -euo pipefail
PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"
python scripts/evaluate_two_pass.py \
  -opt options/cdd_experiments/test_prompt.yml \
  --output /scratch/p24cs0203/krish/analysis/row_d_two_pass_20260827
