#!/bin/bash
#SBATCH --job-name=row_d_fin_resume
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/scratch/p24cs0203/krish/projects/DL_Project}}"
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

bash run_experiments.sh row_d_finetune_resume --force_yml dataloader:batch_size_per_gpu=4 train:ema_decay=0 train:optim_g:lr=2.5e-05
