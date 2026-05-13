#!/bin/bash
#SBATCH --job-name=row_c_fin_resume
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/b23cm1019/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/b23cm1019/logs/slurm/%x_%j.err

set -euo pipefail

source "$(dirname "$0")/_cluster_env.sh"

bash run_experiments.sh row_c_finetune_resume
