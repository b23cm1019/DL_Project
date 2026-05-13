#!/bin/bash
#SBATCH --job-name=sanity_row_d
#SBATCH --partition=btech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/b23cm1019/logs/slurm/%x_%j.out
#SBATCH --error=/scratch/b23cm1019/logs/slurm/%x_%j.err

set -euo pipefail

source "$(dirname "$0")/_cluster_env.sh"

bash run_experiments.sh sanity_row_d
