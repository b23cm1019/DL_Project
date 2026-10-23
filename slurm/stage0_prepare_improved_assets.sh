#!/bin/bash
#SBATCH --job-name=stage0_assets_imp
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_assets_improved_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_assets_improved_%j.err
#SBATCH --time=08:00:00
set -euo pipefail

PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
export TORCH_HOME=/scratch/p24cs0203/krish/stage0_model_cache

python scripts/prepare_stage0_assets.py \
  --clean-root /scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train/clear \
  --onerestore-assets "$PROJECT_ROOT/third_party/OneRestore/syn_data/data" \
  --output-root /scratch/p24cs0203/krish/stage0_assets_improved \
  --mode improved
