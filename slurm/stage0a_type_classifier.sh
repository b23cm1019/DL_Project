#!/bin/bash
#SBATCH --job-name=stage0a_type
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/stage0a_type_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/stage0a_type_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
bash run_experiments.sh stage0a_type_classifier
