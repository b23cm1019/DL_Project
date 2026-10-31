#!/bin/bash
# 15-minute GPU memory probe for Restormer at the 1-GPU batch sizes (Stage 1 B=32, Stage 3b B=32,
# full-image validation). Run it once on the node you plan to use, BEFORE Restormer Stage 1:
#   sbatch --job-name=sevrest_probe --nodelist=<node> slurm/portable/probe_restormer.sh
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=portable_logs/%x_%j.out
#SBATCH --error=portable_logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/portable/env.sh
python scripts/probe_restormer_1gpu_memory.py
