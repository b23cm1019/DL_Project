#!/bin/bash
# Evaluate any checkpoint on the official CDD-11 test set. Usage:
#   sbatch --export=ALL,MODE=dcpt|frozen|full,CKPT=/path/net_g_X.pth,TAG=name[,EXTRA="--shuffle_condition"] slurm/evaluate_cdd11_full.sh
#SBATCH --job-name=cdd11_eval
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/p26cs0003/krish/logs/cdd11_eval_%j.out
#SBATCH --error=/scratch/p26cs0003/krish/logs/cdd11_eval_%j.err
set -euo pipefail

PROJECT_ROOT=/csehome/p26cs0003/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
: "${MODE:?set MODE}" "${CKPT:?set CKPT}" "${TAG:?set TAG}"
# shellcheck disable=SC2086
python scripts/evaluate_cdd11_full.py --mode "$MODE" --ckpt "$CKPT" --tag "$TAG" ${EXTRA:-}
