#!/bin/bash
# Cache full-image condition logits for a Stage 3 ablation that has its own Stage 2 (S5, S10, T5, X3).
# Run after that ablation's Stage 2 finished, from the repo root:
#   sbatch --job-name=abl_cache_x3 --export=ALL,CONFIG=s3_x3_holdout_lhs.yml ablations/slurm/cache.sh
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-08:00:00
#SBATCH --output=ablations/logs/%x_%j.out
#SBATCH --error=ablations/logs/%x_%j.err
set -euo pipefail
: "${CONFIG:?set CONFIG=<stage 3 template>.yml}"
: "${OUT_ROOT:?export OUT_ROOT first, e.g. /scratch/p26cs0003/krish/sevdcpt}"
cd "${SLURM_SUBMIT_DIR}"
[[ -z "${VENV_BASE:-}" && -d "${HOME}/krish/envs/dl_project" ]] && export VENV_BASE="${HOME}/krish/envs"
source slurm/portable/env.sh
source ablations/slurm/_render.sh
render_config "${CONFIG}"
python ablations/scripts/cache_conditions.py --config "${RENDERED}"
