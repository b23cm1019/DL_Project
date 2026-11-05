#!/bin/bash
# End-to-end smoke test of every ablation (few updates each at the real batch size, all scripts).
# Writes only under $OUT_ROOT/ablations_smoke. Run from the repo root:
#   export OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt
#   sbatch --job-name=abl_smoke --nodelist=cn02 ablations/slurm/smoke.sh
# Result: the PASS/FAIL table at the end of ablations/logs/abl_smoke_<id>.out
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-12:00:00
#SBATCH --output=ablations/logs/%x_%j.out
#SBATCH --error=ablations/logs/%x_%j.err
set -euo pipefail
: "${OUT_ROOT:?export OUT_ROOT first, e.g. /scratch/p26cs0003/krish/sevdcpt}"
cd "${SLURM_SUBMIT_DIR}"
[[ -z "${VENV_BASE:-}" && -d "${HOME}/krish/envs/dl_project" ]] && export VENV_BASE="${HOME}/krish/envs"
source slurm/portable/env.sh
python ablations/scripts/smoke_test.py ${EXTRA:-}
