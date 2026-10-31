#!/bin/bash
# Cache full-image severity logits from a frozen Stage-2 analysis copy (Stage 3a/3b input).
# Default: the Restormer Stage 2 trained on this account. Run from the repo root:
#   sbatch --job-name=sevrest_cache --nodelist=<node> slurm/portable/cache.sh
# (NAFNet Plan B needs NO cache job: it reuses @SHARED_ROOT@/severity_full/manifests.)
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=0-06:00:00
#SBATCH --output=portable_logs/%x_%j.out
#SBATCH --error=portable_logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source slurm/portable/env.sh
BACKBONE="${BACKBONE:-restormer}"
S2="${OUT_ROOT}/experiments/SeverityRestormer_Stage2_CDDAdapt_1gpu/models"
GENERATOR="${GENERATOR:-${S2}/net_g_latest.pth}"
EXTRACTOR="${EXTRACTOR:-${S2}/net_extractor_latest.pth}"
CACHE_OUT="${CACHE_OUT:-${OUT_ROOT}/severity_full_restormer}"
for f in "${GENERATOR}" "${EXTRACTOR}"; do [[ -f "$f" ]] || { echo "[ERROR] missing $f (Stage 2 not finished?)" >&2; exit 2; }; done
python scripts/cache_severity_full_conditions.py --backbone "${BACKBONE}" \
  --generator "${GENERATOR}" --extractor "${EXTRACTOR}" --out "${CACHE_OUT}" --temperature_fit
