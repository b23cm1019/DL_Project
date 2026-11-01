#!/bin/bash
# CDD-11 evaluation on one GPU (~15-20 min for the full 2,200-image test set). Run from the repo root:
#   sbatch --job-name=<tag> --nodelist=<node> --export=ALL,MODE=dcpt,CKPT=/path/net_g_X.pth,TAG=<tag> slurm/portable/evaluate.sh
# MODE: dcpt | frozen | full.   Optional EXTRA (quoted), e.g.
#   EXTRA="--limit_per_setting 20"   (the 220-image subset used during training)
#   EXTRA="--shuffle_condition"      (full mode ablation)
# Results: $OUT_ROOT/eval/<TAG>.json and .csv; table printed at the end of portable_logs/<job>_<id>.out
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=portable_logs/%x_%j.out
#SBATCH --error=portable_logs/%x_%j.err
set -euo pipefail
: "${MODE:?set MODE}" "${CKPT:?set CKPT}" "${TAG:?set TAG}"
cd "${SLURM_SUBMIT_DIR}"
source slurm/portable/env.sh
[[ -f "${CKPT}" ]] || { echo "[ERROR] checkpoint not found: ${CKPT}" >&2; exit 2; }
mkdir -p "${OUT_ROOT}/eval"
# shellcheck disable=SC2086
python scripts/evaluate_cdd11_full.py --mode "${MODE}" --ckpt "${CKPT}" --tag "${TAG}" \
  --out_dir "${OUT_ROOT}/eval" ${EXTRA:-}
