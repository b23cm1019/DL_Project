#!/bin/bash
# Evaluate one checkpoint on CDD-11 test (+ the graded set if it exists), with optional probes.
# Run from the repo root, after `export OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt`:
#   sbatch --job-name=ev_c2 --export=ALL,CONFIG=s3_c2_constant.yml,CKPT=<.../net_g_200000.pth>,TAG=c2_constant ablations/slurm/eval.sh
#   DCPT:  --export=ALL,MODE=dcpt,CKPT=<DCPT net_g_200000.pth>,TAG=dcpt_200k
#   Probes: add PROBES="base shuffle null sev_up sev_down noise0.5 noise1.0 noise2.0 oracle_type oracle_full"
#   Test sets: TESTSETS="cdd11" or "graded" (default: both if the graded set exists)
# Results: $OUT_ROOT/ablations/eval/<TAG>/<testset>__<probe>.json
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=ablations/logs/%x_%j.out
#SBATCH --error=ablations/logs/%x_%j.err
set -euo pipefail
: "${CKPT:?set CKPT}" "${TAG:?set TAG}"
: "${OUT_ROOT:?export OUT_ROOT first, e.g. /scratch/p26cs0003/krish/sevdcpt}"
cd "${SLURM_SUBMIT_DIR}"
[[ -z "${VENV_BASE:-}" && -d "${HOME}/krish/envs/dl_project" ]] && export VENV_BASE="${HOME}/krish/envs"
source slurm/portable/env.sh
source ablations/slurm/_render.sh
[[ -f "${CKPT}" ]] || { echo "[ERROR] checkpoint not found: ${CKPT}" >&2; exit 2; }
MODE="${MODE:-config}"
GRADED="${GRADED:-${OUT_ROOT}/ablations/graded_cdd11/manifest.jsonl}"
if [[ -z "${TESTSETS:-}" ]]; then
  TESTSETS="cdd11"
  [[ -f "${GRADED}" ]] && TESTSETS="cdd11 graded"
fi
args=(--mode "${MODE}" --ckpt "${CKPT}" --tag "${TAG}" --out_dir "${OUT_ROOT}/ablations/eval"
      --testsets ${TESTSETS} --probes ${PROBES:-base})
if [[ "${MODE}" == "config" ]]; then
  : "${CONFIG:?set CONFIG=<template>.yml (architecture + analysis copy)}"
  render_config "${CONFIG}"
  args+=(--config "${RENDERED}")
fi
[[ "${TESTSETS}" == *graded* ]] && args+=(--graded_manifest "${GRADED}")
# shellcheck disable=SC2086
python ablations/scripts/eval_ablation.py "${args[@]}" ${EXTRA:-}
