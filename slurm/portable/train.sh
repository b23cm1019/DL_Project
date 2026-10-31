#!/bin/bash
# Generic 1-GPU training job for the portable (other-account) runs. Auto-resumes.
# Run from the repo root (logs go to ./portable_logs/):
#   sbatch --job-name=<name> --nodelist=<node> --export=ALL,CONFIG=<template>.yml slurm/portable/train.sh
# Templates: options/cdd_experiments/portable/*.yml  (@OUT_ROOT@ / @SHARED_ROOT@ are filled in here)
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-18:00:00
#SBATCH --output=portable_logs/%x_%j.out
#SBATCH --error=portable_logs/%x_%j.err
set -euo pipefail
: "${CONFIG:?set CONFIG=<template>.yml}"
cd "${SLURM_SUBMIT_DIR}"
source slurm/portable/env.sh

TEMPLATE="${PROJECT_ROOT}/options/cdd_experiments/portable/${CONFIG}"
RENDERED="${OUT_ROOT}/configs/${CONFIG}"
[[ -f "${TEMPLATE}" ]] || { echo "[ERROR] template not found: ${TEMPLATE}" >&2; exit 2; }
mkdir -p "${OUT_ROOT}/configs"
sed -e "s#@OUT_ROOT@#${OUT_ROOT}#g" -e "s#@SHARED_ROOT@#${SHARED_ROOT}#g" "${TEMPLATE}" > "${RENDERED}"
if grep -n "@OUT_ROOT@\|@SHARED_ROOT@" "${RENDERED}"; then echo "[ERROR] unrendered placeholder" >&2; exit 2; fi
echo "[INFO] config: ${RENDERED}"

# Preflight: every manifest exists; pretrained weights exist unless this run is resuming.
python - "${RENDERED}" <<'PY'
import sys
from pathlib import Path
from basicsr.utils.options import yaml_load
opt = yaml_load(sys.argv[1])
missing = [d["manifest"] for d in opt["datasets"].values() if "manifest" in d and not Path(d["manifest"]).is_file()]
states = Path(opt["path"]["experiments_root"]) / "training_states"
resuming = states.is_dir() and any(states.glob("*.state"))
if not resuming:
    for key in ("pretrain_network_g", "pretrain_network_extractor"):
        value = opt["path"].get(key)
        if value and not Path(value).is_file():
            missing.append(value)
if missing:
    sys.exit("[ERROR] missing inputs (did the previous stage / cache job finish?):\n  " + "\n  ".join(missing))
print(f"[INFO] preflight OK ({'resuming' if resuming else 'fresh start'})")
PY

RUN_MASTER_PORT="$((20000 + (SLURM_JOB_ID % 40000)))"
echo "[INFO] torchrun master port: ${RUN_MASTER_PORT}"
python -m torch.distributed.run --nproc_per_node=1 --master-port "${RUN_MASTER_PORT}" \
  basicsr/severity_train.py -opt "${RENDERED}" --launcher pytorch --auto_resume
