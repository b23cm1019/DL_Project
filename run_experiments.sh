#!/bin/bash
# DCPT experiment runner — Config B (cn07 A6000, 100k pretrain / 500k finetune)
# GPU pinning: only A6000 cards (0,3,4,5) — avoids RTX PRO 6000 Blackwell GPUs (1,2)
# which have CUDA 12.1 and cause a PyTorch/torchvision CUDA version mismatch.

set -euo pipefail

# ── GPU selection ──────────────────────────────────────────────────────────────
# SLURM assigns the GPU and remaps it to index 0 inside the job.
# Default to 0 when running interactively.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# ── Resolve correct Python / torchrun from active venv ────────────────────────
# CRITICAL: torchrun must come from the active venv, NOT the system module.
# The system module (/opt/ohpc/apps/python/3.10.pytorch) has torchvision built
# against CUDA 12.1, which conflicts with the venv's PyTorch (CUDA 11.8).
# Prepending the venv bin ensures venv torchrun and python are used everywhere.
# ── Force venv python for all training processes ──────────────────────────────
# torchrun resolves to the system binary (/opt/ohpc/.../torchrun) even when the
# venv is active, and that binary spawns system python3.10 which has the wrong
# torchvision (CUDA 12.1 vs PyTorch CUDA 11.8).
# Fix: replace torchrun with "python -m torch.distributed.run" which always uses
# whichever python binary is first in PATH — i.e. the venv's python.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    echo "[INFO] python   : $(which python)"
else
    echo "[WARN] No active venv detected."
fi

# Alias torchrun to python -m torch.distributed.run for this session
torchrun() {
    python -m torch.distributed.run "$@"
}
export -f torchrun

export MASTER_PORT="${MASTER_PORT:-29500}"

LOG_FILTER='Starting |Start training|Resuming training|Saving models and training states|Validation|Testing |End of training|Save the latest model|iter:|Traceback|Error|Exception|failed|Killed'

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.sh <stage> [extra_args...]

Stages:
  row_b_pretrain
  row_b_pretrain_resume
  row_b_finetune
  row_c_pretrain
  row_c_pretrain_resume
  row_c_finetune
  row_d_finetune
  row_d_finetune_resume
  test_row_b
  test_row_c
  test_row_d

Examples:
  ./run_experiments.sh row_b_pretrain
  CUDA_VISIBLE_DEVICES=3 ./run_experiments.sh row_c_pretrain
  ./run_experiments.sh row_b_pretrain --auto_resume

GPU note (cn07):
  Safe GPUs  : 0, 3, 4, 5  (NVIDIA RTX A6000, 49 GB, CUDA 11.8-compatible)
  Unsafe GPUs: 1, 2         (NVIDIA RTX PRO 6000 Blackwell, CUDA 12.1 — causes
                              torchvision CUDA version mismatch)
  Default    : GPU 0

Scale targets (per DCPT Scaling Guide):
  Pre-training  : 100,000 iterations  (matches paper exactly)
  Fine-tuning   : 500,000 iterations  (67% of paper, ~90-95% PSNR)
  Batch size    : 32 per GPU           (matches paper exactly on A6000 48 GB)
  LR encoder    : 3e-4
  LR decoder    : 1e-4
EOF
}

preflight_import() {
  local module="$1"
  echo "[INFO] Preflight import check: ${module}"
  python -c "import importlib; importlib.import_module('${module}'); print('[INFO] Preflight import OK: ${module}')"
}

preflight_torchvision_cuda() {
  echo "[INFO] Preflight torch/torchvision CUDA check"
  python - <<'PY'
import sys

import torch
import torchvision

print("[INFO] Preflight PyTorch     :", torch.__version__, " CUDA:", torch.version.cuda)
print("[INFO] Preflight torchvision :", torchvision.__version__)
try:
    from torchvision.extension import _check_cuda_version

    _check_cuda_version()
except Exception as exc:
    print(f"[ERROR] torch/torchvision CUDA compatibility check failed: {exc}", file=sys.stderr)
    sys.exit(1)
print("[INFO] Preflight torch/torchvision CUDA OK")
PY
}

run_stage() {
  set +e
  "$@" 2>&1 | stdbuf -oL -eL grep -E --line-buffered "$LOG_FILTER"
  local pipe_status=("${PIPESTATUS[@]}")
  local cmd_status="${pipe_status[0]:-0}"
  local grep_status="${pipe_status[1]:-0}"
  set -e

  if [[ $cmd_status -ne 0 ]]; then
    return "$cmd_status"
  fi

  if [[ $grep_status -gt 1 ]]; then
    return "$grep_status"
  fi
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

STAGE="$1"
shift
EXTRA_ARGS=("$@")

case "$STAGE" in
  row_b_pretrain)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Starting Row B Pretraining (11-class single-label FocalLoss, 100k iters, GPU ${CUDA_VISIBLE_DEVICES}, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_baseline.yml --launcher pytorch "${EXTRA_ARGS[@]}"
    ;;
  row_b_pretrain_resume)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Resuming Row B Pretraining from latest state (port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_baseline.yml --launcher pytorch --auto_resume --force_yml val:val_freq=25000 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  row_b_finetune)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Starting Row B Finetuning (500k iters, checkpoints every 50k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_baseline.yml --launcher pytorch --force_yml val:val_freq=500001 logger:save_checkpoint_freq=50000 "${EXTRA_ARGS[@]}"
    ;;
  row_c_pretrain)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Starting Row C Pretraining (4-primitive multi-hot BCE, 100k iters, GPU ${CUDA_VISIBLE_DEVICES}, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch "${EXTRA_ARGS[@]}"
    ;;
  row_c_pretrain_resume)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Resuming Row C Pretraining from latest state (port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch --auto_resume --force_yml val:val_freq=25000 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  row_c_finetune)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Starting Row C Finetuning (500k iters, checkpoints every 50k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_multilabel.yml --launcher pytorch --force_yml val:val_freq=500001 logger:save_checkpoint_freq=50000 "${EXTRA_ARGS[@]}"
    ;;
  row_d_finetune)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Starting Row D Finetuning with Prompt Injection (500k iters, checkpoints every 50k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch --force_yml val:val_freq=500001 logger:save_checkpoint_freq=50000 "${EXTRA_ARGS[@]}"
    ;;
  row_d_finetune_resume)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "Resuming Row D Finetuning with Prompt Injection (non-strict generator load, checkpoints every 50k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch --auto_resume --force_yml classify=false resume_remove_dc=true path:strict_load_g=false val:val_freq=500001 logger:save_checkpoint_freq=50000 "${EXTRA_ARGS[@]}"
    ;;
  test_row_b)
    preflight_import basicsr.test
    preflight_torchvision_cuda
    echo "Running Evaluation for Row B"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_baseline.yml "${EXTRA_ARGS[@]}"
    ;;
  test_row_c)
    preflight_import basicsr.test
    preflight_torchvision_cuda
    echo "Running Evaluation for Row C"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_multilabel.yml "${EXTRA_ARGS[@]}"
    ;;
  test_row_d)
    preflight_import basicsr.test
    preflight_torchvision_cuda
    echo "Running Evaluation for Row D"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_prompt.yml "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unknown stage: $STAGE"
    usage
    exit 1
    ;;
esac
