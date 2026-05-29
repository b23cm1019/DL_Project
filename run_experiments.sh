#!/bin/bash
# =============================================================================
# run_experiments.sh  —  DCPT experiment runner for cn07 (A6000 + Blackwell)
#
# IMPORTANT: Do NOT run this directly. It is invoked by SLURM scripts after
# _cluster_env.sh has activated the correct venv.
# Interactive use (for debugging only):
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --pty bash
#   source slurm/_cluster_env.sh  # sets up venv + env vars
#   bash run_experiments.sh <stage>
#
# The venv's python -m torch.distributed.run is used instead of the raw
# `torchrun` binary to guarantee the venv python is used for worker processes.
# =============================================================================

set -euo pipefail

# ── Guard: _cluster_env.sh must have been sourced first ───────────────────────
if [[ -z "${VENV_ROOT:-}" ]]; then
    echo "[ERROR] VENV_ROOT is not set."
    echo "        Source slurm/_cluster_env.sh before calling this script."
    exit 1
fi

if [[ -z "${DCPT_CUDA_FLAVOR:-}" ]]; then
    echo "[ERROR] DCPT_CUDA_FLAVOR is not set."
    echo "        Source slurm/_cluster_env.sh before calling this script."
    exit 1
fi

# ── Ensure venv torchrun is always first in PATH ──────────────────────────────
export PATH="${VENV_ROOT}/bin:${PATH}"

# Replace bare `torchrun` with `python -m torch.distributed.run`.
# This guarantees the correct venv python3 is used for ALL worker processes,
# not the system python that the system torchrun binary hardcodes.
torchrun() {
    python -m torch.distributed.run "$@"
}
export -f torchrun

export MASTER_PORT="${MASTER_PORT:-29500}"

# ── Log filter: only show meaningful lines ────────────────────────────────────
LOG_FILTER='Starting |Start training|Resuming training|Saving models|Validation|Testing |End of training|Save the latest|iter:|Traceback|Error|Exception|failed|Killed|WARN|INFO'

# ── Preflight checks ──────────────────────────────────────────────────────────
preflight_import() {
    local module="$1"
    echo "[INFO] Preflight: import ${module}"
    python -c "import importlib; importlib.import_module('${module}'); print('[OK] ${module}')"
}

preflight_torchvision_cuda() {
    echo "[INFO] Preflight: torch/torchvision CUDA check"
    python - <<'PY'
import sys, torch, torchvision
print(f"[OK] PyTorch     : {torch.__version__}  CUDA: {torch.version.cuda}")
print(f"[OK] torchvision : {torchvision.__version__}")
try:
    from torchvision.extension import _check_cuda_version
    _check_cuda_version()
    print("[OK] torch/torchvision CUDA compatibility: PASSED")
except Exception as exc:
    print(f"[FAIL] torch/torchvision CUDA mismatch: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

# ── Stage runner ──────────────────────────────────────────────────────────────
run_stage() {
    set +e
    "$@" 2>&1 | stdbuf -oL -eL grep -E --line-buffered "${LOG_FILTER}"
    local pipe_status=("${PIPESTATUS[@]}")
    local cmd_status="${pipe_status[0]:-0}"
    local grep_status="${pipe_status[1]:-0}"
    set -e

    if [[ $cmd_status -ne 0 ]]; then
        echo "[ERROR] Stage command exited with status ${cmd_status}"
        return "${cmd_status}"
    fi
    # grep exit 1 means no match (ok); >1 means real grep error
    if [[ $grep_status -gt 1 ]]; then
        return "${grep_status}"
    fi
}

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'EOF'
Usage:
  bash run_experiments.sh <stage> [extra_args...]

Stages:
  row_b_pretrain
  row_b_pretrain_resume
  row_b_finetune
  row_c_pretrain
  row_c_pretrain_resume
  row_c_finetune
  row_c_finetune_resume
  row_d_finetune
  row_d_finetune_resume
  test_row_b
  test_row_c
  test_row_d
  sanity_row_c
  sanity_row_d

Examples:
  bash run_experiments.sh row_c_pretrain
  bash run_experiments.sh row_c_pretrain --auto_resume
  bash run_experiments.sh row_c_pretrain --force_yml dataloader:batch_size_per_gpu=4

GPU note (cn07):
  A6000 (GPUs 0,3,4,5)     : 49 GB, CUDA 11.8  → dl_project venv (cu118)
  Blackwell (GPUs 1,2)      : 98 GB, CUDA 12.1  → dl_project_cu121 venv (cu121)
  SLURM remaps assigned GPU to index 0. _cluster_env.sh auto-selects the venv.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

STAGE="$1"
shift
EXTRA_ARGS=("$@")

# ── Common LR overrides for scaled-down setup ─────────────────────────────────
# These match the EXPERIMENTAL_SETUP.md parameters (batch=8, single GPU).
SCALE_OVERRIDES=(
    "--force_yml"
    "dataloader:batch_size_per_gpu=8"
    "train:optim_g:lr=1e-4"
    "train:optim_dc:lr=5e-5"
    "train:ema_decay=0"
)

case "${STAGE}" in
  # ── Row B (single-label FocalLoss baseline) ──────────────────────────────────
  row_b_pretrain)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row B Pretrain | GPU ${CUDA_VISIBLE_DEVICES} | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/pretrain_baseline.yml \
        --launcher pytorch \
        "${SCALE_OVERRIDES[@]}" \
        "${EXTRA_ARGS[@]}"
    ;;

  row_b_pretrain_resume)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row B Pretrain Resume | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/pretrain_baseline.yml \
        --launcher pytorch --auto_resume \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml val:val_freq=25000 logger:save_checkpoint_freq=25000 \
        "${EXTRA_ARGS[@]}"
    ;;

  row_b_finetune)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row B Finetune | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/finetune_baseline.yml \
        --launcher pytorch \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml val:val_freq=500001 logger:save_checkpoint_freq=50000 \
        "${EXTRA_ARGS[@]}"
    ;;

  # ── Row C (multi-label BCE) ───────────────────────────────────────────────────
  row_c_pretrain)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row C Pretrain | GPU ${CUDA_VISIBLE_DEVICES} | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/pretrain_multilabel.yml \
        --launcher pytorch \
        "${SCALE_OVERRIDES[@]}" \
        "${EXTRA_ARGS[@]}"
    ;;

  row_c_pretrain_resume)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row C Pretrain Resume | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/pretrain_multilabel.yml \
        --launcher pytorch --auto_resume \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml val:val_freq=25000 logger:save_checkpoint_freq=25000 \
        "${EXTRA_ARGS[@]}"
    ;;

  row_c_finetune)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row C Finetune | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/finetune_multilabel.yml \
        --launcher pytorch \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml val:val_freq=500001 logger:save_checkpoint_freq=50000 \
        "${EXTRA_ARGS[@]}"
    ;;

  row_c_finetune_resume)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row C Finetune Resume | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/finetune_multilabel.yml \
        --launcher pytorch --auto_resume \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml val:val_freq=500001 logger:save_checkpoint_freq=50000 \
        "${EXTRA_ARGS[@]}"
    ;;

  # ── Row D (Row C + Prompt Injection) ─────────────────────────────────────────
  row_d_finetune)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row D Finetune (Prompt Injection) | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/finetune_prompt.yml \
        --launcher pytorch \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml val:val_freq=500001 logger:save_checkpoint_freq=50000 \
        "${EXTRA_ARGS[@]}"
    ;;

  row_d_finetune_resume)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Row D Finetune Resume | port ${MASTER_PORT}"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/finetune_prompt.yml \
        --launcher pytorch --auto_resume \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml classify=false resume_remove_dc=true \
                     path:strict_load_g=false \
                     val:val_freq=500001 logger:save_checkpoint_freq=50000 \
        "${EXTRA_ARGS[@]}"
    ;;

  # ── Evaluation ────────────────────────────────────────────────────────────────
  test_row_b)
    preflight_import basicsr.test
    preflight_torchvision_cuda
    echo "[INFO] Test Row B"
    run_stage python basicsr/test.py \
        -opt options/cdd_experiments/test_baseline.yml \
        "${EXTRA_ARGS[@]}"
    ;;

  test_row_c)
    preflight_import basicsr.test
    preflight_torchvision_cuda
    echo "[INFO] Test Row C"
    run_stage python basicsr/test.py \
        -opt options/cdd_experiments/test_multilabel.yml \
        "${EXTRA_ARGS[@]}"
    ;;

  test_row_d)
    preflight_import basicsr.test
    preflight_torchvision_cuda
    echo "[INFO] Test Row D"
    run_stage python basicsr/test.py \
        -opt options/cdd_experiments/test_prompt.yml \
        "${EXTRA_ARGS[@]}"
    ;;

  # ── Sanity (1 iteration smoke test) ──────────────────────────────────────────
  sanity_row_c)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Sanity Row C (5 iters)"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/pretrain_multilabel.yml \
        --launcher pytorch \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml train:total_iter=5 logger:print_freq=1 \
                     val:val_freq=6 logger:save_checkpoint_freq=999999 \
        "${EXTRA_ARGS[@]}"
    ;;

  sanity_row_d)
    preflight_import basicsr.all_in_one_train
    preflight_torchvision_cuda
    echo "[INFO] Sanity Row D (5 iters)"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 \
        basicsr/all_in_one_train.py \
        -opt options/cdd_experiments/finetune_prompt.yml \
        --launcher pytorch \
        "${SCALE_OVERRIDES[@]}" \
        --force_yml train:total_iter=5 logger:print_freq=1 \
                     val:val_freq=6 logger:save_checkpoint_freq=999999 \
                     classify=false \
        "${EXTRA_ARGS[@]}"
    ;;

  *)
    echo "[ERROR] Unknown stage: ${STAGE}"
    usage
    exit 1
    ;;
esac
