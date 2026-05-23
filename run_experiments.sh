#!/bin/bash
# DCPT CDD-11 experiment runner for Row C and Row D.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export DCPT_DATA_ROOT="${DCPT_DATA_ROOT:-${SCRIPT_DIR}/datasets/CDD11}"
export BASICSR_EXPERIMENTS_ROOT="${BASICSR_EXPERIMENTS_ROOT:-${SCRIPT_DIR}/outputs}"
export BASICSR_MODELS_ROOT="${BASICSR_MODELS_ROOT:-${SCRIPT_DIR}/checkpoints}"
export BASICSR_TRAINING_STATES_ROOT="${BASICSR_TRAINING_STATES_ROOT:-${SCRIPT_DIR}/training_states}"
export BASICSR_LOG_ROOT="${BASICSR_LOG_ROOT:-${SCRIPT_DIR}/logs}"
export BASICSR_VIS_ROOT="${BASICSR_VIS_ROOT:-${SCRIPT_DIR}/visualizations}"
export BASICSR_RESULTS_ROOT="${BASICSR_RESULTS_ROOT:-${SCRIPT_DIR}/results}"
export BASICSR_TB_ROOT="${BASICSR_TB_ROOT:-${SCRIPT_DIR}/tb_logger}"

LOG_FILTER='Starting |Start training|Resuming training|Saving models and training states|Validation|Testing |End of training|Save the latest model|iter:|Traceback|Error|Exception|failed|Killed'

usage() {
  cat <<'EOF'
Usage:
  ./run_experiments.sh <stage> [extra_args...]

Stages:
  sanity_row_c
  sanity_row_d
  row_c_pretrain
  row_c_pretrain_resume
  row_c_finetune
  row_c_finetune_resume
  row_d_finetune
  row_d_finetune_resume
  test_row_c
  test_row_d

Examples:
  ./run_experiments.sh sanity_row_c
  ./run_experiments.sh row_c_pretrain
  ./run_experiments.sh row_c_finetune
  ./run_experiments.sh row_c_pretrain_resume
  ./run_experiments.sh row_d_finetune
  ./run_experiments.sh test_row_c

Notes:
  - Row D depends on the Row C pretraining checkpoint and classifier.
  - Full logs, checkpoints, and outputs are routed by BASICSR_* env vars.
  - All long training stages save checkpoints every 2500 iters and skip in-training validation to reduce walltime overhead.
  - If you run multiple stages at once, give each one a different MASTER_PORT.
  - Any extra args are forwarded to the underlying Python entrypoint.
EOF
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
  sanity_row_c)
    echo "Running a short Row C sanity check on CDD-11 (100 iters, batch size 1, port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch --force_yml train:total_iter=100 logger:save_checkpoint_freq=1000 val:val_freq=101 dataloader:batch_size_per_gpu=1 dataloader:num_worker_per_gpu=0 dataloader_val:num_worker_per_gpu=0 train:ema_decay=0 "${EXTRA_ARGS[@]}"
    ;;
  sanity_row_d)
    echo "Running a short Row D sanity check on CDD-11 (100 iters, batch size 1, port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch --force_yml train:total_iter=100 logger:save_checkpoint_freq=1000 val:val_freq=101 dataloader:batch_size_per_gpu=1 dataloader:num_worker_per_gpu=0 dataloader_val:num_worker_per_gpu=0 train:ema_decay=0 path:strict_load_g=false "${EXTRA_ARGS[@]}"
    ;;
  row_c_pretrain)
    echo "Starting Row C pretraining on CDD-11 (multi-label BCE, port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch --force_yml val:val_freq=10000 logger:save_checkpoint_freq=5000 "${EXTRA_ARGS[@]}"
    ;;
  row_c_pretrain_resume)
    echo "Resuming Row C pretraining from the latest saved state (port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch --auto_resume --force_yml val:val_freq=10000 logger:save_checkpoint_freq=5000 "${EXTRA_ARGS[@]}"
    ;;
  row_c_finetune)
    echo "Starting Row C finetuning on CDD-11 (port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_multilabel.yml --launcher pytorch --force_yml val:val_freq=50000 logger:save_checkpoint_freq=10000 "${EXTRA_ARGS[@]}"
    ;;
  row_c_finetune_resume)
    echo "Resuming Row C finetuning on CDD-11 (port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_multilabel.yml --launcher pytorch --auto_resume --force_yml val:val_freq=50000 logger:save_checkpoint_freq=10000 "${EXTRA_ARGS[@]}"
    ;;
  row_d_finetune)
    echo "Starting Row D finetuning on CDD-11 with prompt injection (port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch --force_yml path:strict_load_g=false val:val_freq=50000 logger:save_checkpoint_freq=10000 "${EXTRA_ARGS[@]}"
    ;;
  row_d_finetune_resume)
    echo "Resuming Row D finetuning with prompt injection (port ${MASTER_PORT})"
    run_stage python -m torch.distributed.run --master_port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch --auto_resume --force_yml path:strict_load_g=false classify=false resume_remove_dc=true val:val_freq=50000 logger:save_checkpoint_freq=10000 "${EXTRA_ARGS[@]}"
    ;;
  test_row_c)
    echo "Running CDD-11 evaluation for Row C"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_multilabel.yml "${EXTRA_ARGS[@]}"
    ;;
  test_row_d)
    echo "Running CDD-11 evaluation for Row D"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_prompt.yml "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unknown stage: $STAGE"
    usage
    exit 1
    ;;
esac
