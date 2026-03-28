#!/bin/bash
# DCPT experiment runner with explicit stage selection.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
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
  ./run_experiments.sh row_b_finetune
  ./run_experiments.sh row_b_pretrain --auto_resume
  ./run_experiments.sh row_b_pretrain_resume
  ./run_experiments.sh row_d_finetune_resume
  ./run_experiments.sh row_b_pretrain --force_yml path:resume_state=experiments/Row_B_Baseline_Pretrain_CDD11/training_states/5000.state

Notes:
  - Pretraining and finetuning are intentionally separated so failures are isolated.
  - Finetuning expects the required pretrained checkpoint to already exist.
  - The terminal only shows important progress/error lines; full logs are still saved by BasicSR.
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
  row_b_pretrain)
    echo "Starting Row B Pretraining (11-class single-label FocalLoss, 25k iters, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_baseline.yml --launcher pytorch "${EXTRA_ARGS[@]}"
    ;;
  row_b_pretrain_resume)
    echo "Resuming Row B Pretraining from latest state with reduced validation/checkpoint frequency (port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_baseline.yml --launcher pytorch --auto_resume --force_yml val:val_freq=25000 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  row_b_finetune)
    echo "Starting Row B Finetuning (100k iters, checkpoints every 25k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_baseline.yml --launcher pytorch --force_yml val:val_freq=100001 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  row_c_pretrain)
    echo "Starting Row C Pretraining (4-primitive multi-hot BCE, 25k iters, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch "${EXTRA_ARGS[@]}"
    ;;
  row_c_pretrain_resume)
    echo "Resuming Row C Pretraining from latest state with reduced validation/checkpoint frequency (port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch --auto_resume --force_yml val:val_freq=25000 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  row_c_finetune)
    echo "Starting Row C Finetuning (100k iters, checkpoints every 25k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_multilabel.yml --launcher pytorch --force_yml val:val_freq=100001 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  row_d_finetune)
    echo "Starting Row D Finetuning with Prompt Injection (uses Row C multilabel pretrain, checkpoints every 25k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch --force_yml val:val_freq=100001 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  row_d_finetune_resume)
    echo "Resuming Row D Finetuning with Prompt Injection (non-strict generator load, checkpoints every 25k, no in-training validation, port ${MASTER_PORT})"
    run_stage torchrun --master-port "${MASTER_PORT}" --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch --auto_resume --force_yml classify=false resume_remove_dc=true path:strict_load_g=false val:val_freq=100001 logger:save_checkpoint_freq=25000 "${EXTRA_ARGS[@]}"
    ;;
  test_row_b)
    echo "Running Evaluation for Row B"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_baseline.yml "${EXTRA_ARGS[@]}"
    ;;
  test_row_c)
    echo "Running Evaluation for Row C"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_multilabel.yml "${EXTRA_ARGS[@]}"
    ;;
  test_row_d)
    echo "Running Evaluation for Row D"
    run_stage python basicsr/test.py -opt options/cdd_experiments/test_prompt.yml "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Unknown stage: $STAGE"
    usage
    exit 1
    ;;
esac
