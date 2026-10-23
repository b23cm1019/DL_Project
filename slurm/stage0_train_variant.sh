#!/bin/bash
# Unified launcher for the intended Stage 0 experiments.
# Required exports:
#   STAGE0_PHASE   = D, A, or B
#   STAGE0_VARIANT = proxy or improved
#
# D: synthetic severity pretraining only
# A: synthetic severity pretraining -> CDD-11 type fine-tuning
# B: joint synthetic severity + CDD-11 type training
#
# Every phase/asset variant has an isolated output/checkpoint namespace.
# The official CDD-11 test set is never used here.
#SBATCH --job-name=stage0_variant
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_variant_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_variant_%j.err
#SBATCH --time=2-00:00:00

set -euo pipefail

: "${STAGE0_PHASE:?Set STAGE0_PHASE to D, A, or B}"
: "${STAGE0_VARIANT:?Set STAGE0_VARIANT to proxy or improved}"

if [[ ! "$STAGE0_PHASE" =~ ^[DAB]$ ]]; then
  echo "Invalid STAGE0_PHASE=$STAGE0_PHASE; expected D, A, or B" >&2
  exit 2
fi
if [[ ! "$STAGE0_VARIANT" =~ ^(proxy|improved)$ ]]; then
  echo "Invalid STAGE0_VARIANT=$STAGE0_VARIANT; expected proxy or improved" >&2
  exit 2
fi

PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
SYNTH_ROOT=/scratch/p24cs0203/krish/stage0_synthetic_${STAGE0_VARIANT}
EXPERIMENT_ROOT=/scratch/p24cs0203/krish/outputs/Stage0_${STAGE0_PHASE}_${STAGE0_VARIANT}
CHECKPOINT_ROOT=/scratch/p24cs0203/krish/checkpoints/Stage0_${STAGE0_PHASE}_${STAGE0_VARIANT}
STATE_ROOT=/scratch/p24cs0203/krish/training_states/Stage0_${STAGE0_PHASE}_${STAGE0_VARIANT}
LOG_ROOT=$EXPERIMENT_ROOT
TB_ROOT=/scratch/p24cs0203/krish/tb_logger/Stage0_${STAGE0_PHASE}_${STAGE0_VARIANT}
D_CHECKPOINT_ROOT=/scratch/p24cs0203/krish/checkpoints/Stage0_D_${STAGE0_VARIANT}

cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
# Each simultaneous single-GPU job on cn07 must use a different rendezvous
# port.  Derive it from the Slurm job ID instead of sharing run_experiments'
# default 29500.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

case "$STAGE0_PHASE" in
  D)
    STAGE=stage0_synthetic_pretrain
    CONFIG=options/cdd_experiments/stage0_synthetic_pretrain.yml
    PRETRAIN=null
    ;;
  A)
    STAGE=stage0_sequential_cdd_finetune
    CONFIG=options/cdd_experiments/stage0_sequential_cdd_finetune.yml
    PRETRAIN="$D_CHECKPOINT_ROOT/net_g_latest.pth"
    ;;
  B)
    STAGE=stage0_mixed_finetune
    CONFIG=options/cdd_experiments/stage0_mixed_finetune.yml
    PRETRAIN="$D_CHECKPOINT_ROOT/net_g_latest.pth"
    ;;
esac

if [[ "$STAGE0_PHASE" != D && ! -f "$PRETRAIN" ]]; then
  echo "Missing required D checkpoint: $PRETRAIN" >&2
  exit 3
fi

FORCE=(
  "name=Stage0_${STAGE0_PHASE}_${STAGE0_VARIANT}"
  "path:experiments_root=$EXPERIMENT_ROOT"
  "path:models=$CHECKPOINT_ROOT"
  "path:training_states=$STATE_ROOT"
  "path:log=$LOG_ROOT"
  "path:visualization=$EXPERIMENT_ROOT/visualization"
  "path:tb_logger=$TB_ROOT"
)

if [[ "$STAGE0_PHASE" == D ]]; then
  FORCE+=(
    "datasets:train:manifest=$SYNTH_ROOT/manifests/train.jsonl"
    "datasets:val:manifest=$SYNTH_ROOT/manifests/val.jsonl"
  )
elif [[ "$STAGE0_PHASE" == A ]]; then
  FORCE+=("path:pretrain_network_g=$PRETRAIN")
else
  FORCE+=(
    "datasets:train_synthetic:manifest=$SYNTH_ROOT/manifests/train.jsonl"
    "datasets:val_synthetic:manifest=$SYNTH_ROOT/manifests/val.jsonl"
    "path:pretrain_network_g=$PRETRAIN"
  )
fi

echo "[INFO] Stage0 phase=$STAGE0_PHASE variant=$STAGE0_VARIANT"
echo "[INFO] config=$CONFIG"
echo "[INFO] synthetic_root=$SYNTH_ROOT"
echo "[INFO] experiment_root=$EXPERIMENT_ROOT"
echo "[INFO] checkpoint_root=$CHECKPOINT_ROOT"
echo "[INFO] master_port=$MASTER_PORT"

RUN_ARGS=(--force_yml "${FORCE[@]}")
if [[ "${STAGE0_AUTO_RESUME:-0}" == "1" ]]; then
  RUN_ARGS+=(--auto_resume)
  echo "[INFO] auto_resume=1"
fi

bash run_experiments.sh "$STAGE" "${RUN_ARGS[@]}"
