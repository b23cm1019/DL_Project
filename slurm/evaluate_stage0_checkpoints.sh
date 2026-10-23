#!/usr/bin/env bash
# Evaluate Stage 0 D/A/B checkpoints on the held-out test manifests.
#
# Usage:
#   sbatch --export=NONE,STAGE0_EVAL_VARIANT=proxy \
#     slurm/evaluate_stage0_checkpoints.sh
#
# Include B when its checkpoint is available:
#   sbatch --export=NONE,STAGE0_EVAL_VARIANT=proxy,STAGE0_EVAL_INCLUDE_B=1 \
#     slurm/evaluate_stage0_checkpoints.sh
#
# Submit a separate job for STAGE0_EVAL_VARIANT=improved.

#SBATCH --job-name=stage0_eval
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_eval_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_eval_%j.err

set -euo pipefail

PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
SCRATCH_ROOT=/scratch/p24cs0203/krish
VARIANT=${STAGE0_EVAL_VARIANT:-proxy}
INCLUDE_B=${STAGE0_EVAL_INCLUDE_B:-0}

if [[ "$VARIANT" != proxy && "$VARIANT" != improved ]]; then
  echo "Invalid STAGE0_EVAL_VARIANT=$VARIANT; expected proxy or improved" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PYTHON_BIN=${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}
PYTHON_BIN=${PYTHON_BIN:-python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 3
fi

CHECKPOINT_ROOT="$SCRATCH_ROOT/checkpoints"
SYNTH_ROOT="$SCRATCH_ROOT/stage0_synthetic_${VARIANT}"
EVAL_ROOT="$SCRATCH_ROOT/evaluations"
mkdir -p "$EVAL_ROOT"

TEST_MANIFEST="$SYNTH_ROOT/manifests/test.jsonl"
UNSEEN_MANIFEST="$SYNTH_ROOT/manifests/unseen_rain_snow.jsonl"
CDD11_TEST_ROOT="$SCRATCH_ROOT/datasets/CDD11/test"

MODEL_D="$CHECKPOINT_ROOT/Stage0_D_${VARIANT}/net_g_latest.pth"
MODEL_A="$CHECKPOINT_ROOT/Stage0_A_${VARIANT}/net_g_latest.pth"
MODEL_B="$CHECKPOINT_ROOT/Stage0_B_${VARIANT}/net_g_latest.pth"
OUTPUT="$EVAL_ROOT/stage0_D_A_${VARIANT}.json"

for required in "$TEST_MANIFEST" "$UNSEEN_MANIFEST" "$CDD11_TEST_ROOT" "$MODEL_D" "$MODEL_A"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required evaluation input: $required" >&2
    exit 4
  fi
done

MODEL_ARGS=(
  --model "D_${VARIANT}=$MODEL_D"
  --model "A_${VARIANT}=$MODEL_A"
)

if [[ "$INCLUDE_B" == 1 ]]; then
  if [[ ! -f "$MODEL_B" ]]; then
    echo "B checkpoint requested but missing: $MODEL_B" >&2
    exit 5
  fi
  MODEL_ARGS+=(--model "B_${VARIANT}=$MODEL_B")
  OUTPUT="$EVAL_ROOT/stage0_D_A_B_${VARIANT}.json"
fi

echo "[INFO] variant=$VARIANT"
echo "[INFO] python=$PYTHON_BIN"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] output=$OUTPUT"
echo "[INFO] models: ${MODEL_ARGS[*]}"

"$PYTHON_BIN" scripts/evaluate_stage0_checkpoints.py \
  "${MODEL_ARGS[@]}" \
  --synthetic-test "$TEST_MANIFEST" \
  --synthetic-unseen "$UNSEEN_MANIFEST" \
  --cdd11-test-root "$CDD11_TEST_ROOT" \
  --output "$OUTPUT" \
  --batch-size 64 \
  --workers 4 \
  --device cuda

echo "[RESULT] Evaluation completed: $OUTPUT"
