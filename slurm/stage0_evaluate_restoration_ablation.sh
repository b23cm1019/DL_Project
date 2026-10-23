#!/usr/bin/env bash
# Evaluate one R0-R4 restoration ablation on a selected synthetic split.
#
# Example:
#   sbatch --export=NONE,STAGE0_RESTORE_VARIANT=improved,STAGE0_RESTORE_ABLATION=R4,STAGE0_RESTORE_SPLIT=test \
#     slurm/stage0_evaluate_restoration_ablation.sh

#SBATCH --job-name=stage0_restore_eval
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_restore_eval_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_restore_eval_%j.err

set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
SCRATCH_ROOT=/scratch/p24cs0203/krish
VARIANT=${STAGE0_RESTORE_VARIANT:-improved}
ABLATION=${STAGE0_RESTORE_ABLATION:-R4}
SPLIT=${STAGE0_RESTORE_SPLIT:-test}
ROOT="$SCRATCH_ROOT/stage0_restoration_${VARIANT}"
NAME="Stage0_${ABLATION}_${VARIANT}"

[[ "$VARIANT" == proxy || "$VARIANT" == improved ]] || { echo "Invalid variant" >&2; exit 2; }
[[ "$ABLATION" =~ ^R[0-4]$ ]] || { echo "Invalid ablation; expected R0-R4" >&2; exit 2; }
[[ "$SPLIT" == test || "$SPLIT" == unseen ]] || { echo "Invalid split; expected test or unseen" >&2; exit 2; }

case "$ABLATION" in
  R0) MODE=none; DIM=0; CHECKPOINT="$SCRATCH_ROOT/checkpoints/Row_C_MultiLabel_Finetune_CDD11/net_g_latest.pth";;
  R1) MODE=oracle_type; DIM=4; CHECKPOINT="$SCRATCH_ROOT/checkpoints/$NAME/net_g_latest.pth";;
  R2) MODE=predicted_type; DIM=4; CHECKPOINT="$SCRATCH_ROOT/checkpoints/$NAME/net_g_latest.pth";;
  R3) MODE=oracle_severity; DIM=16; CHECKPOINT="$SCRATCH_ROOT/checkpoints/$NAME/net_g_latest.pth";;
  R4) MODE=predicted_severity; DIM=16; CHECKPOINT="$SCRATCH_ROOT/checkpoints/$NAME/net_g_latest.pth";;
esac

cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
MANIFEST="$ROOT/manifests/${SPLIT}.jsonl"
[[ "$ABLATION" == R0 ]] || MANIFEST="$ROOT/manifests/${SPLIT}_${MODE}.jsonl"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 3; }
[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 4; }

OUTPUT="$SCRATCH_ROOT/restoration_evaluations/${NAME}/${SPLIT}"
python scripts/evaluate_stage0_restoration.py \
  --mode "$ABLATION" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$MANIFEST" \
  --condition-dim "$DIM" \
  --output "$OUTPUT" \
  --device cuda

echo "[RESULT] Wrote $OUTPUT/summary.json"
