#!/usr/bin/env bash
# Train one Stage-0 restoration ablation with a frozen Row-C restoration base.
#
# R1: oracle type (4D)
# R2: B_improved predicted type (4D)
# R3: oracle severity distribution (16D)
# R4: B_improved predicted severity distribution (16D)

#SBATCH --job-name=stage0_restore
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_restore_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_restore_%j.err

set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
SCRATCH_ROOT=/scratch/p24cs0203/krish
VARIANT=${STAGE0_RESTORE_VARIANT:-improved}
ABLATION=${STAGE0_RESTORE_ABLATION:-R1}
ROOT="$SCRATCH_ROOT/stage0_restoration_${VARIANT}"
NAME="Stage0_${ABLATION}_${VARIANT}"

[[ "$VARIANT" == proxy || "$VARIANT" == improved ]] || { echo "Invalid variant: $VARIANT" >&2; exit 2; }
[[ "$ABLATION" =~ ^R[1-4]$ ]] || { echo "Invalid ablation: $ABLATION; expected R1-R4" >&2; exit 2; }

case "$ABLATION" in
  R1) MODE=oracle_type; DIM=4;;
  R2) MODE=predicted_type; DIM=4;;
  R3) MODE=oracle_severity; DIM=16;;
  R4) MODE=predicted_severity; DIM=16;;
esac

TRAIN_MANIFEST="$ROOT/manifests/train_${MODE}.jsonl"
VAL_MANIFEST="$ROOT/manifests/val_${MODE}.jsonl"
for path in "$TRAIN_MANIFEST" "$VAL_MANIFEST" \
  "$SCRATCH_ROOT/checkpoints/Row_C_MultiLabel_Finetune_CDD11/net_g_latest.pth"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 3; }
done

cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

OVERRIDES=(
  "name=$NAME"
  "datasets:train:manifest=$TRAIN_MANIFEST"
  "datasets:val:manifest=$VAL_MANIFEST"
  "datasets:train:condition_dim=$DIM"
  "datasets:val:condition_dim=$DIM"
  "network_g:feedback_dim=$DIM"
  "path:experiments_root=$SCRATCH_ROOT/outputs/$NAME"
  "path:models=$SCRATCH_ROOT/checkpoints/$NAME"
  "path:training_states=$SCRATCH_ROOT/training_states/$NAME"
  "path:log=$SCRATCH_ROOT/logs/$NAME"
  "path:visualization=$SCRATCH_ROOT/outputs/$NAME/visualization"
  "path:tb_logger=$SCRATCH_ROOT/tb_logger/$NAME"
  "train:total_iter=${STAGE0_RESTORE_ITERS:-10000}"
  "train:scheduler:periods=[${STAGE0_RESTORE_ITERS:-10000}]"
  "val:val_freq=${STAGE0_RESTORE_VAL_FREQ:-1000}"
  "logger:save_checkpoint_freq=${STAGE0_RESTORE_SAVE_FREQ:-1000}"
)

echo "[INFO] ablation=$ABLATION variant=$VARIANT mode=$MODE dim=$DIM"
echo "[INFO] train_manifest=$TRAIN_MANIFEST"
echo "[INFO] val_manifest=$VAL_MANIFEST"

RESUME=()
if [[ "${STAGE0_RESTORE_AUTO_RESUME:-0}" == 1 ]]; then
  RESUME=(--auto_resume)
fi

python -m torch.distributed.run \
  --master-port "$MASTER_PORT" \
  --nproc_per_node=1 \
  basicsr/all_in_one_train.py \
  -opt options/cdd_experiments/stage0_restoration_conditioned.yml \
  --launcher pytorch \
  "${RESUME[@]}" \
  --force_yml "${OVERRIDES[@]}"
