#!/usr/bin/env bash
# Experiment 2: public DCPT NAFNet initialization + B_improved conditioning.
#
# Before the first run, set the path to the checkpoint downloaded from the
# official DCPT release:
#   export STAGE0_PUBLIC_NAFNET_CKPT=/scratch/p24cs0203/krish/checkpoints/public/dcpt_nafnet_cdd.pth
#
# First run:
#   sbatch --export=NONE,STAGE0_PUBLIC_NAFNET_CKPT=/path/to/dcpt_nafnet_cdd.pth,STAGE0_R4_PUBLIC_TOTAL_ITERS=100000,STAGE0_R4_PUBLIC_AUTO_RESUME=0 slurm/stage0_train_public_r4.sh
# Continue to 250k/500k/750k by changing only TOTAL_ITERS and using AUTO_RESUME=1.
# Do not change the experiment name or output paths.
#
# If the public checkpoint uses a different state-dict key, override:
#   STAGE0_PUBLIC_PARAM_KEY=params
#
#SBATCH --job-name=stage0_r4_public
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_r4_public_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_r4_public_%j.err

set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
NAME=Stage0_R4_public_full_improved
TOTAL_ITERS=${STAGE0_R4_PUBLIC_TOTAL_ITERS:-100000}
SAVE_FREQ=${STAGE0_R4_PUBLIC_SAVE_FREQ:-10000}
VAL_FREQ=${STAGE0_R4_PUBLIC_VAL_FREQ:-10000}
PUBLIC_CKPT=${STAGE0_PUBLIC_NAFNET_CKPT:-}
PARAM_KEY=${STAGE0_PUBLIC_PARAM_KEY:-params_ema}

[[ -n "$PUBLIC_CKPT" && -f "$PUBLIC_CKPT" ]] || {
  echo "Missing STAGE0_PUBLIC_NAFNET_CKPT: $PUBLIC_CKPT" >&2
  exit 2
}
[[ "$TOTAL_ITERS" =~ ^[0-9]+$ ]] || { echo "Invalid TOTAL_ITERS" >&2; exit 2; }
(( TOTAL_ITERS >= 100000 && TOTAL_ITERS <= 750000 )) || {
  echo "TOTAL_ITERS must be 100000..750000" >&2
  exit 2
}

cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
ROOT=/scratch/p24cs0203/krish/stage0_official_r4_improved

for p in "$ROOT/manifests/train_predicted_severity.jsonl" \
         "$ROOT/manifests/val_predicted_severity.jsonl" "$PUBLIC_CKPT"; do
  [[ -e "$p" ]] || { echo "Missing required path: $p" >&2; exit 3; }
done

OVERRIDES=(
  "name=$NAME"
  "datasets:train:manifest=$ROOT/manifests/train_predicted_severity.jsonl"
  "datasets:val:manifest=$ROOT/manifests/val_predicted_severity.jsonl"
  "path:pretrain_network_g=$PUBLIC_CKPT"
  "path:param_key_g=$PARAM_KEY"
  "path:experiments_root=/scratch/p24cs0203/krish/outputs/$NAME"
  "path:models=/scratch/p24cs0203/krish/checkpoints/$NAME"
  "path:training_states=/scratch/p24cs0203/krish/training_states/$NAME"
  "path:log=/scratch/p24cs0203/krish/logs/$NAME"
  "path:visualization=/scratch/p24cs0203/krish/outputs/$NAME/visualization"
  "path:tb_logger=/scratch/p24cs0203/krish/tb_logger/$NAME"
  "train:total_iter=$TOTAL_ITERS"
  "train:scheduler:periods=[750000]"
  "val:val_freq=$VAL_FREQ"
  "logger:save_checkpoint_freq=$SAVE_FREQ"
)

RESUME=()
if [[ "${STAGE0_R4_PUBLIC_AUTO_RESUME:-0}" == 1 ]]; then RESUME=(--auto_resume); fi
echo "[INFO] name=$NAME total_iters=$TOTAL_ITERS save_freq=$SAVE_FREQ auto_resume=${STAGE0_R4_PUBLIC_AUTO_RESUME:-0}"
echo "[INFO] initialization=official DCPT public checkpoint: $PUBLIC_CKPT"
echo "[INFO] extractor=B_improved frozen; restoration=full fine-tune; scheduler_horizon=750000"

python -m torch.distributed.run --master-port "$MASTER_PORT" --nproc_per_node=1 \
  basicsr/all_in_one_train.py \
  -opt options/cdd_experiments/stage0_r4_public_full.yml \
  --launcher pytorch "${RESUME[@]}" --force_yml "${OVERRIDES[@]}"
