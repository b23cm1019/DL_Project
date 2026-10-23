#!/usr/bin/env bash
# Build paired synthetic restoration manifests and cache R1-R4 conditions.
#
# Run once for the selected extractor:
#   sbatch --export=NONE,STAGE0_RESTORE_VARIANT=improved \
#     slurm/stage0_prepare_restoration_conditions.sh

#SBATCH --job-name=stage0_restore_prep
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_restore_prep_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_restore_prep_%j.err

set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
SCRATCH_ROOT=/scratch/p24cs0203/krish
VARIANT=${STAGE0_RESTORE_VARIANT:-improved}
EXTRACTOR="$SCRATCH_ROOT/checkpoints/Stage0_B_${VARIANT}/net_g_latest.pth"
SYNTH_ROOT="$SCRATCH_ROOT/stage0_synthetic_${VARIANT}"
RESTORE_ROOT="$SCRATCH_ROOT/stage0_restoration_${VARIANT}"

[[ "$VARIANT" == proxy || "$VARIANT" == improved ]] || { echo "Invalid variant: $VARIANT" >&2; exit 2; }
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
mkdir -p "$RESTORE_ROOT/manifests" "$RESTORE_ROOT/conditions"
[[ -f "$EXTRACTOR" ]] || { echo "Missing extractor checkpoint: $EXTRACTOR" >&2; exit 3; }

for split in train val test unseen; do
  if [[ "$split" == unseen ]]; then
    input="$SYNTH_ROOT/manifests/unseen_rain_snow.jsonl"
  else
    input="$SYNTH_ROOT/manifests/${split}.jsonl"
  fi
  output="$RESTORE_ROOT/manifests/${split}.jsonl"
  [[ -f "$input" ]] || { echo "Missing manifest: $input" >&2; exit 4; }
  python scripts/build_stage0_restoration_manifests.py --input "$input" --output "$output"
done

for split in train val test unseen; do
  for mode in oracle_type predicted_type oracle_severity predicted_severity; do
    dim=4
    [[ "$mode" == oracle_severity || "$mode" == predicted_severity ]] && dim=16
    python scripts/cache_stage0_restoration_conditions.py \
      --manifest "$RESTORE_ROOT/manifests/${split}.jsonl" \
      --checkpoint "$EXTRACTOR" \
      --output-root "$RESTORE_ROOT/conditions/${mode}" \
      --output-manifest "$RESTORE_ROOT/manifests/${split}_${mode}.jsonl" \
      --mode "$mode" \
      --device cuda
    echo "[INFO] cached split=$split mode=$mode dim=$dim"
  done
done

echo "[RESULT] Prepared Stage 0 restoration conditions under $RESTORE_ROOT"
