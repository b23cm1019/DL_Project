#!/bin/bash
#SBATCH --job-name=stage0_synth
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_synth_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_synth_%j.err
#SBATCH --time=1-00:00:00
set -euo pipefail
PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
cd "$PROJECT_ROOT"
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /etc/profile.d/lmod.sh ]]; then
    source /etc/profile.d/lmod.sh
else
    echo "No module initialization script found" >&2
    exit 1
fi
module purge 2>/dev/null || true
module load python/3.10.pytorch
source /csehome/p24cs0203/krish/envs/dl_project/bin/activate
SYNTH_ROOT=${STAGE0_SYNTH_ROOT:-/scratch/p24cs0203/krish/stage0_synthetic}
REPORT_ROOT=${STAGE0_REPORT_ROOT:-${SYNTH_ROOT}/reports}
CDD_ROOT=${STAGE0_CDD_ROOT:-/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train}
ASSET_ROOT=${STAGE0_ASSET_ROOT:-/scratch/p24cs0203/krish/stage0_assets_proxy}
mkdir -p "${REPORT_ROOT}"
export STAGE0_DATASET_REPORT="${REPORT_ROOT}/stage0_dataset_integrity.json"
CLEAN_ROOT=/scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train/clear
LIGHT_ROOT=${STAGE0_LIGHT_ROOT:-${ASSET_ROOT}/light_map}
DEPTH_ROOT=${STAGE0_DEPTH_ROOT:-${ASSET_ROOT}/depth_map}
RAIN_ROOT=${STAGE0_RAIN_ROOT:-${ASSET_ROOT}/rain_mask}
SNOW_ROOT=${STAGE0_SNOW_ROOT:-${ASSET_ROOT}/snow_mask}
SAMPLES_PER_SCENE=${STAGE0_SAMPLES_PER_SCENE:-12}
MAX_SCENES=${STAGE0_MAX_SCENES:-0}

for required_root in "$CLEAN_ROOT" "$LIGHT_ROOT" "$DEPTH_ROOT" "$RAIN_ROOT" "$SNOW_ROOT" "$CDD_ROOT"; do
  [[ -d "$required_root" ]] || { echo "Missing required directory: $required_root" >&2; exit 2; }
done
echo "[INFO] node=$(hostname) clean_root=$CLEAN_ROOT"
echo "[INFO] clean_files=$(find "$CLEAN_ROOT" -type f | wc -l)"
for required_root in "$CLEAN_ROOT" "$LIGHT_ROOT" "$DEPTH_ROOT" "$RAIN_ROOT" "$SNOW_ROOT"; do
  # Do not use find | grep -q here: with pipefail enabled, grep's early
  # exit can make find receive SIGPIPE and falsely report an empty directory.
  file_count=$(find "$required_root" -type f -print | wc -l)
  if [[ "$file_count" -eq 0 ]]; then
    echo "Required asset directory is empty: $required_root" >&2
    exit 2
  fi
  echo "[INFO] asset_files[$required_root]=$file_count"
done
GENERATOR=scripts/generate_stage0_onerestore.py
echo "[INFO] clean scenes: $(find "$CLEAN_ROOT" -type f | wc -l)"
COMMON=(--clean-root "$CLEAN_ROOT" --light-root "$LIGHT_ROOT" --depth-root "$DEPTH_ROOT"
  --rain-root "$RAIN_ROOT" --snow-root "$SNOW_ROOT" --output-root "$SYNTH_ROOT/images"
  --val-fraction 0.2 --test-fraction 0.1 --unseen-fraction 0.1 --split-seed 0
  --samples-per-scene "$SAMPLES_PER_SCENE" --max-scenes "$MAX_SCENES")
python3 "$GENERATOR" "${COMMON[@]}" --manifest "$SYNTH_ROOT/manifests/train.jsonl" --split train --seed 0
python3 "$GENERATOR" "${COMMON[@]}" --manifest "$SYNTH_ROOT/manifests/val.jsonl" --split val --seed 1
python3 "$GENERATOR" "${COMMON[@]}" --manifest "$SYNTH_ROOT/manifests/test.jsonl" --split test --seed 2
python3 "$GENERATOR" "${COMMON[@]}" --manifest "$SYNTH_ROOT/manifests/unseen_rain_snow.jsonl" \
  --split unseen --unseen-combination rain_snow --seed 3

python3 scripts/validate_stage0_severity.py \
  --manifest "$SYNTH_ROOT/manifests/train.jsonl" \
  --report "$SYNTH_ROOT/reports/train_severity_proxies.json"
python3 scripts/validate_stage0_severity.py \
  --manifest "$SYNTH_ROOT/manifests/val.jsonl" \
  --report "$SYNTH_ROOT/reports/val_severity_proxies.json"
python3 scripts/validate_stage0_severity.py \
  --manifest "$SYNTH_ROOT/manifests/test.jsonl" \
  --report "$SYNTH_ROOT/reports/test_severity_proxies.json"
python3 scripts/validate_stage0_dataset.py \
  --manifest train="$SYNTH_ROOT/manifests/train.jsonl" \
  val="$SYNTH_ROOT/manifests/val.jsonl" \
  test="$SYNTH_ROOT/manifests/test.jsonl" \
  unseen="$SYNTH_ROOT/manifests/unseen_rain_snow.jsonl"

python3 scripts/build_cdd11_scene_split.py \
  --root "${CDD_ROOT}" \
  --categories clear low haze rain snow low_haze low_rain low_snow haze_rain haze_snow low_haze_rain low_haze_snow \
  --output /scratch/p24cs0203/krish/stage0_cdd_split.json
