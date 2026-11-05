#!/bin/bash
# T5 step 1/2: in-protocol sources + render plan, then clean/light/depth assets (MiDaS, GPU).
# Reuses scripts/prepare_lsdir_severity_assets.py unchanged. Run from the repo root:
#   sbatch --job-name=abl_t5_assets ablations/slurm/t5_assets.sh
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=ablations/logs/%x_%j.out
#SBATCH --error=ablations/logs/%x_%j.err
set -euo pipefail
: "${OUT_ROOT:?export OUT_ROOT first, e.g. /scratch/p26cs0003/krish/sevdcpt}"
cd "${SLURM_SUBMIT_DIR}"
if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; else source /etc/profile.d/lmod.sh; fi
module purge 2>/dev/null || true
module load python/3.10.pytorch
source "${ASSET_VENV:-/scratch/p26cs0003/krish/envs/lsdir_midas_assets_py310}/bin/activate"
export TORCH_HOME="${TORCH_HOME:-/scratch/p26cs0003/krish/severity_cdd11_lsdir_outdoor_v1/torch_cache}"
T5="${OUT_ROOT}/ablations/t5_inprotocol"
if [[ ! -f "${T5}/plans/train.jsonl" ]]; then
  python ablations/scripts/build_inprotocol_plan.py --out_root "${T5}"
fi
if [[ ! -f "${T5}/assets/asset_provenance.json" ]]; then
  python scripts/prepare_lsdir_severity_assets.py \
    --source-manifest "${T5}/sources/train_sources.jsonl" \
    --source-manifest "${T5}/sources/val_sources.jsonl" \
    --output-root "${T5}/assets" \
    --rain-mask /csehome/p26cs0003/krish/projects/OneRestore/img_file/rain_mask.jpg \
    --snow-mask /csehome/p26cs0003/krish/projects/OneRestore/img_file/snow_mask.png \
    --max-long-side 512 --min-short-side 128 --device cuda
fi
echo "[INFO] T5 assets ready: ${T5}/assets/source_assets.jsonl"
