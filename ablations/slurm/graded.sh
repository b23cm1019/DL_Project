#!/bin/bash
# Build the severity-graded CDD-11 test set (50 clean test images x 11 compositions x 4 levels = 2,200).
# MiDaS depth needs a GPU; uses the same MiDaS venv and model cache as the Stage 1 asset job.
# Run from the repo root, after `export OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt`:
#   sbatch --job-name=abl_graded ablations/slurm/graded.sh
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
python ablations/scripts/build_graded_testset.py --out "${OUT_ROOT}/ablations/graded_cdd11" ${EXTRA:-}
