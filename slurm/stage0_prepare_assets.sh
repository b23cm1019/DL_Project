#!/bin/bash
#SBATCH --job-name=stage0_assets
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/stage0_assets_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/stage0_assets_%j.err
#SBATCH --time=04:00:00
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

python scripts/prepare_stage0_assets.py \
  --clean-root /scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train/clear \
  --onerestore-assets "$PROJECT_ROOT/third_party/OneRestore/syn_data/data" \
  --output-root /scratch/p24cs0203/krish/stage0_assets_proxy \
  --mode proxy
