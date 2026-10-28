#!/bin/bash
# Independent Proposed-Restormer Stage 2: B=8/rank = 4 CDD + 4 synthetic.
#SBATCH --job-name=sevrest_s2
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/p26cs0003/krish/logs/sevrest_s2_%j.out
#SBATCH --error=/scratch/p26cs0003/krish/logs/sevrest_s2_%j.err
set -euo pipefail

PROJECT_ROOT=/csehome/p26cs0003/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
[[ "$(python -c 'import torch; print(torch.cuda.device_count())')" == "2" ]] || {
  echo "This run requires exactly two Slurm-visible GPUs." >&2; exit 2;
}
RESUME=()
if [[ "${SEVERITY_AUTO_RESUME:-0}" == "1" ]]; then RESUME=(--auto_resume); fi
RUN_MASTER_PORT="${SEVERITY_RESTORMER_STAGE2_MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 40000)))}"
python -m torch.distributed.run --nproc_per_node=2 --master-port "$RUN_MASTER_PORT" \
  basicsr/severity_train.py -opt options/cdd_experiments/severity_restormer_stage2_cdd_adapt_2gpu.yml \
  --launcher pytorch "${RESUME[@]}"
