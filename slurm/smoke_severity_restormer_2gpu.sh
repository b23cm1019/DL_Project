#!/bin/bash
# Run this before any long Proposed-Restormer launch on a two-GPU node.
#SBATCH --job-name=sevrest_probe
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/p26cs0003/krish/logs/sevrest_probe_%j.out
#SBATCH --error=/scratch/p26cs0003/krish/logs/sevrest_probe_%j.err
set -euo pipefail

PROJECT_ROOT=/csehome/p26cs0003/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
[[ "$(python -c 'import torch; print(torch.cuda.device_count())')" == "2" ]] || {
  echo "This probe requires exactly two Slurm-visible GPUs." >&2; exit 2;
}
python scripts/smoke_test_severity_restormer.py
RUN_MASTER_PORT="${SEVERITY_RESTORMER_PROBE_MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 40000)))}"
python -m torch.distributed.run --nproc_per_node=2 --master-port "$RUN_MASTER_PORT" \
  scripts/probe_severity_restormer_ddp.py
