#!/bin/bash
# Stage 3b: full fine-tune of the severity-conditioned restorer (400k iters). Auto-resumes.
#SBATCH --job-name=sevfull_s3b
#SBATCH --partition=phd
#SBATCH --nodelist=cn07
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=5-18:00:00
#SBATCH --output=/scratch/p26cs0003/krish/logs/sevfull_s3b_%j.out
#SBATCH --error=/scratch/p26cs0003/krish/logs/sevfull_s3b_%j.err
set -euo pipefail

PROJECT_ROOT=/csehome/p26cs0003/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
# Isolate rendezvous ports across simultaneous cn07 jobs.
RUN_MASTER_PORT="$((20000 + (SLURM_JOB_ID % 40000)))"
echo "[INFO] torchrun master port: ${RUN_MASTER_PORT}"
python -m torch.distributed.run --nproc_per_node=1 --master-port "${RUN_MASTER_PORT}" \
  basicsr/severity_train.py \
  -opt options/cdd_experiments/severity_full_stage3b.yml \
  --launcher pytorch --auto_resume
