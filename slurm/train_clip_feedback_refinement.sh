#!/bin/bash
#SBATCH --job-name=rowc_clip_refine
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/rowc_clip_refine_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/rowc_clip_refine_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail
cd /csehome/p24cs0203/krish/projects/DL_Project
source slurm/_cluster_env.sh
train_args=(
  -m torch.distributed.run --nproc_per_node=1 basicsr/train.py
  -opt options/cdd_experiments/train_clip_feedback_refinement.yml
  --launcher pytorch
)
if [[ -n "${REFINEMENT_TOTAL_ITER:-}" ]]; then
  train_args+=(--force_yml "train:total_iter=${REFINEMENT_TOTAL_ITER}")
fi
if [[ -n "${REFINEMENT_BATCH_SIZE:-}" ]]; then
  train_args+=(--force_yml "datasets:train:batch_size_per_gpu=${REFINEMENT_BATCH_SIZE}")
fi
python "${train_args[@]}"
