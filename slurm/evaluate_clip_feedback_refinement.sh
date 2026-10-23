#!/bin/bash
#SBATCH --job-name=eval_clip_refine
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/eval_clip_refine_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/eval_clip_refine_%j.err
#SBATCH --time=1-00:00:00
set -euo pipefail
cd /csehome/p24cs0203/krish/projects/DL_Project
source slurm/_cluster_env.sh
python scripts/evaluate_refinement_conditions.py \
  --manifest "${EVAL_MANIFEST:-/scratch/p24cs0203/krish/refinement_poc/manifests/val_clip.jsonl}" \
  --m2-checkpoint "${M2_CHECKPOINT:-/scratch/p24cs0203/krish/checkpoints/CLIP_Feedback_Refinement_RowC/net_g_50000.pth}" \
  --rowc-checkpoint "${ROWC_CHECKPOINT:-/scratch/p24cs0203/krish/checkpoints/Row_C_MultiLabel_Finetune_CDD11/net_g_latest.pth}" \
  --output "${EVAL_OUTPUT:-/scratch/p24cs0203/krish/results/CLIP_Feedback_Refinement_RowC}"
