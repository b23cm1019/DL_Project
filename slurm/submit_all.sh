#!/bin/bash
# Submit the full experiment pipeline for cn07 (A6000 48GB, Config B).
# Pre-trains for Row B and Row C run first (can overlap on separate GPUs).
# Fine-tuning jobs are submitted with afterok dependencies.
#
# Usage: bash slurm/submit_all.sh
#
# To run Row B and Row C pretraining in parallel on two different A6000 GPUs,
# edit the CUDA_VISIBLE_DEVICES lines in each slurm script before submitting.

set -euo pipefail

mkdir -p logs

# ── Pre-training (submit both; they can run on different A6000 GPUs) ───────────
echo "Submitting Row B pre-training..."
JOB_PTB=$(sbatch --parsable slurm/pretrain_rowB.sh)
echo "  Row B pretrain → job ${JOB_PTB}"

echo "Submitting Row C pre-training..."
JOB_PTC=$(sbatch --parsable slurm/pretrain_rowC.sh)
echo "  Row C pretrain → job ${JOB_PTC}"

# ── Fine-tuning (each waits for the relevant pretrain to finish) ──────────────
echo "Submitting Row B fine-tuning (after job ${JOB_PTB})..."
JOB_FTB=$(sbatch --parsable --dependency=afterok:${JOB_PTB} slurm/finetune_rowB.sh)
echo "  Row B finetune → job ${JOB_FTB}"

echo "Submitting Row C fine-tuning (after job ${JOB_PTC})..."
JOB_FTC=$(sbatch --parsable --dependency=afterok:${JOB_PTC} slurm/finetune_rowC.sh)
echo "  Row C finetune → job ${JOB_FTC}"

echo "Submitting Row D fine-tuning (after job ${JOB_PTC})..."
JOB_FTD=$(sbatch --parsable --dependency=afterok:${JOB_PTC} slurm/finetune_rowD.sh)
echo "  Row D finetune → job ${JOB_FTD}"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue --me"
echo "  watch -n 30 'squeue --me'"
