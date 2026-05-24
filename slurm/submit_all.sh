#!/bin/bash
# Submit the full DCPT experiment pipeline.
# Usage: bash slurm/submit_all.sh   (run from project root)

set -euo pipefail

PARTITION="phd"
ACCOUNT="root"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
mkdir -p logs

echo "Working directory: $(pwd)"
echo "Using partition=${PARTITION} account=${ACCOUNT}"
echo ""

SBATCH_ARGS="--partition=${PARTITION} --account=${ACCOUNT} --nodelist=cn07"

echo "Submitting Row B pre-training..."
JOB_PTB=$(sbatch --parsable ${SBATCH_ARGS} slurm/pretrain_rowB.sh)
echo "  Row B pretrain → job ${JOB_PTB}"

echo "Submitting Row C pre-training..."
JOB_PTC=$(sbatch --parsable ${SBATCH_ARGS} slurm/pretrain_rowC.sh)
echo "  Row C pretrain → job ${JOB_PTC}"

echo "Submitting Row B fine-tuning (depends on job ${JOB_PTB})..."
JOB_FTB=$(sbatch --parsable --dependency=afterok:${JOB_PTB} ${SBATCH_ARGS} slurm/finetune_rowB.sh)
echo "  Row B finetune → job ${JOB_FTB}"

echo "Submitting Row C fine-tuning (depends on job ${JOB_PTC})..."
JOB_FTC=$(sbatch --parsable --dependency=afterok:${JOB_PTC} ${SBATCH_ARGS} slurm/finetune_rowC.sh)
echo "  Row C finetune → job ${JOB_FTC}"

echo "Submitting Row D fine-tuning (depends on job ${JOB_PTC})..."
JOB_FTD=$(sbatch --parsable --dependency=afterok:${JOB_PTC} ${SBATCH_ARGS} slurm/finetune_rowD.sh)
echo "  Row D finetune → job ${JOB_FTD}"

echo ""
echo "All 5 jobs submitted:"
echo "  pretrain_rowB : ${JOB_PTB}"
echo "  pretrain_rowC : ${JOB_PTC}"
echo "  finetune_rowB : ${JOB_FTB}  (starts after ${JOB_PTB})"
echo "  finetune_rowC : ${JOB_FTC}  (starts after ${JOB_PTC})"
echo "  finetune_rowD : ${JOB_FTD}  (starts after ${JOB_PTC})"
echo ""
echo "Monitor : squeue --me"
echo "Cancel  : scancel ${JOB_PTB} ${JOB_PTC} ${JOB_FTB} ${JOB_FTC} ${JOB_FTD}"
