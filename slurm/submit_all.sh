#!/bin/bash

set -euo pipefail

PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"

cd "${PROJECT_ROOT}"

echo "Working directory: $(pwd)"
echo "Using partition=phd account=root"

echo ""
echo "[INFO] You currently have $(squeue --me | tail -n +2 | wc -l) job(s) in the queue."

echo "Submitting Row C pre-training..."

JOB_ID=$(sbatch slurm/pretrain_rowC.sh | awk '{print $4}')

echo "  Row C pretrain → job ${JOB_ID}"

echo ""
echo "Next steps — after Row C pretrain (job ${JOB_ID}) finishes:"

echo ""
echo "  bash slurm/submit_finetune.sh C"
echo "  bash slurm/submit_finetune.sh D"

echo ""
echo "Monitor : squeue --me"
echo "Log     : tail -f /csehome/p24cs0203/krish/logs/slurm/pretrain_rowC_${JOB_ID}.out"
echo "Cancel  : scancel ${JOB_ID}"
