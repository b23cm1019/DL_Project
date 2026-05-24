#!/bin/bash
# Submit Row C pretrain → then Row C and Row D finetune.
#
# QOS limits (phd partition, shared account):
#   MaxSubmitPU = 3  (submitted + running)
#   MaxJobsPU   = 2  (running at once)
#   MaxTRESPU   = gres/gpu=2
#
# Strategy:
#   1. Submit Row C pretrain now (1 job).
#   2. After it finishes, run: bash slurm/submit_finetune.sh C
#                          and: bash slurm/submit_finetune.sh D

set -euo pipefail

PARTITION="phd"
ACCOUNT="root"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Working directory: $(pwd)"
echo "Using partition=${PARTITION} account=${ACCOUNT}"
echo ""

# ── Check current job count ────────────────────────────────────────────────────
CURRENT_JOBS=$(squeue --me --noheader | wc -l)
echo "[INFO] You currently have ${CURRENT_JOBS} job(s) in the queue."
if [ "$CURRENT_JOBS" -ge 3 ]; then
    echo "[ERROR] Queue full (${CURRENT_JOBS}/3 slots used). Wait for a job to finish."
    echo "        squeue --me"
    exit 1
fi

SBATCH_ARGS="--partition=${PARTITION} --account=${ACCOUNT} --nodelist=cn07"

# ── Submit Row C pretrain only ─────────────────────────────────────────────────
echo "Submitting Row C pre-training..."
JOB_PTC=$(sbatch --parsable ${SBATCH_ARGS} slurm/pretrain_rowC.sh)
echo "  Row C pretrain → job ${JOB_PTC}"

echo ""
echo "Next steps — after Row C pretrain (job ${JOB_PTC}) finishes:"
echo ""
echo "  bash slurm/submit_finetune.sh C   # Row C finetune"
echo "  bash slurm/submit_finetune.sh D   # Row D finetune (uses Row C checkpoint)"
echo ""
echo "Monitor : squeue --me"
echo "Log     : tail -f /scratch/p24cs0203/krish/logs/pretrain_rowC_${JOB_PTC}.out"
echo "Cancel  : scancel ${JOB_PTC}"
