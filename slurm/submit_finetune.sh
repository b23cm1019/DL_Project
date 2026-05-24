#!/bin/bash
# Submit Row C or Row D finetune. Run after Row C pretrain completes.
#
# Usage:
#   bash slurm/submit_finetune.sh C   → Row C finetune
#   bash slurm/submit_finetune.sh D   → Row D finetune (uses Row C checkpoint)

set -euo pipefail

PARTITION="phd"
ACCOUNT="root"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

ROW="${1:-}"
if [ -z "$ROW" ]; then
    echo "Usage: bash slurm/submit_finetune.sh <C|D>"
    exit 1
fi

ROW=$(echo "$ROW" | tr '[:lower:]' '[:upper:]')
case "$ROW" in
    C) SCRIPT="slurm/finetune_rowC.sh" ;;
    D) SCRIPT="slurm/finetune_rowD.sh" ;;
    *)
        echo "Unknown row: $ROW. Use C or D only."
        exit 1
        ;;
esac

CURRENT_JOBS=$(squeue --me --noheader | wc -l)
echo "[INFO] You currently have ${CURRENT_JOBS} job(s) in the queue."
if [ "$CURRENT_JOBS" -ge 3 ]; then
    echo "[ERROR] Queue full (${CURRENT_JOBS}/3 slots used). Wait for a job to finish."
    echo "        squeue --me"
    exit 1
fi

SBATCH_ARGS="--partition=${PARTITION} --account=${ACCOUNT} --nodelist=cn07"

echo "Submitting Row ${ROW} fine-tuning..."
JOB_ID=$(sbatch --parsable ${SBATCH_ARGS} ${SCRIPT})
echo "  Row ${ROW} finetune → job ${JOB_ID}"
echo ""
echo "Monitor : squeue --me"
echo "Log     : tail -f /scratch/p24cs0203/krish/logs/finetune_row${ROW}_${JOB_ID}.out"
