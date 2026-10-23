#!/bin/bash
#SBATCH --job-name=test_rowC_s2
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/test_rowC_s2_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/test_rowC_s2_%j.err
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
LOG_ROOT="/csehome/p24cs0203/krish/logs/slurm"

mkdir -p "${LOG_ROOT}"

cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "============================================================"
echo "Stage       : Test Row C Mixed Calibration"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "Started     : $(date)"
echo "============================================================"

python basicsr/test.py \
    -opt options/cdd_experiments/test_stage2_mixed_calibration.yaml

echo "============================================================"
echo "TEST ROW C MIXED CALIBRATION COMPLETE"
echo "Finished : $(date)"
echo "============================================================"