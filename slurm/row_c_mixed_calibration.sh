#!/bin/bash
#SBATCH --job-name=rowC_mixed_calib
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/row_c_mixed_calibration_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/row_c_mixed_calibration_%j.err
#SBATCH --time=2-00:00:00

set -euo pipefail

PROJECT_ROOT="/csehome/p24cs0203/krish/projects/DL_Project"
LOG_ROOT="/csehome/p24cs0203/krish/logs/slurm"

mkdir -p "${LOG_ROOT}"

cd "${PROJECT_ROOT}"

source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "============================================================"
echo "Stage       : Row C Mixed Calibration"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node        : $(hostname)"
echo "GPU ID      : ${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU Name    : ${DCPT_GPU_NAME:-unknown}"
echo "GPU Class   : ${DCPT_GPU_CLASS:-unknown}"
echo "MASTER_PORT : ${MASTER_PORT}"
echo "Started     : $(date)"
echo "============================================================"

bash "${PROJECT_ROOT}/run_experiments.sh" row_c_mixed_calibration

echo "============================================================"
echo "ROW C MIXED CALIBRATION COMPLETE"
echo "Finished : $(date)"
echo "============================================================"
