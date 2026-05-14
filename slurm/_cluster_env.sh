#!/bin/bash

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/b23cm1019}"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${SCRATCH_ROOT}/projects/DL_Project}}"

module purge
module load python/3.10.pytorch

export DCPT_DATA_ROOT="${DCPT_DATA_ROOT:-${SCRATCH_ROOT}/datasets/CDD11}"
export BASICSR_EXPERIMENTS_ROOT="${BASICSR_EXPERIMENTS_ROOT:-${SCRATCH_ROOT}/outputs}"
export BASICSR_MODELS_ROOT="${BASICSR_MODELS_ROOT:-${SCRATCH_ROOT}/checkpoints}"
export BASICSR_TRAINING_STATES_ROOT="${BASICSR_TRAINING_STATES_ROOT:-${SCRATCH_ROOT}/training_states}"
export BASICSR_LOG_ROOT="${BASICSR_LOG_ROOT:-${SCRATCH_ROOT}/logs}"
export BASICSR_VIS_ROOT="${BASICSR_VIS_ROOT:-${SCRATCH_ROOT}/outputs/visualizations}"
export BASICSR_RESULTS_ROOT="${BASICSR_RESULTS_ROOT:-${SCRATCH_ROOT}/outputs/test_results}"
export BASICSR_TB_ROOT="${BASICSR_TB_ROOT:-${SCRATCH_ROOT}/logs/tb_logger}"
export MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "${SCRATCH_ROOT}/logs/slurm"
mkdir -p "${BASICSR_EXPERIMENTS_ROOT}"
mkdir -p "${BASICSR_MODELS_ROOT}"
mkdir -p "${BASICSR_TRAINING_STATES_ROOT}"
mkdir -p "${BASICSR_LOG_ROOT}"
mkdir -p "${BASICSR_VIS_ROOT}"
mkdir -p "${BASICSR_RESULTS_ROOT}"
mkdir -p "${BASICSR_TB_ROOT}"

cd "${PROJECT_ROOT}"

python3 --version
hostname
nvidia-smi
