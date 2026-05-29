#!/bin/bash
# Source this at the top of every SLURM or interactive cn07 job.

set -euo pipefail

export HOME_ROOT="/csehome/p24cs0203/krish"
export SCRATCH_ROOT="/scratch/p24cs0203/krish"

export PROJECT_ROOT="${HOME_ROOT}/projects/DL_Project"
export VENV_BASE="${HOME_ROOT}/envs"
export LOG_ROOT="${HOME_ROOT}/logs"
export DCPT_DATA_ROOT="${SCRATCH_ROOT}/datasets/CDD11"

export BASICSR_EXPERIMENTS_ROOT="${HOME_ROOT}/outputs"
export BASICSR_MODELS_ROOT="${HOME_ROOT}/checkpoints"
export BASICSR_TRAINING_STATES_ROOT="${HOME_ROOT}/training_states"

VALIDATOR="${PROJECT_ROOT}/scripts/validate_cluster_env.py"

if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /etc/profile.d/lmod.sh ]]; then
    source /etc/profile.d/lmod.sh
else
    echo "[WARN] No module init script found; proceeding without module system"
fi

module purge 2>/dev/null || true
module load python/3.10.pytorch 2>/dev/null || \
    echo "[WARN] Could not load python/3.10.pytorch; proceeding"

export PYTHONNOUSERSITE=1
unset PYTHONHOME 2>/dev/null || true
unset PYTHONPATH 2>/dev/null || true

GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"

GPU_NAME="$(nvidia-smi -i "${GPU_ID}" \
    --query-gpu=name \
    --format=csv,noheader \
    2>/dev/null | head -n1 | tr -d '\r')"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

GPU_NAME="${GPU_NAME:-unknown}"
GPU_NAME_LC="${GPU_NAME,,}"

if [[ -z "${GPU_NAME}" || "${GPU_NAME}" == "unknown" ]]; then
    echo "[ERROR] Could not detect GPU via nvidia-smi. Is a GPU allocated?"
    exit 1
fi

echo "[INFO] Assigned GPU : ${GPU_NAME}"
export DCPT_GPU_NAME="${GPU_NAME}"

if [[ "${GPU_NAME_LC}" == *blackwell* || "${GPU_NAME_LC}" == *"pro 6000"* ]]; then
    export DCPT_GPU_CLASS="blackwell"
    export DCPT_CUDA_FLAVOR="cu121"
    export DCPT_EXPECTED_TORCH_CUDA="12.1"
    export DCPT_EXPECTED_TORCH_VERSION="2.3.1+cu121"
    export DCPT_EXPECTED_TORCHVISION_VERSION="0.18.1+cu121"
    export VENV_ROOT="${VENV_BASE}/dl_project_cu121"
    echo "[INFO] GPU class     : Blackwell -> cu121 venv"
else
    export DCPT_GPU_CLASS="a6000"
    export DCPT_CUDA_FLAVOR="cu118"
    export DCPT_EXPECTED_TORCH_CUDA="11.8"
    export DCPT_EXPECTED_TORCH_VERSION="2.1.2+cu118"
    export DCPT_EXPECTED_TORCHVISION_VERSION="0.16.2+cu118"
    export VENV_ROOT="${VENV_BASE}/dl_project"
    echo "[INFO] GPU class     : A6000 -> cu118 venv"
fi

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
    echo "[ERROR] venv not found: ${VENV_ROOT}"
    echo "        Run setup_envs.sh on cn07 first."
    exit 1
fi

source "${VENV_ROOT}/bin/activate"
export PATH="${VENV_ROOT}/bin:${PATH}"

PTH_FILE="${VENV_ROOT}/lib/python3.10/site-packages/dcpt_project.pth"
if [[ ! -f "${PTH_FILE}" ]]; then
    echo "[INFO] Writing basicsr .pth file -> ${PTH_FILE}"
    echo "${PROJECT_ROOT}" > "${PTH_FILE}"
fi

python "${VALIDATOR}" \
    --project-root "${PROJECT_ROOT}" \
    --venv-root "${VENV_ROOT}" \
    --data-root "${DCPT_DATA_ROOT}" \
    --expected-cuda "${DCPT_EXPECTED_TORCH_CUDA}" \
    --expected-torch "${DCPT_EXPECTED_TORCH_VERSION}" \
    --expected-torchvision "${DCPT_EXPECTED_TORCHVISION_VERSION}" \
    --check-dataset

mkdir -p \
    "${LOG_ROOT}/slurm" \
    "${BASICSR_EXPERIMENTS_ROOT}" \
    "${BASICSR_MODELS_ROOT}" \
    "${BASICSR_TRAINING_STATES_ROOT}" \
    "${PROJECT_ROOT}/experiments"

export MASTER_PORT="${MASTER_PORT:-29500}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "[INFO] Cluster env ready"
echo "[INFO] GPU class      : ${DCPT_GPU_CLASS} (${DCPT_CUDA_FLAVOR})"
echo "[INFO] VENV           : ${VENV_ROOT}"
echo "[INFO] PROJECT_ROOT   : ${PROJECT_ROOT}"
echo "[INFO] DCPT_DATA_ROOT : ${DCPT_DATA_ROOT}"
echo "[INFO] python         : $(which python)"
echo "[INFO] torchrun       : $(which torchrun)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
