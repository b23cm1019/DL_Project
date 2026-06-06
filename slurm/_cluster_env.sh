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

hash -r

echo "[INFO] python  = $(command -v python 2>/dev/null || echo not-found)"
echo "[INFO] python3 = $(command -v python3 2>/dev/null || echo not-found)"

export PYTHONNOUSERSITE=1
unset PYTHONHOME 2>/dev/null || true
unset PYTHONPATH 2>/dev/null || true

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# GPU_NAME="$(python3 - <<'PY'
# import torch

# if torch.cuda.is_available():
#     print(torch.cuda.get_device_name(0))
# else:
#     print("unknown")
# PY
# )"

if command -v python3 >/dev/null 2>&1; then

    GPU_NAME="$(python3 - <<'PY'
import torch

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("unknown")
PY
)"

else

    echo "[ERROR] python3 not available after module load."
    module list 2>&1 || true
    exit 1

fi

GPU_NAME="${GPU_NAME:-unknown}"
GPU_NAME_LC="${GPU_NAME,,}"

if [[ -z "${GPU_NAME}" || "${GPU_NAME}" == "unknown" ]]; then
    echo "[ERROR] Could not detect GPU through PyTorch."
    exit 1
fi

echo "[INFO] Visible GPU : ${GPU_NAME}"

export DCPT_GPU_NAME="${GPU_NAME}"

if [[ "${GPU_NAME_LC}" == *blackwell* || "${GPU_NAME_LC}" == *"pro 6000"* ]]; then

    export DCPT_GPU_CLASS="blackwell"
    export DCPT_CUDA_FLAVOR="cu128"

    export DCPT_EXPECTED_TORCH_CUDA="12.8"
    export DCPT_EXPECTED_TORCH_VERSION="2.11.0+cu128"
    export DCPT_EXPECTED_TORCHVISION_VERSION="0.26.0+cu128"

    export VENV_ROOT="${VENV_BASE}/dl_project_bw"

    echo "[INFO] GPU class     : Blackwell -> dl_project_bw"

else

    export DCPT_GPU_CLASS="a6000"
    export DCPT_CUDA_FLAVOR="cu118"

    export DCPT_EXPECTED_TORCH_CUDA="11.8"
    export DCPT_EXPECTED_TORCH_VERSION="2.1.2+cu118"
    export DCPT_EXPECTED_TORCHVISION_VERSION="0.16.2+cu118"

    export VENV_ROOT="${VENV_BASE}/dl_project"

    echo "[INFO] GPU class     : A6000 -> dl_project"

fi

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
    echo "[ERROR] venv not found: ${VENV_ROOT}"
    echo "        Run setup_envs.sh on cn07 first."
    exit 1
fi

source "${VENV_ROOT}/bin/activate"
echo "[INFO] activated python: $(which python)"
echo "[INFO] activated pip   : $(which pip)"
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
