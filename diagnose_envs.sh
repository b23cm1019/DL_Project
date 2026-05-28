#!/bin/bash
# Run this interactively on cn07 to inspect both venvs.
#
# Usage:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --pty bash
#   bash /csehome/p24cs0203/krish/projects/DL_Project/diagnose_envs.sh

set -euo pipefail

HOME_ROOT="/csehome/p24cs0203/krish"
SCRATCH_ROOT="/scratch/p24cs0203/krish"
PROJECT_ROOT="${HOME_ROOT}/projects/DL_Project"
DATA_ROOT="${SCRATCH_ROOT}/datasets/CDD11"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_cluster_env.py"
VENV_CU118="${HOME_ROOT}/envs/dl_project"
VENV_CU121="${HOME_ROOT}/envs/dl_project_cu121"

if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load python/3.10.pytorch 2>/dev/null || true

export PYTHONNOUSERSITE=1

echo "============================================================"
echo " GPU on this node"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo ""

check_venv() {
    local venv_root="$1"
    local expected_cuda="$2"
    local expected_torch="$3"
    local expected_torchvision="$4"
    local label="$5"

    echo "============================================================"
    echo " Checking: ${label} -> ${venv_root}"
    echo "============================================================"

    if [[ ! -f "${venv_root}/bin/activate" ]]; then
        echo "[MISSING] venv does not exist: ${venv_root}"
        return
    fi

    source "${venv_root}/bin/activate"
    export PATH="${venv_root}/bin:${PATH}"

    echo "python  : $(which python)"
    echo "pip     : $(which pip)"
    echo ""

    echo "--- pyvenv.cfg ---"
    cat "${venv_root}/pyvenv.cfg"
    echo ""

    python "${VALIDATOR}" \
        --project-root "${PROJECT_ROOT}" \
        --venv-root "${venv_root}" \
        --data-root "${DATA_ROOT}" \
        --expected-cuda "${expected_cuda}" \
        --expected-torch "${expected_torch}" \
        --expected-torchvision "${expected_torchvision}" \
        --check-dataset

    deactivate
    echo ""
}

check_venv "${VENV_CU118}" "11.8" "2.1.2+cu118" "0.16.2+cu118" "A6000 venv (cu118)"
check_venv "${VENV_CU121}" "12.1" "2.3.1+cu121" "0.18.1+cu121" "Blackwell venv (cu121)"

echo "============================================================"
echo " System torchrun location"
echo "============================================================"
which torchrun 2>/dev/null || echo "(not found in PATH)"
module load python/3.10.pytorch 2>/dev/null || true
which torchrun 2>/dev/null || echo "(not found after module load)"
