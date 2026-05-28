#!/bin/bash
# Repair or populate both existing cn07 venvs without rebuilding them.
#
# Run interactively:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --pty bash
#   bash /csehome/p24cs0203/krish/projects/DL_Project/install_venv_packages.sh

set -euo pipefail

HOME_ROOT="/csehome/p24cs0203/krish"
SCRATCH_ROOT="/scratch/p24cs0203/krish"
PROJECT_ROOT="${HOME_ROOT}/projects/DL_Project"
DATA_ROOT="${SCRATCH_ROOT}/datasets/CDD11"
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements-cluster.txt"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_cluster_env.py"
VENV_CU118="${HOME_ROOT}/envs/dl_project"
VENV_CU121="${HOME_ROOT}/envs/dl_project_cu121"
TORCH_CU118="2.1.2+cu118"
TORCHVISION_CU118="0.16.2+cu118"
TORCH_CU121="2.3.1+cu121"
TORCHVISION_CU121="0.18.1+cu121"

if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load python/3.10.pytorch 2>/dev/null || true

export PYTHONNOUSERSITE=1

install_into_venv() {
    local venv_root="$1"
    local cuda_flavor="$2"
    local torch_ver="$3"
    local torchvision_ver="$4"
    local expected_cuda="$5"

    if [[ ! -f "${venv_root}/bin/activate" ]]; then
        echo "[SKIP] venv not found: ${venv_root}"
        return
    fi

    source "${venv_root}/bin/activate"
    export PATH="${venv_root}/bin:${PATH}"

    echo ""
    echo "=== Repairing ${venv_root} (${cuda_flavor}) ==="
    echo "[INFO] python: $(which python)"

    pip install --quiet \
        "torch==${torch_ver}" \
        "torchvision==${torchvision_ver}" \
        --index-url "https://download.pytorch.org/whl/${cuda_flavor}"

    pip install --quiet -r "${REQUIREMENTS_FILE}"

    echo "${PROJECT_ROOT}" > "${venv_root}/lib/python3.10/site-packages/dcpt_project.pth"

    python "${VALIDATOR}" \
        --project-root "${PROJECT_ROOT}" \
        --venv-root "${venv_root}" \
        --data-root "${DATA_ROOT}" \
        --expected-cuda "${expected_cuda}" \
        --expected-torch "${torch_ver}" \
        --expected-torchvision "${torchvision_ver}" \
        --check-dataset

    deactivate
}

install_into_venv "${VENV_CU118}" "cu118" "${TORCH_CU118}" "${TORCHVISION_CU118}" "11.8"
install_into_venv "${VENV_CU121}" "cu121" "${TORCH_CU121}" "${TORCHVISION_CU121}" "12.1"

echo ""
echo "=== Both venvs repaired and validated. ==="
