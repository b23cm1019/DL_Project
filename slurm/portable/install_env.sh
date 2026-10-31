#!/bin/bash
# Create/populate both venvs (cu118 for A6000/A30, cu128 for Blackwell).

set -euo pipefail

# Portable copy of install_venv_packages.sh for another account. Run from the repo root:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --pty bash
#   bash slurm/portable/install_env.sh
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
SHARED_ROOT="${SHARED_ROOT:-/scratch/p26cs0003/krish}"
SCRATCH_ROOT="${OUT_ROOT:-/scratch/${USER}/sevdcpt}"
VENV_BASE="${VENV_BASE:-${HOME}/envs}"
DATA_ROOT="${SHARED_ROOT}/datasets/CDD11"
[[ -f "${PROJECT_ROOT}/requirements-cluster.txt" ]] || { echo "Run from the repo root (requirements-cluster.txt not found)" >&2; exit 2; }
REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements-cluster.txt"
VALIDATOR="${PROJECT_ROOT}/scripts/validate_cluster_env.py"
PIP_CACHE_DIR="${SCRATCH_ROOT}/pip-cache"
WHEELHOUSE_ROOT="${SCRATCH_ROOT}/wheelhouse/dl_project_py310"
SHARED_WHEELHOUSE="${WHEELHOUSE_ROOT}/shared"
CU118_WHEELHOUSE="${WHEELHOUSE_ROOT}/cu118"
CU128_WHEELHOUSE="${WHEELHOUSE_ROOT}/cu128"
VENV_CU118="${VENV_BASE}/dl_project"
VENV_CU128="${VENV_BASE}/dl_project_bw"
TORCH_CU118="2.1.2+cu118"
TORCHVISION_CU118="0.16.2+cu118"
TORCH_CU128="2.11.0+cu128"
TORCHVISION_CU128="0.26.0+cu128"

if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load python/3.10.pytorch 2>/dev/null || true

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR

prepare_shared_wheelhouse() {
    mkdir -p "${PIP_CACHE_DIR}" "${SHARED_WHEELHOUSE}"
    python3 -m pip download \
        --disable-pip-version-check \
        --dest "${SHARED_WHEELHOUSE}" \
        -r "${REQUIREMENTS_FILE}"
}

prepare_torch_wheelhouse() {
    local cuda_flavor="$1"
    local torch_ver="$2"
    local torchvision_ver="$3"
    local torch_wheelhouse="$4"

    mkdir -p "${torch_wheelhouse}"
    python3 -m pip download \
        --disable-pip-version-check \
        --dest "${torch_wheelhouse}" \
        --index-url "https://download.pytorch.org/whl/${cuda_flavor}" \
        "torch==${torch_ver}" \
        "torchvision==${torchvision_ver}"
}

install_into_venv() {
    local venv_root="$1"
    local cuda_flavor="$2"
    local torch_ver="$3"
    local torchvision_ver="$4"
    local expected_cuda="$5"
    local torch_wheelhouse="$6"

    if [[ ! -f "${venv_root}/bin/activate" ]]; then
        echo "[WARN] venv not found, creating: ${venv_root}"
        python3 -m venv "${venv_root}" --without-pip
    fi

    source "${venv_root}/bin/activate"
    export PATH="${venv_root}/bin:${PATH}"

    echo ""
    echo "=== Repairing ${venv_root} (${cuda_flavor}) ==="
    echo "[INFO] python: $(which python)"

    python -m ensurepip --upgrade
    python -m pip install --quiet --upgrade pip setuptools wheel

    pip install --quiet \
        --no-index \
        --find-links "${torch_wheelhouse}" \
        --find-links "${SHARED_WHEELHOUSE}" \
        "torch==${torch_ver}" \
        "torchvision==${torchvision_ver}" \
        --prefer-binary

    pip install --quiet \
        --no-index \
        --find-links "${SHARED_WHEELHOUSE}" \
        -r "${REQUIREMENTS_FILE}"

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

prepare_shared_wheelhouse
prepare_torch_wheelhouse "cu118" "${TORCH_CU118}" "${TORCHVISION_CU118}" "${CU118_WHEELHOUSE}"
prepare_torch_wheelhouse "cu128" "${TORCH_CU128}" "${TORCHVISION_CU128}" "${CU128_WHEELHOUSE}"

install_into_venv "${VENV_CU118}" "cu118" "${TORCH_CU118}" "${TORCHVISION_CU118}" "11.8" "${CU118_WHEELHOUSE}"
install_into_venv "${VENV_CU128}" "cu128" "${TORCH_CU128}" "${TORCHVISION_CU128}" "12.8" "${CU128_WHEELHOUSE}"

echo ""
echo "=== Both venvs repaired and validated. ==="
