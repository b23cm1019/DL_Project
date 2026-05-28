#!/bin/bash
# =============================================================================
# setup_envs.sh - Build both venvs from scratch on cn07.
#
# Run once interactively:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
#   bash /csehome/p24cs0203/krish/projects/DL_Project/setup_envs.sh
#
# Folder layout:
#   Project : /csehome/p24cs0203/krish/projects/DL_Project
#   Envs    : /csehome/p24cs0203/krish/envs/
#   Datasets: /scratch/p24cs0203/krish/datasets/CDD11/
# =============================================================================

set -euo pipefail

HOME_ROOT="/csehome/p24cs0203/krish"
SCRATCH_ROOT="/scratch/p24cs0203/krish"
PROJECT="${HOME_ROOT}/projects/DL_Project"
DATA_ROOT="${SCRATCH_ROOT}/datasets/CDD11"
VENV_CU118="${HOME_ROOT}/envs/dl_project"
VENV_CU121="${HOME_ROOT}/envs/dl_project_cu121"
REQUIREMENTS_FILE="${PROJECT}/requirements-cluster.txt"
VALIDATOR="${PROJECT}/scripts/validate_cluster_env.py"
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

log()  { echo "[INFO ] $*"; }
warn() { echo "[WARN ] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

require_gpu() {
    local count
    count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    [[ "${count}" -ge 1 ]] || die "No GPU detected. Run inside 'srun --gres=gpu:1 ...' on cn07."
    log "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
}

build_venv() {
    local venv_root="$1"
    local cuda_flavor="$2"
    local torch_ver="$3"
    local torchvision_ver="$4"
    local expected_cuda="$5"

    log "================================================"
    log "Building : ${venv_root}"
    log "Flavor   : ${cuda_flavor}  torch=${torch_ver}  tv=${torchvision_ver}"
    log "================================================"

    [[ -d "${venv_root}" ]] && { warn "Removing existing: ${venv_root}"; rm -rf "${venv_root}"; }
    python3 -m venv "${venv_root}" --without-pip
    source "${venv_root}/bin/activate"
    export PATH="${venv_root}/bin:${PATH}"

    python -m ensurepip --upgrade
    python -m pip install --quiet --upgrade pip setuptools wheel

    python -c "import site; assert not site.ENABLE_USER_SITE, 'user site enabled!'"
    log "User site isolation: OK"

    log "Installing torch ${torch_ver} + torchvision ${torchvision_ver} ..."
    pip install --quiet \
        "torch==${torch_ver}" \
        "torchvision==${torchvision_ver}" \
        --index-url "https://download.pytorch.org/whl/${cuda_flavor}"

    log "Installing project requirements from ${REQUIREMENTS_FILE} ..."
    pip install --quiet -r "${REQUIREMENTS_FILE}"

    local pth_file="${venv_root}/lib/python3.10/site-packages/dcpt_project.pth"
    echo "${PROJECT}" > "${pth_file}"
    log "basicsr .pth written: ${pth_file}"

    log "Running environment validator ..."
    python "${VALIDATOR}" \
        --project-root "${PROJECT}" \
        --venv-root "${venv_root}" \
        --data-root "${DATA_ROOT}" \
        --expected-cuda "${expected_cuda}" \
        --expected-torch "${torch_ver}" \
        --expected-torchvision "${torchvision_ver}" \
        --check-dataset

    log "Venv ready: ${venv_root}"
    deactivate
}

[[ -f "${REQUIREMENTS_FILE}" ]] || die "Missing requirements file: ${REQUIREMENTS_FILE}"
[[ -f "${VALIDATOR}" ]] || die "Missing validator script: ${VALIDATOR}"

require_gpu

TRAIN_CHECK="${DATA_ROOT}/train/CDD-11_train/clear"
TEST_CHECK="${DATA_ROOT}/test/clear"
[[ -d "${TRAIN_CHECK}" ]] || die "Train dataset not found: ${TRAIN_CHECK}"
[[ -d "${TEST_CHECK}"  ]] || die "Test dataset not found: ${TEST_CHECK}"
log "Dataset paths verified OK"

build_venv "${VENV_CU118}" "cu118" "${TORCH_CU118}" "${TORCHVISION_CU118}" "11.8"

echo ""

build_venv "${VENV_CU121}" "cu121" "${TORCH_CU121}" "${TORCHVISION_CU121}" "12.1"

echo ""
log "================================================"
log "Both venvs ready!"
log "  A6000    : ${VENV_CU118}"
log "  Blackwell: ${VENV_CU121}"
log "================================================"
log "Next: sbatch slurm/sanity_row_c.sh"
