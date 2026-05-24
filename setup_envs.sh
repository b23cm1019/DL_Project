#!/bin/bash
# =============================================================================
# setup_envs.sh  —  Clean reproducible rebuild of BOTH venvs
#
# Run ONLY from an interactive GPU shell on cn07:
#
# srun --partition=phd \
#      --nodelist=cn07 \
#      --gres=gpu:1 \
#      --cpus-per-task=8 \
#      --mem=32G \
#      --pty bash
#
# cd /scratch/p24cs0203/krish/projects/DL_Project
# bash setup_envs.sh
#
# =============================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
SCRATCH="/scratch/p24cs0203/krish"

PROJECT="${SCRATCH}/projects/DL_Project"

VENV_CU118="${SCRATCH}/envs/dl_project"

VENV_CU121="${SCRATCH}/envs/dl_project_cu121"

# ──────────────────────────────────────────────────────────────────────────────
# Base module setup
# ──────────────────────────────────────────────────────────────────────────────
module purge
module load python/3.10.pytorch

echo ""
echo "========== Loaded Modules =========="
module list
echo "===================================="
echo ""

# Prevent ~/.local contamination
export PYTHONNOUSERSITE=1

# Prevent accidental inherited Python state
unset PYTHONHOME || true
unset PYTHONPATH || true

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
log() {
    echo "[INFO ] $*"
}

warn() {
    echo "[WARN ] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_gpu() {
    local gpu_count

    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)

    if [[ "$gpu_count" -lt 1 ]]; then
        die "No GPU detected. Run inside srun --gres=gpu:1 ..."
    fi

    log "GPU(s):"
    nvidia-smi --query-gpu=name --format=csv,noheader
}

# ──────────────────────────────────────────────────────────────────────────────
# Shared packages
# ──────────────────────────────────────────────────────────────────────────────
SHARED_PACKAGES=(
    "numpy==1.25.0"

    "scikit-image==0.21.0"
    "opencv-python==4.8.0.74"
    "imageio==2.31.1"
    "tifffile==2023.4.12"
    "Pillow==9.4.0"

    "einops==0.6.1"
    "timm==0.6.13"
    "torchinfo==1.8.0"
    "fvcore==0.1.5.post20221221"

    "scikit-learn==1.3.0"
    "scipy==1.11.1"

    "lmdb==1.4.1"
    "h5py==3.9.0"
    "mrcfile==1.4.3"
    "pandas==2.0.3"
    "pyyaml==6.0"
    "requests==2.29.0"

    "tqdm==4.65.0"

    "tensorboard==2.13.0"
    "tensorboard-data-server==0.7.1"

    "wandb==0.15.8"

    "matplotlib==3.7.2"
    "seaborn==0.12.2"

    "huggingface-hub==0.15.1"

    "packaging==23.1"
    "filelock==3.12.2"
    "fsspec==2023.6.0"
)

# ──────────────────────────────────────────────────────────────────────────────
# Build function
# ──────────────────────────────────────────────────────────────────────────────
build_venv() {

    local VENV="$1"
    local CUDA_FLAVOR="$2"
    local TORCH_VER="$3"
    local TV_VER="$4"

    echo ""
    log "========================================================"
    log "Building venv: ${VENV}"
    log "CUDA flavor : ${CUDA_FLAVOR}"
    log "Torch       : ${TORCH_VER}"
    log "Torchvision : ${TV_VER}"
    log "========================================================"

    # -------------------------------------------------------------------------
    # Remove old env
    # -------------------------------------------------------------------------
    if [[ -d "${VENV}" ]]; then
        warn "Removing existing venv: ${VENV}"
        rm -rf "${VENV}"
    fi

    # -------------------------------------------------------------------------
    # Create venv
    # -------------------------------------------------------------------------
    python3 -m venv "${VENV}" --without-pip

    log "Fresh venv created."

    # -------------------------------------------------------------------------
    # Activate
    # -------------------------------------------------------------------------
    source "${VENV}/bin/activate"

    # IMPORTANT: ensure venv binaries dominate PATH
    export PATH="${VENV}/bin:${PATH}"

    echo ""
    log "Environment verification:"
    which python
    which pip || true

    python --version || die "python failed after activation"

    # -------------------------------------------------------------------------
    # Bootstrap pip
    # -------------------------------------------------------------------------
    python -m ensurepip --upgrade

    python -m pip install --upgrade pip setuptools wheel

    echo ""
    log "Post-bootstrap verification:"
    which python
    which pip

    python --version
    pip --version

    # -------------------------------------------------------------------------
    # Verify no user-site leakage
    # -------------------------------------------------------------------------
    python - <<PYEOF
import site
assert not site.ENABLE_USER_SITE, "User site leakage detected"
print("[OK] PYTHONNOUSERSITE enforced")
PYEOF

    # -------------------------------------------------------------------------
    # Install torch stack
    # -------------------------------------------------------------------------
    log "Installing torch stack..."

    pip install \
        "torch==${TORCH_VER}" \
        "torchvision==${TV_VER}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_FLAVOR}"

    # -------------------------------------------------------------------------
    # Install project packages
    # -------------------------------------------------------------------------
    log "Installing shared packages..."

    pip install "${SHARED_PACKAGES[@]}"

    # -------------------------------------------------------------------------
    # Install project editable
    # -------------------------------------------------------------------------
    log "Adding project root via .pth ..."

    echo "${PROJECT}" > \
    "${VENV}/lib/python3.10/site-packages/dl_project.pth"

    # -------------------------------------------------------------------------
    # Smoke tests
    # -------------------------------------------------------------------------
    log "Running smoke tests..."

    python - <<PYEOF

import sys

failures = []

def check(name, stmt):
    try:
        exec(stmt)
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")
        failures.append(name)

print("\n========== TORCH ==========")

check("torch",
      "import torch; print(torch.__version__, torch.version.cuda)")

check("torchvision",
      "import torchvision; print(torchvision.__version__)")

check("cuda",
      "import torch; print(torch.cuda.get_device_name(0))")

print("\n========== CORE ==========")

check("numpy", "import numpy")
check("opencv", "import cv2")
check("scikit-image", "import skimage.io")
check("imageio", "import imageio")
check("PIL", "from PIL import Image")

print("\n========== DL ==========")

check("einops", "from einops import rearrange")
check("timm", "from timm.models.layers import DropPath")
check("fvcore", "import fvcore.nn.weight_init")
check("torchinfo", "from torchinfo import summary")

print("\n========== SCIENCE ==========")

check("scipy", "import scipy")
check("sklearn", "import sklearn")
check("pandas", "import pandas")

print("\n========== LOGGING ==========")

check("tensorboard",
      "from torch.utils.tensorboard import SummaryWriter")

check("wandb", "import wandb")

print("\n========== PROJECT ==========")

check("basicsr", "import basicsr")

check(
    "degrad_classify_arch",
    "from basicsr.archs.degrad_classify_arch import DegradClassifyNet"
)

print()

if failures:
    print(f"[RESULT] FAILURES: {failures}")
    sys.exit(1)

print("[RESULT] ALL IMPORTS OK")

PYEOF

    log "SUCCESS: ${VENV}"

    deactivate
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
require_gpu

echo ""
log "Project root : ${PROJECT}"
log "Scratch root : ${SCRATCH}"
echo ""

# -------------------------------------------------------------------------
# CUDA 11.8 / A6000
# -------------------------------------------------------------------------
build_venv \
    "${VENV_CU118}" \
    "cu118" \
    "2.1.2+cu118" \
    "0.16.2+cu118"

# -------------------------------------------------------------------------
# CUDA 12.1 / Blackwell
# -------------------------------------------------------------------------
build_venv \
    "${VENV_CU121}" \
    "cu121" \
    "2.3.1+cu121" \
    "0.18.1+cu121"

echo ""
log "========================================================"
log "BOTH ENVIRONMENTS SUCCESSFULLY BUILT"
log "========================================================"

log "A6000 env    : ${VENV_CU118}"
log "Blackwell env: ${VENV_CU121}"

echo ""
log "Next:"
log "sbatch slurm/sanity_row_c.slurm"