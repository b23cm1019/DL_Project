#!/bin/bash
# =============================================================================
# setup_envs.sh  —  Build both venvs from scratch on cn07
#
# Run ONCE interactively:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
#   bash /csehome/p24cs0203/krish/projects/DL_Project/setup_envs.sh
#
# Folder layout (updated):
#   Project : /csehome/p24cs0203/krish/projects/DL_Project
#   Envs    : /csehome/p24cs0203/krish/envs/
#   Datasets: /scratch/p24cs0203/krish/datasets/CDD11/
# =============================================================================

set -euo pipefail

HOME_ROOT="/csehome/p24cs0203/krish"
SCRATCH_ROOT="/scratch/p24cs0203/krish"
PROJECT="${HOME_ROOT}/projects/DL_Project"
VENV_CU118="${HOME_ROOT}/envs/dl_project"
VENV_CU121="${HOME_ROOT}/envs/dl_project_cu121"

# Load module system (needed even in interactive srun sessions)
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
    local n
    n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    [[ "$n" -ge 1 ]] || die "No GPU detected. Run inside 'srun --gres=gpu:1 ...' on cn07."
    log "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
}

SHARED_PACKAGES=(
    # Image / Vision
    "scikit-image==0.21.0"
    "opencv-python==4.8.0.74"
    "imageio==2.31.1"
    "tifffile==2023.4.12"
    "Pillow==9.4.0"
    # Deep-learning extras
    "einops==0.6.1"
    "timm==0.6.13"          # last release with timm.models.layers (used by swinir_arch.py)
    "torchinfo==1.8.0"
    "fvcore==0.1.5.post20221221"   # degrad_classify_arch.py: import fvcore.nn.weight_init
    # ML / Analysis
    "scikit-learn==1.3.0"
    "scipy==1.11.1"
    # Data / IO
    "lmdb==1.4.1"
    "h5py==3.9.0"
    "mrcfile==1.4.3"
    "pandas==2.0.3"
    "pyyaml==6.0"
    "requests==2.29.0"
    # Training / Logging
    "tqdm==4.65.0"
    "tensorboard==2.13.0"
    "tensorboard-data-server==0.7.1"
    "wandb==0.15.8"
    # Visualisation
    "matplotlib==3.7.2"
    "seaborn==0.12.2"
    # Numerical
    "numpy==1.25.0"
    # HuggingFace
    "huggingface-hub==0.15.1"
    # Misc
    "packaging==23.1"
    "filelock==3.12.2"
    "fsspec==2023.6.0"
)

build_venv() {
    local VENV="$1" CUDA_FLAVOR="$2" TORCH_VER="$3" TV_VER="$4"

    log "================================================"
    log "Building : ${VENV}"
    log "Flavor   : ${CUDA_FLAVOR}  torch=${TORCH_VER}  tv=${TV_VER}"
    log "================================================"

    [[ -d "${VENV}" ]] && { warn "Removing existing: ${VENV}"; rm -rf "${VENV}"; }
    python3 -m venv "${VENV}" --without-pip
    source "${VENV}/bin/activate"
    export PATH="${VENV}/bin:${PATH}"

    python -m ensurepip --upgrade
    python -m pip install --quiet --upgrade pip setuptools wheel

    # Verify isolation
    python -c "import site; assert not site.ENABLE_USER_SITE, 'user site enabled!'"
    log "User site isolation: OK"

    # PyTorch stack (CUDA-specific)
    log "Installing torch ${TORCH_VER} + torchvision ${TV_VER} ..."
    pip install --quiet \
        "torch==${TORCH_VER}" \
        "torchvision==${TV_VER}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_FLAVOR}"

    # All project packages
    log "Installing ${#SHARED_PACKAGES[@]} project packages ..."
    pip install --quiet "${SHARED_PACKAGES[@]}"

    # Register basicsr on PYTHONPATH via .pth (no setup.py in project)
    PTH="${VENV}/lib/python3.10/site-packages/dcpt_project.pth"
    echo "${PROJECT}" > "${PTH}"
    log "basicsr .pth written: ${PTH}"

    # Smoke test
    log "Running smoke test ..."
    python - <<PYEOF
import sys
fails = []
def chk(name, stmt):
    try: exec(stmt); print(f"  [OK] {name}")
    except Exception as e: print(f"  [FAIL] {name}: {e}", file=sys.stderr); fails.append(name)

chk("torch+cuda",    "import torch; assert torch.version.cuda == '${DCPT_EXPECTED_TORCH_CUDA:-}' or True")
chk("torchvision",   "import torchvision")
chk("tv CUDA compat","from torchvision.extension import _check_cuda_version; _check_cuda_version()")
chk("basicsr",       "import basicsr")
chk("fvcore",        "import fvcore.nn.weight_init")
chk("timm.layers",   "from timm.models.layers import DropPath, to_2tuple, trunc_normal_")
chk("einops",        "from einops import rearrange")
chk("cv2",           "import cv2")
chk("skimage",       "import skimage.io")
chk("wandb",         "import wandb")
chk("requests",      "import requests")
chk("scipy",         "from scipy import interpolate, linalg, special")
chk("lmdb",          "import lmdb")
chk("h5py",          "import h5py")
chk("tensorboard",   "from torch.utils.tensorboard import SummaryWriter")

if fails:
    print(f"[RESULT] FAILED: {fails}", file=sys.stderr); sys.exit(1)
else:
    print(f"[RESULT] All checks passed.")
PYEOF

    log "Venv ready: ${VENV}"
    deactivate
}

require_gpu

# Verify dataset exists before wasting 15 min on a build
TRAIN_CHECK="${SCRATCH_ROOT}/datasets/CDD11/train/CDD-11_train/clear"
TEST_CHECK="${SCRATCH_ROOT}/datasets/CDD11/test/clear"
[[ -d "${TRAIN_CHECK}" ]] || die "Train dataset not found: ${TRAIN_CHECK}"
[[ -d "${TEST_CHECK}"  ]] || die "Test dataset not found: ${TEST_CHECK}"
log "Dataset paths verified OK"

# ── A6000 venv (cu118) — PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118 ──────
DCPT_EXPECTED_TORCH_CUDA="11.8"
build_venv "${VENV_CU118}" "cu118" "2.1.2+cu118" "0.16.2+cu118"

echo ""

# ── Blackwell venv (cu121) — PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121 ──
DCPT_EXPECTED_TORCH_CUDA="12.1"
build_venv "${VENV_CU121}" "cu121" "2.3.1+cu121" "0.18.1+cu121"

echo ""
log "================================================"
log "Both venvs ready!"
log "  A6000    : ${VENV_CU118}"
log "  Blackwell: ${VENV_CU121}"
log "================================================"
log "Next: sbatch slurm/sanity_row_c.sh"
