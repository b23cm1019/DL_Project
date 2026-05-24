#!/bin/bash
# =============================================================================
# setup_envs.sh  —  Create BOTH venvs from scratch on cn07 (interactive shell)
#
# Run this ONCE interactively on cn07:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --cpus-per-task=8 --mem=32G --pty bash
#   bash /scratch/p24cs0203/krish/projects/DL_Project/setup_envs.sh
#
# What it does:
#   1. Deletes and recreates both venvs cleanly (no leftover contamination)
#   2. Installs the correct pinned torch + torchvision per CUDA flavour
#   3. Installs ALL packages actually imported by the codebase
#   4. Installs basicsr from source in editable mode
#   5. Runs a full import smoke-test for each venv
#
# After this script succeeds, NEVER pip-install into these venvs manually.
# Use this script to rebuild if you need to change a dependency.
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH="/scratch/p24cs0203/krish"
PROJECT="${SCRATCH}/projects/DL_Project"
VENV_CU118="${SCRATCH}/envs/dl_project"
VENV_CU121="${SCRATCH}/envs/dl_project_cu121"

# ── Base module ───────────────────────────────────────────────────────────────
# The system module provides libpython3.10.so.1.0 which venvs link against.
# Always load it BEFORE creating/activating any venv.
module purge
module load python/3.10.pytorch

# Block ~/.local from polluting venvs — this is the #1 cause of version mismatches
export PYTHONNOUSERSITE=1

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo "[INFO ] $*"; }
warn() { echo "[WARN ] $*"; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

require_gpu() {
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
    if [[ "$gpu_count" -lt 1 ]]; then
        die "No GPU detected. Run this inside 'srun --gres=gpu:1 ...' on cn07."
    fi
    log "GPU(s) present: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"
}

# =============================================================================
# ALL packages used by the codebase — audited from every import in basicsr/,
# knn.py, knn_gen.py, t_sne.py, and the training/test scripts.
#
# Split into sections so the purpose of each package is clear.
# ALL versions are pinned for reproducibility.
# =============================================================================
SHARED_PACKAGES=(
    # ── Image / Vision ────────────────────────────────────────────────────────
    # scikit-image: used as `import skimage.io as skio` in utils/img_util.py
    "scikit-image==0.21.0"
    # opencv: used as `import cv2` throughout data/ and utils/
    "opencv-python==4.8.0.74"
    # imageio: used in metrics/ and data/ for reading/writing images
    "imageio==2.31.1"
    # tifffile: backend used by imageio and directly in some data loaders
    "tifffile==2023.4.12"
    # Pillow: used by torchvision internally + some data transforms
    "Pillow==9.4.0"

    # ── Deep-Learning Extras ──────────────────────────────────────────────────
    # einops: `from einops import rearrange` in restormer_arch.py, promptir_arch.py
    "einops==0.6.1"
    # timm: `from timm.models.layers import DropPath, to_2tuple, trunc_normal_`
    #       `from timm.utils.metrics import accuracy`
    #       Used in swinir_arch.py and classification models.
    #       0.6.13 is the last release where timm.models.layers exists (not timm.layers).
    "timm==0.6.13"
    # torchinfo: `from torchinfo import summary` in classification models
    "torchinfo==1.8.0"
    # fvcore: `import fvcore.nn.weight_init as weight_init` in degrad_classify_arch.py
    #         Facebook Research's core library — NOT in your original install script!
    "fvcore==0.1.5.post20221221"

    # ── Machine Learning / Analysis ───────────────────────────────────────────
    # scikit-learn: `from sklearn.manifold import TSNE`, `KNeighborsClassifier`,
    #               `classification_report`, `confusion_matrix` — used in knn.py, t_sne.py
    "scikit-learn==1.3.0"
    # scipy: `from scipy import interpolate, linalg, special`, `scipy.ndimage`,
    #        `scipy.special.gamma`, `scipy.stats.multivariate_normal`
    #        Used extensively in degradations.py and metrics/
    "scipy==1.11.1"

    # ── Data / IO ─────────────────────────────────────────────────────────────
    # lmdb: `import lmdb` — used for fast image dataset storage in data/
    "lmdb==1.4.1"
    # h5py: `import h5py` — used in data/ for HDF5 dataset loading
    "h5py==3.9.0"
    # mrcfile: used in some data utilities for electron microscopy format support
    "mrcfile==1.4.3"
    # pandas: `import pandas as pd` — used in knn.py, t_sne.py, analysis scripts
    "pandas==2.0.3"
    # pyyaml: `import yaml` — config files throughout
    "pyyaml==6.0"
    # requests: `import requests` — used in utils/download_util.py for model downloads
    "requests==2.29.0"

    # ── Training / Logging ────────────────────────────────────────────────────
    # tqdm: `from tqdm import tqdm, trange` — progress bars everywhere
    "tqdm==4.65.0"
    # tensorboard: `from torch.utils.tensorboard import SummaryWriter`
    #              `from tensorboard.backend.event_processing.event_accumulator import EventAccumulator`
    "tensorboard==2.13.0"
    "tensorboard-data-server==0.7.1"
    # wandb: `import wandb` in utils/logger.py (init_wandb_logger, lazy import)
    #        Needed even if you don't use it — the import must not crash at startup
    "wandb==0.15.8"

    # ── Visualisation ─────────────────────────────────────────────────────────
    # matplotlib: `import matplotlib`, `import matplotlib.pyplot as plt`
    #             Used in knn.py, t_sne.py, and metrics/
    "matplotlib==3.7.2"
    # seaborn: `import seaborn as sns` — used in knn.py, t_sne.py for plots
    "seaborn==0.12.2"

    # ── Numerical ─────────────────────────────────────────────────────────────
    # numpy: used everywhere; pin to match the torch wheel's expectation
    "numpy==1.25.0"

    # ── HuggingFace (optional model hub) ─────────────────────────────────────
    # huggingface-hub: used in utils/download_util.py for hf_hub_download
    "huggingface-hub==0.15.1"

    # ── Misc ──────────────────────────────────────────────────────────────────
    "packaging==23.1"
    "filelock==3.12.2"
    "fsspec==2023.6.0"
)

build_venv() {
    local VENV="$1"
    local CUDA_FLAVOR="$2"   # cu118 or cu121
    local TORCH_VER="$3"     # e.g. 2.1.2+cu118
    local TV_VER="$4"        # e.g. 0.16.2+cu118

    log "========================================================"
    log "Building venv : ${VENV}"
    log "CUDA flavour  : ${CUDA_FLAVOR}"
    log "PyTorch       : ${TORCH_VER}"
    log "torchvision   : ${TV_VER}"
    log "========================================================"

    # ── 1. Wipe and recreate the venv cleanly ─────────────────────────────────
    if [[ -d "${VENV}" ]]; then
        warn "Removing existing venv: ${VENV}"
        rm -rf "${VENV}"
    fi
    python3 -m venv "${VENV}" --without-pip
    log "Created fresh venv: ${VENV}"

    # ── 2. Activate and bootstrap pip ────────────────────────────────────────
    source "${VENV}/bin/activate"
    export PATH="${VENV}/bin:${PATH}"

    python -m ensurepip --upgrade
    python -m pip install --quiet --upgrade pip setuptools wheel

    log "python : $(which python)"
    log "pip    : $(which pip)"

    # Confirm PYTHONNOUSERSITE is respected
    python -c "import site; assert not site.ENABLE_USER_SITE, 'user site still enabled — pip install --user packages will leak in!'"
    log "User site isolation: OK"

    # ── 3. Install PyTorch stack (CUDA-version-specific) ──────────────────────
    log "Installing PyTorch ${TORCH_VER} + torchvision ${TV_VER} ..."
    pip install --quiet \
        "torch==${TORCH_VER}" \
        "torchvision==${TV_VER}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_FLAVOR}"

    # ── 4. Install all shared/project packages ────────────────────────────────
    log "Installing ${#SHARED_PACKAGES[@]} project packages ..."
    pip install --quiet "${SHARED_PACKAGES[@]}"

    # ── 5. Install basicsr from project source (editable) ────────────────────
    log "Installing basicsr from source (editable) ..."
    pip install --quiet -e "${PROJECT}/"

    # ── 6. Full smoke test — every import the codebase actually uses ──────────
    log "Running full import smoke test ..."
    python - <<PYEOF
import sys

failures = []

def check(pkg, import_stmt):
    try:
        exec(import_stmt)
        print(f"  [OK] {pkg}")
    except Exception as e:
        print(f"  [FAIL] {pkg}: {e}", file=sys.stderr)
        failures.append(pkg)

print("\n--- PyTorch stack ---")
check("torch",       "import torch; print(f'       torch={torch.__version__} CUDA={torch.version.cuda}')")
check("torchvision", "import torchvision; print(f'       torchvision={torchvision.__version__}')")
check("torchvision CUDA compat",
      "from torchvision.extension import _check_cuda_version; _check_cuda_version()")
check("GPU",         "import torch; print(f'       GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NONE\"}')")

print("\n--- Image / Vision ---")
check("scikit-image", "import skimage.io")
check("opencv",       "import cv2")
check("imageio",      "import imageio")
check("tifffile",     "import tifffile")
check("Pillow",       "from PIL import Image")

print("\n--- Deep-learning extras ---")
check("einops",    "from einops import rearrange")
check("timm.models.layers",  "from timm.models.layers import DropPath, to_2tuple, trunc_normal_")
check("timm.utils.metrics",  "from timm.utils.metrics import accuracy")
check("torchinfo", "from torchinfo import summary")
check("fvcore",    "import fvcore.nn.weight_init as weight_init")

print("\n--- ML / Analysis ---")
check("scikit-learn KNN",      "from sklearn.neighbors import KNeighborsClassifier")
check("scikit-learn TSNE",     "from sklearn.manifold import TSNE")
check("scikit-learn metrics",  "from sklearn.metrics import classification_report, confusion_matrix")
check("scikit-learn boundary", "from sklearn.inspection import DecisionBoundaryDisplay")
check("scipy.interpolate",     "from scipy import interpolate")
check("scipy.linalg",          "from scipy import linalg")
check("scipy.special",         "from scipy import special")
check("scipy.ndimage",         "from scipy.ndimage import convolve")
check("scipy.stats",           "from scipy.stats import multivariate_normal")

print("\n--- Data / IO ---")
check("lmdb",           "import lmdb")
check("h5py",           "import h5py")
check("mrcfile",        "import mrcfile")
check("pandas",         "import pandas as pd")
check("pyyaml",         "import yaml")
check("requests",       "import requests")

print("\n--- Training / Logging ---")
check("tqdm",                    "from tqdm import tqdm, trange")
check("tensorboard SummaryWriter","from torch.utils.tensorboard import SummaryWriter")
check("tensorboard EventAccum",  "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator")
check("wandb",                   "import wandb")

print("\n--- Visualisation ---")
check("matplotlib", "import matplotlib.pyplot as plt")
check("seaborn",    "import seaborn as sns")

print("\n--- Project ---")
check("basicsr", "import basicsr")
check("basicsr.archs.degrad_classify_arch",
      "from basicsr.archs.degrad_classify_arch import DegradClassifyNet")

print()
if failures:
    print(f"[RESULT] {len(failures)} package(s) FAILED: {failures}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"[RESULT] All {28 - len(failures)} imports OK — venv is ready.")
PYEOF

    log "Venv ${VENV} is ready."
    deactivate
}

# ── Main ──────────────────────────────────────────────────────────────────────
require_gpu

log "Project root : ${PROJECT}"
log "Scratch root : ${SCRATCH}"
echo ""

# ── A6000 venv  (CUDA 11.8 — PyTorch 2.1.2+cu118) ────────────────────────────
# PyTorch 2.1.x is the last release shipping cu118 wheels.
# torchvision 0.16.x matches torch 2.1.x exactly.
build_venv "${VENV_CU118}" "cu118" "2.1.2+cu118" "0.16.2+cu118"

echo ""

# ── Blackwell venv  (CUDA 12.1 — PyTorch 2.3.1+cu121) ────────────────────────
# PyTorch 2.3.1 includes Blackwell SM_89/SM_90 support.
# torchvision 0.18.1 matches torch 2.3.1 exactly.
build_venv "${VENV_CU121}" "cu121" "2.3.1+cu121" "0.18.1+cu121"

echo ""
log "========================================================"
log "Both venvs built successfully!"
log "  A6000    (cu118) : ${VENV_CU118}"
log "  Blackwell (cu121) : ${VENV_CU121}"
log "========================================================"
log "Next step: sbatch slurm/sanity_row_c.slurm to verify end-to-end."
