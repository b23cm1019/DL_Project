#!/bin/bash
# Run this ONCE on cn07 to install all missing packages into both venvs.
# Usage: bash install_venv_packages.sh
#
# Run interactively:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --pty bash
#   bash /scratch/p24cs0203/krish/projects/DL_Project/install_venv_packages.sh

set -euo pipefail

module load python/3.10.pytorch
export PYTHONNOUSERSITE=1

# All pip packages needed by the project (from environment.yaml)
PACKAGES=(
    scikit-image==0.21.0
    scikit-learn==1.3.0
    opencv-python==4.8.0.74
    einops==0.6.1
    lmdb==1.4.1
    pyyaml==6.0
    tqdm==4.65.0
    tensorboard==2.13.0
    timm==0.6.13
    imageio==2.31.1
    scipy==1.11.1
    matplotlib==3.7.2
    h5py==3.9.0
    pandas==2.0.3
    seaborn==0.12.2
    huggingface-hub==0.15.1
    mrcfile==1.4.3
    tifffile==2023.4.12
    torchinfo==1.8.0
    packaging==23.1
)

install_into_venv() {
    local venv="$1"
    local cuda_tag="$2"

    if [ ! -f "$venv/bin/activate" ]; then
        echo "[SKIP] venv not found: $venv"
        return
    fi

    source "$venv/bin/activate"
    export PATH="$venv/bin:$PATH"
    echo ""
    echo "=== Installing into $venv (${cuda_tag}) ==="
    echo "[INFO] python: $(which python)"

    # Install torch + torchvision for this cuda version first
    if [ "$cuda_tag" = "cu118" ]; then
        pip install --quiet torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
            --index-url https://download.pytorch.org/whl/cu118
    else
        pip install --quiet torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121
    fi

    # Install all project packages
    pip install --quiet "${PACKAGES[@]}"

    # Install basicsr from project source if not already installed
    if ! python -c "import basicsr" 2>/dev/null; then
        echo "[INFO] Installing basicsr from source..."
        pip install --quiet -e /scratch/p24cs0203/krish/projects/DL_Project/
    fi

    echo "[INFO] Verifying key imports..."
    python -c "
import torch, torchvision, skimage, cv2, einops, lmdb, yaml, tqdm, tensorboard
print('[OK] All key packages import successfully')
print('[OK] PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('[OK] torchvision:', torchvision.__version__)
print('[OK] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"
    deactivate
}

install_into_venv "/scratch/p24cs0203/krish/envs/dl_project" "cu118"
install_into_venv "/scratch/p24cs0203/krish/envs/dl_project_cu121" "cu121"

echo ""
echo "=== All done. Both venvs are ready. ==="
