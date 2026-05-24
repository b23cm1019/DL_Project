#!/bin/bash
# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
module purge
module load python/3.10.pytorch

# Prevent contamination from ~/.local
export PYTHONNOUSERSITE=1

# Clean inherited Python state
unset PYTHONHOME || true
unset PYTHONPATH || true

# -----------------------------------------------------------------------------
# GPU detection
# -----------------------------------------------------------------------------
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

if [[ -z "${GPU_NAME}" ]]; then
    echo "[ERROR] Could not detect GPU"
    exit 1
fi

# -----------------------------------------------------------------------------
# Environment selection
# -----------------------------------------------------------------------------
if [[ "${GPU_NAME}" == *"Blackwell"* ]]; then

    export VENV_ROOT="/scratch/p24cs0203/krish/envs/dl_project_cu121"

else

    export VENV_ROOT="/scratch/p24cs0203/krish/envs/dl_project"

fi

# -----------------------------------------------------------------------------
# Validate env existence
# -----------------------------------------------------------------------------
if [[ ! -d "${VENV_ROOT}" ]]; then
    echo "[ERROR] Missing venv: ${VENV_ROOT}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Activate env
# -----------------------------------------------------------------------------
source "${VENV_ROOT}/bin/activate"

# Ensure venv binaries dominate PATH
export PATH="${VENV_ROOT}/bin:${PATH}"

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
echo "[INFO] Activated venv : ${VENV_ROOT}"
echo "[INFO] GPU            : ${GPU_NAME}"

echo "[INFO] python         : $(which python)"
echo "[INFO] pip            : $(which pip)"

python --version

# -----------------------------------------------------------------------------
# Torch validation
# -----------------------------------------------------------------------------
python - <<'PYEOF'
import torch
import torchvision

print(f"[INFO] PyTorch     : {torch.__version__}  CUDA: {torch.version.cuda}")
print(f"[INFO] torchvision : {torchvision.__version__}")
print(f"[INFO] GPU         : {torch.cuda.get_device_name(0)}")
PYEOF