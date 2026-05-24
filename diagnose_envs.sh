#!/bin/bash
# =============================================================================
# diagnose_envs.sh  —  Run this interactively on cn07 to see what's broken.
#
# Usage:
#   srun --partition=phd --nodelist=cn07 --gres=gpu:1 --pty bash
#   bash /scratch/p24cs0203/krish/projects/DL_Project/diagnose_envs.sh
# =============================================================================

set -euo pipefail

SCRATCH="/scratch/p24cs0203/krish"
VENV_CU118="${SCRATCH}/envs/dl_project"
VENV_CU121="${SCRATCH}/envs/dl_project_cu121"

module purge
module load python/3.10.pytorch
export PYTHONNOUSERSITE=1

echo "============================================================"
echo " GPU on this node"
echo "============================================================"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo ""

check_venv() {
    local VENV="$1"
    local EXPECTED_CUDA="$2"
    local LABEL="$3"

    echo "============================================================"
    echo " Checking: ${LABEL}  →  ${VENV}"
    echo "============================================================"

    if [[ ! -f "${VENV}/bin/activate" ]]; then
        echo "[MISSING] venv does not exist: ${VENV}"
        echo "          → Run setup_envs.sh to build it."
        return
    fi

    source "${VENV}/bin/activate"
    export PATH="${VENV}/bin:${PATH}"

    echo "python  : $(which python)"
    echo "pip     : $(which pip)"
    echo ""

    # pyvenv.cfg
    echo "--- pyvenv.cfg ---"
    cat "${VENV}/pyvenv.cfg"
    echo ""

    # torch
    python - <<PYEOF
import sys
print(f"Python sys.path[:3]: {sys.path[:3]}")

try:
    import torch
    print(f"[OK] torch         : {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"     torch location: {torch.__file__}")
    cuda_ok = torch.version.cuda == "${EXPECTED_CUDA}"
    if not cuda_ok:
        print(f"[MISMATCH] Expected CUDA ${EXPECTED_CUDA}, got {torch.version.cuda}")
    else:
        print(f"[OK] torch CUDA matches expected ${EXPECTED_CUDA}")
except ImportError as e:
    print(f"[FAIL] torch import: {e}")

try:
    import torchvision
    print(f"[OK] torchvision   : {torchvision.__version__}")
    print(f"     tv  location  : {torchvision.__file__}")
    from torchvision.extension import _check_cuda_version
    _check_cuda_version()
    print("[OK] torchvision CUDA check: PASSED")
except ImportError as e:
    print(f"[FAIL] torchvision import: {e}")
except Exception as e:
    print(f"[FAIL] torchvision CUDA check: {e}")

for pkg in ('basicsr', 'skimage', 'cv2', 'einops', 'lmdb', 'yaml', 'tqdm', 'tensorboard', 'timm'):
    try:
        m = __import__(pkg)
        print(f"[OK] {pkg}")
    except ImportError as e:
        print(f"[FAIL] {pkg}: {e}")

# Check for user site contamination
import site
if site.ENABLE_USER_SITE:
    print("[WARN] ~/.local is in sys.path! PYTHONNOUSERSITE is not effective here.")
else:
    print("[OK] User site isolation active (no ~/.local contamination)")
PYEOF

    deactivate
    echo ""
}

check_venv "${VENV_CU118}" "11.8" "A6000 venv (cu118)"
check_venv "${VENV_CU121}" "12.1" "Blackwell venv (cu121)"

echo "============================================================"
echo " System torchrun location (the one that causes the bug)"
echo "============================================================"
which torchrun 2>/dev/null || echo "(not found in PATH)"
module load python/3.10.pytorch 2>/dev/null || true
which torchrun 2>/dev/null || echo "(not found after module load)"
echo ""
echo "If you see /opt/ohpc/... above, that's the system torchrun."
echo "The fixed scripts ensure the venv torchrun is always used instead."
