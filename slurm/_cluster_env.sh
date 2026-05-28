#!/bin/bash
# =============================================================================
# slurm/_cluster_env.sh  —  Source at the TOP of every SLURM/interactive job.
#
# Fixes applied vs original:
#   1. module command guarded with /etc/profile.d/modules.sh source
#      (original failed silently in non-login SLURM shells → set -e killed job)
#   2. DCPT_CUDA_FLAVOR exported  (original forgot → run_experiments.sh exited 1)
#   3. All paths updated to new folder structure:
#        datasets : /scratch/p24cs0203/krish/datasets/CDD11
#        project  : /csehome/p24cs0203/krish/projects/DL_Project
#        envs     : /csehome/p24cs0203/krish/envs/
#        outputs  : /csehome/p24cs0203/krish/outputs/
#        logs     : /csehome/p24cs0203/krish/logs/
#   4. basicsr.pth written into venv if missing (Blackwell fix)
#   5. GPU detection uses -i 0 (SLURM-remapped device only)
# =============================================================================

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────────────────────────
# Project lives in csehome; datasets live in scratch.
export HOME_ROOT="/csehome/p24cs0203/krish"
export SCRATCH_ROOT="/scratch/p24cs0203/krish"

export PROJECT_ROOT="${HOME_ROOT}/projects/DL_Project"
export VENV_BASE="${HOME_ROOT}/envs"
export LOG_ROOT="${HOME_ROOT}/logs"

# Dataset root — actual layout:
#   /scratch/p24cs0203/krish/datasets/CDD11/train/CDD-11_train/<degradation>/
#   /scratch/p24cs0203/krish/datasets/CDD11/test/<degradation>/
export DCPT_DATA_ROOT="${SCRATCH_ROOT}/datasets/CDD11"

# Output dirs
export BASICSR_EXPERIMENTS_ROOT="${HOME_ROOT}/outputs"
export BASICSR_MODELS_ROOT="${HOME_ROOT}/checkpoints"
export BASICSR_TRAINING_STATES_ROOT="${HOME_ROOT}/training_states"

# ── Load module system ────────────────────────────────────────────────────────
# SLURM batch jobs run non-login, non-interactive shells — 'module' is not
# available unless we source the init script explicitly first.
if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
elif [[ -f /etc/profile.d/lmod.sh ]]; then
    source /etc/profile.d/lmod.sh
else
    echo "[WARN] No module init script found — proceeding without module system"
fi

module purge  2>/dev/null || true
module load python/3.10.pytorch 2>/dev/null || \
    echo "[WARN] Could not load python/3.10.pytorch — proceeding"

# Block ~/.local from polluting venvs
export PYTHONNOUSERSITE=1

# Clean any inherited Python state
unset PYTHONHOME  2>/dev/null || true
unset PYTHONPATH  2>/dev/null || true

# ── GPU detection ─────────────────────────────────────────────────────────────
# SLURM remaps the assigned GPU to device index 0 inside the job.
GPU_NAME="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null \
            | head -n1 | tr -d '\r' || echo 'unknown')"
GPU_NAME="${GPU_NAME:-unknown}"
GPU_NAME_LC="${GPU_NAME,,}"

if [[ -z "${GPU_NAME}" || "${GPU_NAME}" == "unknown" ]]; then
    echo "[ERROR] Could not detect GPU via nvidia-smi. Is a GPU allocated?"
    exit 1
fi

echo "[INFO] Assigned GPU : ${GPU_NAME}"
export DCPT_GPU_NAME="${GPU_NAME}"

# ── Venv selection ────────────────────────────────────────────────────────────
if [[ "${GPU_NAME_LC}" == *blackwell* || "${GPU_NAME_LC}" == *"pro 6000"* ]]; then
    export DCPT_GPU_CLASS="blackwell"
    export DCPT_CUDA_FLAVOR="cu121"
    export DCPT_EXPECTED_TORCH_CUDA="12.1"
    export VENV_ROOT="${VENV_BASE}/dl_project_cu121"
    echo "[INFO] GPU class     : Blackwell → cu121 venv"
else
    export DCPT_GPU_CLASS="a6000"
    export DCPT_CUDA_FLAVOR="cu118"
    export DCPT_EXPECTED_TORCH_CUDA="11.8"
    export VENV_ROOT="${VENV_BASE}/dl_project"
    echo "[INFO] GPU class     : A6000 → cu118 venv"
fi

# ── Validate venv exists ──────────────────────────────────────────────────────
if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
    echo "[ERROR] venv not found: ${VENV_ROOT}"
    echo "        Run setup_envs.sh on cn07 first."
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
source "${VENV_ROOT}/bin/activate"
export PATH="${VENV_ROOT}/bin:${PATH}"

# ── Ensure basicsr is on PYTHONPATH via .pth (Blackwell fix) ─────────────────
PTH_FILE="${VENV_ROOT}/lib/python3.10/site-packages/dcpt_project.pth"
if [[ ! -f "${PTH_FILE}" ]]; then
    echo "[INFO] Writing basicsr .pth file → ${PTH_FILE}"
    echo "${PROJECT_ROOT}" > "${PTH_FILE}"
fi

# ── Torch stack verification ──────────────────────────────────────────────────
python - <<PYEOF
import os, sys, torch, torchvision

torch_cuda = torch.version.cuda or "None"
expected   = os.environ["DCPT_EXPECTED_TORCH_CUDA"]

print(f"[INFO] PyTorch     : {torch.__version__}  CUDA: {torch_cuda}")
print(f"[INFO] torchvision : {torchvision.__version__}")
print(f"[INFO] GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

if torch_cuda != expected:
    print(f"[ERROR] torch CUDA {torch_cuda} != expected {expected}", file=sys.stderr)
    sys.exit(1)

try:
    from torchvision.extension import _check_cuda_version
    _check_cuda_version()
    print(f"[INFO] torch/torchvision CUDA compat : OK")
except Exception as e:
    print(f"[ERROR] torchvision CUDA mismatch: {e}", file=sys.stderr)
    sys.exit(1)

import basicsr
print(f"[INFO] basicsr     : OK  ({basicsr.__file__})")
PYEOF

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p \
    "${LOG_ROOT}/slurm" \
    "${BASICSR_EXPERIMENTS_ROOT}" \
    "${BASICSR_MODELS_ROOT}" \
    "${BASICSR_TRAINING_STATES_ROOT}" \
    "${PROJECT_ROOT}/experiments"

export MASTER_PORT="${MASTER_PORT:-29500}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo "[INFO] Cluster env ready"
echo "[INFO] GPU class      : ${DCPT_GPU_CLASS} (${DCPT_CUDA_FLAVOR})"
echo "[INFO] VENV           : ${VENV_ROOT}"
echo "[INFO] PROJECT_ROOT   : ${PROJECT_ROOT}"
echo "[INFO] DCPT_DATA_ROOT : ${DCPT_DATA_ROOT}"
echo "[INFO] python         : $(which python)"
echo "[INFO] torchrun       : $(which torchrun)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
