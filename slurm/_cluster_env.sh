#!/bin/bash
# =============================================================================
# slurm/_cluster_env.sh  —  Source this at the TOP of every SLURM script.
#
# Responsibilities:
#   1. Detect the assigned GPU model via nvidia-smi (SLURM remaps to index 0)
#   2. Select the correct venv (cu118 for A6000, cu121 for Blackwell)
#   3. Activate the venv and hard-prepend it to PATH so venv torchrun wins
#   4. Verify that torch + torchvision CUDA versions agree — exit if not
#   5. Set all BASICSR_* / DCPT_* environment variables
#   6. Create required output directories
#
# Usage in SLURM scripts:
#   source "${PROJECT_ROOT}/slurm/_cluster_env.sh"
#
# Overridable env vars (set before sourcing):
#   VENV_CU118    — path to A6000 venv   (default: ${SCRATCH_ROOT}/envs/dl_project)
#   VENV_CU121    — path to Blackwell venv (default: ${SCRATCH_ROOT}/envs/dl_project_cu121)
#   MASTER_PORT   — torchrun master port  (default: 29500)
# =============================================================================

set -euo pipefail

# ── Resolve PROJECT_ROOT and SCRATCH_ROOT ─────────────────────────────────────
# _cluster_env.sh lives in slurm/, so PROJECT_ROOT is one level up.
_CE_SELF="${BASH_SOURCE[0]}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${_CE_SELF}")/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/logs}"

export PROJECT_ROOT SCRATCH_ROOT LOG_ROOT

# ── Load base Python module ───────────────────────────────────────────────────
# Required: provides libpython3.10.so.1.0 that both venvs link against.
# module purge avoids stale module state from parent shell.
module purge 2>/dev/null || true
module load python/3.10.pytorch 2>/dev/null || {
    echo "[WARN] Could not load python/3.10.pytorch module — proceeding without it."
}

# Prevent ~/.local from overriding venv packages (the #1 cause of version mismatches)
export PYTHONNOUSERSITE=1

# ── GPU detection ─────────────────────────────────────────────────────────────
# SLURM remaps the assigned GPU to device index 0 inside the job.
# We query index 0 only — do NOT iterate all GPUs.
GPU_NAME="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null \
            | head -n 1 | tr -d '\r' || echo 'unknown')"
GPU_NAME="${GPU_NAME:-unknown}"
GPU_NAME_LC="${GPU_NAME,,}"  # lowercase for matching

echo "[INFO] Assigned GPU : ${GPU_NAME}"
export DCPT_GPU_NAME="${GPU_NAME}"

# ── Venv selection ────────────────────────────────────────────────────────────
# Match patterns:
#   Blackwell: name contains "blackwell" OR "pro 6000" (RTX PRO 6000)
#   A6000    : everything else (includes "a6000", "a5000", etc.)
if [[ "${GPU_NAME_LC}" == *blackwell* || "${GPU_NAME_LC}" == *"pro 6000"* ]]; then
    export DCPT_GPU_CLASS="blackwell"
    export DCPT_CUDA_FLAVOR="cu121"
    export DCPT_EXPECTED_TORCH_CUDA="12.1"
    # PyTorch 2.3.1+cu121, torchvision 0.18.1+cu121
    VENV_ROOT="${VENV_CU121:-${SCRATCH_ROOT}/envs/dl_project_cu121}"
    echo "[INFO] GPU class     : Blackwell → selecting cu121 venv"
else
    export DCPT_GPU_CLASS="a6000"
    export DCPT_CUDA_FLAVOR="cu118"
    export DCPT_EXPECTED_TORCH_CUDA="11.8"
    # PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118
    VENV_ROOT="${VENV_CU118:-${SCRATCH_ROOT}/envs/dl_project}"
    echo "[INFO] GPU class     : A6000 → selecting cu118 venv"
fi
export VENV_ROOT

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
    echo "[ERROR] venv not found: ${VENV_ROOT}"
    echo "        Run setup_envs.sh on cn07 first."
    exit 1
fi

source "${VENV_ROOT}/bin/activate"

# CRITICAL: prepend venv bin AFTER activation so venv's torchrun wins over
# the system torchrun at /opt/ohpc/.../bin/torchrun.
# The system torchrun spawns system python which has the wrong torchvision.
export PATH="${VENV_ROOT}/bin:${PATH}"

echo "[INFO] Activated venv : ${VENV_ROOT}"
echo "[INFO] python         : $(command -v python)"
echo "[INFO] torchrun       : $(command -v torchrun)"

# ── Torch stack verification ──────────────────────────────────────────────────
# Verifies:
#   a) torch.version.cuda matches DCPT_EXPECTED_TORCH_CUDA
#   b) torchvision internal CUDA version matches torch's CUDA
# Exits with a descriptive error if either check fails.
_verify_torch_stack() {
python - <<'PYEOF'
import os, sys
import torch
import torchvision

torch_ver   = torch.__version__
tv_ver      = torchvision.__version__
torch_cuda  = torch.version.cuda or "None"
expected    = os.environ["DCPT_EXPECTED_TORCH_CUDA"]
flavor      = os.environ["DCPT_CUDA_FLAVOR"]

print(f"[INFO] PyTorch     : {torch_ver}  CUDA: {torch_cuda}")
print(f"[INFO] torchvision : {tv_ver}")
print(f"[INFO] GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

# Check 1: torch CUDA version
if torch_cuda != expected:
    print(
        f"[ERROR] Torch CUDA mismatch: expected {expected} for {flavor}, got {torch_cuda}.\n"
        f"        Re-run setup_envs.sh to rebuild the venv.",
        file=sys.stderr,
    )
    sys.exit(2)

# Check 2: torchvision ↔ torch CUDA version internal check
try:
    from torchvision.extension import _check_cuda_version
    _check_cuda_version()
    print(f"[INFO] torch/torchvision CUDA compatibility : OK ({expected})")
except Exception as exc:
    print(
        f"[ERROR] torchvision CUDA compatibility check failed: {exc}\n"
        f"        This means torchvision was built against a different CUDA than torch.\n"
        f"        Re-run setup_envs.sh to rebuild the venv.",
        file=sys.stderr,
    )
    sys.exit(3)
PYEOF
}

if ! _verify_torch_stack; then
    VERIFY_STATUS=$?
    echo "[ERROR] Torch stack verification failed (exit ${VERIFY_STATUS}). Aborting."
    exit "${VERIFY_STATUS}"
fi

# ── BASICSR / DCPT environment variables ─────────────────────────────────────
export DCPT_DATA_ROOT="${DCPT_DATA_ROOT:-${SCRATCH_ROOT}/datasets/CDD11}"
export BASICSR_EXPERIMENTS_ROOT="${BASICSR_EXPERIMENTS_ROOT:-${SCRATCH_ROOT}/outputs}"
export BASICSR_MODELS_ROOT="${BASICSR_MODELS_ROOT:-${SCRATCH_ROOT}/checkpoints}"
export BASICSR_TRAINING_STATES_ROOT="${BASICSR_TRAINING_STATES_ROOT:-${SCRATCH_ROOT}/training_states}"
export BASICSR_LOG_ROOT="${BASICSR_LOG_ROOT:-${LOG_ROOT}}"
export BASICSR_VIS_ROOT="${BASICSR_VIS_ROOT:-${SCRATCH_ROOT}/outputs/visualizations}"
export BASICSR_RESULTS_ROOT="${BASICSR_RESULTS_ROOT:-${SCRATCH_ROOT}/outputs/test_results}"
export BASICSR_TB_ROOT="${BASICSR_TB_ROOT:-${LOG_ROOT}/tb_logger}"

export MASTER_PORT="${MASTER_PORT:-29500}"

# Helps prevent CUDA OOM on 48 GB A6000 with large batch sizes
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

# ── Create required directories ───────────────────────────────────────────────
mkdir -p \
    "${LOG_ROOT}/slurm" \
    "${BASICSR_EXPERIMENTS_ROOT}" \
    "${BASICSR_MODELS_ROOT}" \
    "${BASICSR_TRAINING_STATES_ROOT}" \
    "${BASICSR_LOG_ROOT}" \
    "${BASICSR_VIS_ROOT}" \
    "${BASICSR_RESULTS_ROOT}" \
    "${BASICSR_TB_ROOT}"

# ── Final summary ─────────────────────────────────────────────────────────────
echo "[INFO] Cluster env ready"
echo "[INFO] GPU class  : ${DCPT_GPU_CLASS} (${DCPT_CUDA_FLAVOR})"
echo "[INFO] VENV       : ${VENV_ROOT}"
echo "[INFO] Data root  : ${DCPT_DATA_ROOT}"
echo "[INFO] Project    : ${PROJECT_ROOT}"
hostname
python3 --version
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
