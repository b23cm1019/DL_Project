#!/bin/bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$(cd "${PROJECT_ROOT}/../.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/logs}"

module purge
module load python/3.10.pytorch
export PYTHONNOUSERSITE=1

GPU_NAME="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '\r')"
GPU_NAME="${GPU_NAME:-unknown}"
GPU_NAME_LC="${GPU_NAME,,}"
echo "[INFO] Assigned GPU : ${GPU_NAME}"

if [[ "${GPU_NAME_LC}" == *blackwell* || "${GPU_NAME_LC}" == *"pro 6000"* ]]; then
  export DCPT_GPU_CLASS="blackwell"
  export DCPT_CUDA_FLAVOR="cu121"
  export DCPT_EXPECTED_TORCH_CUDA="12.1"
  VENV_ROOT="${VENV_CU121:-${SCRATCH_ROOT}/envs/dl_project_cu121}"
else
  export DCPT_GPU_CLASS="a6000"
  export DCPT_CUDA_FLAVOR="cu118"
  export DCPT_EXPECTED_TORCH_CUDA="11.8"
  VENV_ROOT="${VENV_CU118:-${SCRATCH_ROOT}/envs/dl_project}"
fi

export DCPT_GPU_NAME="${GPU_NAME}"
export VENV_ROOT

if [[ ! -f "${VENV_ROOT}/bin/activate" ]]; then
  echo "[ERROR] venv not found: ${VENV_ROOT}"
  exit 1
fi

echo "[INFO] Selected runtime : ${DCPT_CUDA_FLAVOR} (expected torch CUDA ${DCPT_EXPECTED_TORCH_CUDA})"
source "${VENV_ROOT}/bin/activate"
echo "[INFO] Activated venv : ${VENV_ROOT}"

export DCPT_DATA_ROOT="${DCPT_DATA_ROOT:-${SCRATCH_ROOT}/datasets/CDD11}"
export BASICSR_EXPERIMENTS_ROOT="${BASICSR_EXPERIMENTS_ROOT:-${SCRATCH_ROOT}/outputs}"
export BASICSR_MODELS_ROOT="${BASICSR_MODELS_ROOT:-${SCRATCH_ROOT}/checkpoints}"
export BASICSR_TRAINING_STATES_ROOT="${BASICSR_TRAINING_STATES_ROOT:-${SCRATCH_ROOT}/training_states}"
export BASICSR_LOG_ROOT="${BASICSR_LOG_ROOT:-${LOG_ROOT}}"
export BASICSR_VIS_ROOT="${BASICSR_VIS_ROOT:-${SCRATCH_ROOT}/outputs/visualizations}"
export BASICSR_RESULTS_ROOT="${BASICSR_RESULTS_ROOT:-${SCRATCH_ROOT}/outputs/test_results}"
export BASICSR_TB_ROOT="${BASICSR_TB_ROOT:-${LOG_ROOT}/tb_logger}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

verify_torch_stack() {
  python - <<'PY'
import os
import sys

import torch
import torchvision

print("[INFO] PyTorch     :", torch.__version__, " CUDA:", torch.version.cuda)
print("[INFO] torchvision :", torchvision.__version__)
print("[INFO] GPU         :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

expected_cuda = os.environ["DCPT_EXPECTED_TORCH_CUDA"]
actual_cuda = torch.version.cuda or "None"
if actual_cuda != expected_cuda:
    print(
        f"[ERROR] Expected torch CUDA {expected_cuda} for runtime {os.environ['DCPT_CUDA_FLAVOR']}, "
        f"but found {actual_cuda}.",
        file=sys.stderr,
    )
    sys.exit(2)

try:
    from torchvision.extension import _check_cuda_version

    _check_cuda_version()
except Exception as exc:
    print(f"[ERROR] torchvision CUDA compatibility check failed: {exc}", file=sys.stderr)
    sys.exit(3)

print(f"[INFO] torch/torchvision CUDA compatibility OK ({expected_cuda})")
PY
}

if verify_torch_stack; then
  :
else
  VERIFY_STATUS=$?
  if [[ ${VERIFY_STATUS} -eq 3 ]]; then
    echo "[WARN] Repairing torchvision for ${DCPT_CUDA_FLAVOR}..."
    pip install --quiet --force-reinstall "torchvision==0.16.2" --index-url "https://download.pytorch.org/whl/${DCPT_CUDA_FLAVOR}"
    verify_torch_stack
  else
    exit "${VERIFY_STATUS}"
  fi
fi

mkdir -p "${LOG_ROOT}/slurm"
mkdir -p "${BASICSR_EXPERIMENTS_ROOT}"
mkdir -p "${BASICSR_MODELS_ROOT}"
mkdir -p "${BASICSR_TRAINING_STATES_ROOT}"
mkdir -p "${BASICSR_LOG_ROOT}"
mkdir -p "${BASICSR_VIS_ROOT}"
mkdir -p "${BASICSR_RESULTS_ROOT}"
mkdir -p "${BASICSR_TB_ROOT}"

cd "${PROJECT_ROOT}"

echo "[INFO] Python      : $(command -v python)"
echo "[INFO] torchrun    : $(command -v torchrun)"
python3 --version
hostname
nvidia-smi
