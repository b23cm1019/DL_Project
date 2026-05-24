#!/bin/bash
#SBATCH --job-name=dcpt_pretrain_rowC
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pretrain_rowC_%j.out
#SBATCH --error=logs/pretrain_rowC_%j.err

# ── Activate Python venv ───────────────────────────────────────────────────────
# SLURM jobs do not source ~/.bashrc so we activate the venv explicitly.
source /scratch/p24cs0203/krish/envs/dl_project/bin/activate

# ── Guard: block RTX PRO 6000 Blackwell GPUs (CUDA 12.1, 98GB) ────────────────
# SLURM on cn07 may assign any of the 6 GPUs. GPUs 1 and 2 are Blackwell cards
# compiled with CUDA 12.1, incompatible with this venv (PyTorch CUDA 11.8).
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
echo "[INFO] Assigned GPU : $GPU_NAME"
if echo "$GPU_NAME" | grep -qi "blackwell\|PRO 6000"; then
    echo "[ERROR] Assigned GPU is RTX PRO 6000 Blackwell (CUDA 12.1)."
    echo "        Incompatible with this venv (PyTorch CUDA 11.8)."
    echo "        Re-submit — SLURM will assign a different GPU next time."
    exit 1
fi

# ── Auto-fix torchvision CUDA mismatch ────────────────────────────────────────
# Original crash: PyTorch CUDA=11.8, torchvision CUDA=12.1
TV_IMPORT=$(python -c "import torchvision; print('ok')" 2>/dev/null || echo "fail")
if [ "$TV_IMPORT" = "fail" ]; then
    echo "[WARN] torchvision import failed — reinstalling for cu118..."
    pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu118
    echo "[INFO] torchvision reinstalled."
fi

# ── Environment summary ────────────────────────────────────────────────────────
python -c "
import torch, torchvision
print('[INFO] PyTorch     :', torch.__version__, ' CUDA:', torch.version.cuda)
print('[INFO] torchvision :', torchvision.__version__)
print('[INFO] GPU         :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

# ── cd to project root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
mkdir -p logs

echo "Node    : $(hostname)"
echo "GPU(s)  : ${CUDA_VISIBLE_DEVICES}"
echo "Started : $(date)"

echo "=== STARTING ROW C PRE-TRAINING ==="
bash run_experiments.sh row_c_pretrain
echo "=== ROW C PRE-TRAINING COMPLETE: $(date) ==="
