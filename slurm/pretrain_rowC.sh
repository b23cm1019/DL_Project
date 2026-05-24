#!/bin/bash
#SBATCH --job-name=dcpt_pretrain_rowC
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/pretrain_rowC_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/pretrain_rowC_%j.err

# ── Load base Python module (provides libpython3.10.so.1.0) ───────────────────
module load python/3.10.pytorch

# ── Detect GPU and activate matching venv ─────────────────────────────────────
# cn07 has two GPU types:
#   RTX A6000      (49GB, CUDA 11.8) → use venv: dl_project        (cu118)
#   RTX PRO 6000 Blackwell (98GB, CUDA 12.1) → use venv: dl_project_cu121 (cu121)
#
# We detect via nvidia-smi before activating any venv.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "[INFO] Assigned GPU : $GPU_NAME"

if echo "$GPU_NAME" | grep -qi "blackwell\|PRO 6000"; then
    echo "[INFO] Blackwell GPU detected — using cu121 venv"
    VENV="/scratch/p24cs0203/krish/envs/dl_project_cu121"
else
    echo "[INFO] A6000 GPU detected — using cu118 venv"
    VENV="/scratch/p24cs0203/krish/envs/dl_project"
fi

# Check if the required venv exists
if [ ! -f "$VENV/bin/activate" ]; then
    echo "[ERROR] venv not found at: $VENV"
    if echo "$GPU_NAME" | grep -qi "blackwell\|PRO 6000"; then
        echo "        Run this to create it:"
        echo "        srun --partition=phd --nodelist=cn07 --gres=gpu:1 --pty bash"
        echo "        module load python/3.10.pytorch"
        echo "        python -m venv /scratch/p24cs0203/krish/envs/dl_project_cu121"
        echo "        source /scratch/p24cs0203/krish/envs/dl_project_cu121/bin/activate"
        echo "        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        echo "        pip install -r requirements.txt  # or pip install basicsr etc."
    fi
    exit 1
fi

source "$VENV/bin/activate"
echo "[INFO] Activated venv : $VENV"

# ── Verify torchvision imports cleanly ────────────────────────────────────────
TV_IMPORT=$(python -c "import torchvision; print('ok')" 2>/dev/null || echo "fail")
if [ "$TV_IMPORT" = "fail" ]; then
    if echo "$GPU_NAME" | grep -qi "blackwell\|PRO 6000"; then
        echo "[WARN] torchvision broken — reinstalling for cu121..."
        pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "[WARN] torchvision broken — reinstalling for cu118..."
        pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu118
    fi
fi

# ── Environment summary ────────────────────────────────────────────────────────
python -c "
import torch, torchvision
print('[INFO] PyTorch     :', torch.__version__, ' CUDA:', torch.version.cuda)
print('[INFO] torchvision :', torchvision.__version__)
print('[INFO] GPU         :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

# ── Run ───────────────────────────────────────────────────────────────────────
cd /scratch/p24cs0203/krish/projects/DL_Project
echo "Node    : $(hostname)"
echo "GPU(s)  : $${CUDA_VISIBLE_DEVICES}"
echo "Started : $(date)"

echo "=== STARTING ROW C PRE-TRAINING ==="
bash run_experiments.sh row_c_pretrain
echo "=== ROW C PRE-TRAINING COMPLETE: $(date) ==="
