#!/bin/bash
#SBATCH --job-name=dcpt_finetune_rowC
#SBATCH --partition=phd
#SBATCH --account=root
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/p24cs0203/krish/logs/finetune_rowC_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/finetune_rowC_%j.err

# ── Load base Python module (provides libpython3.10.so.1.0) ───────────────────
module load python/3.10.pytorch

# Ignore ~/.local packages — they override the venv and cause lib conflicts
export PYTHONNOUSERSITE=1

# ── Detect assigned GPU and activate matching venv ────────────────────────────
# SLURM remaps assigned GPU to index 0 inside the job.
# Use nvidia-smi -i 0 to query only the assigned GPU.
GPU_NAME=$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null || echo "unknown")
echo "[INFO] Assigned GPU : $GPU_NAME"

if echo "$GPU_NAME" | grep -qi "blackwell\|PRO 6000"; then
    echo "[INFO] Blackwell GPU — activating cu121 venv"
    VENV="/scratch/p24cs0203/krish/envs/dl_project_cu121"
else
    echo "[INFO] A6000 GPU — activating cu118 venv"
    VENV="/scratch/p24cs0203/krish/envs/dl_project"
fi

if [ ! -f "$VENV/bin/activate" ]; then
    echo "[ERROR] venv not found: $VENV"
    exit 1
fi

source "$VENV/bin/activate"
echo "[INFO] Activated venv : $VENV"

# ── Verify torchvision ────────────────────────────────────────────────────────
TV_IMPORT=$(python -c "import torchvision; print('ok')" 2>/dev/null || echo "fail")
if [ "$TV_IMPORT" = "fail" ]; then
    if echo "$GPU_NAME" | grep -qi "blackwell\|PRO 6000"; then
        echo "[WARN] reinstalling torchvision for cu121..."
        pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        echo "[WARN] reinstalling torchvision for cu118..."
        pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu118
    fi
fi

# ── Environment summary ───────────────────────────────────────────────────────
python -c "
import torch, torchvision
print('[INFO] PyTorch     :', torch.__version__, ' CUDA:', torch.version.cuda)
print('[INFO] torchvision :', torchvision.__version__)
print('[INFO] GPU         :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

# ── Run ───────────────────────────────────────────────────────────────────────
cd /scratch/p24cs0203/krish/projects/DL_Project
echo "Node    : $(hostname)"
echo "GPU(s)  : $CUDA_VISIBLE_DEVICES"
echo "Started : $(date)"

echo "=== STARTING ROW C FINE-TUNING ==="
bash run_experiments.sh row_c_finetune
echo "=== ROW C FINE-TUNING COMPLETE: $(date) ==="
