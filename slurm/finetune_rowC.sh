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

# ── Environment ────────────────────────────────────────────────────────────────
# Load the system Python 3.10 module first (provides libpython3.10.so.1.0),
# then activate the project venv which was built on top of it.
module load python/3.10.pytorch
source /scratch/p24cs0203/krish/envs/dl_project/bin/activate

# ── Verify no Blackwell GPU assigned (CUDA 12.1, incompatible) ────────────────
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
echo "[INFO] GPU      : $GPU_NAME"
if echo "$GPU_NAME" | grep -qi "blackwell\|PRO 6000"; then
    echo "[ERROR] Assigned a Blackwell GPU (CUDA 12.1) — incompatible with this venv."
    echo "        Re-submit and SLURM will assign a different GPU."
    exit 1
fi

# ── Verify torchvision CUDA matches PyTorch ────────────────────────────────────
TV_IMPORT=$(python -c "import torchvision; print('ok')" 2>/dev/null || echo "fail")
if [ "$TV_IMPORT" = "fail" ]; then
    echo "[WARN] torchvision mismatch — reinstalling for cu118..."
    pip install --quiet torchvision --index-url https://download.pytorch.org/whl/cu118
fi

python -c "
import torch, torchvision
print('[INFO] PyTorch    :', torch.__version__, 'CUDA:', torch.version.cuda)
print('[INFO] torchvision:', torchvision.__version__)
print('[INFO] GPU        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"

# ── Run ────────────────────────────────────────────────────────────────────────
cd /scratch/p24cs0203/krish/projects/DL_Project
echo "Node    : $(hostname)"
echo "GPU(s)  : ${CUDA_VISIBLE_DEVICES}"
echo "Started : $(date)"

echo "=== STARTING ROW C FINE-TUNING ==="
bash run_experiments.sh row_c_finetune
echo "=== ROW C FINE-TUNING COMPLETE: $(date) ==="
