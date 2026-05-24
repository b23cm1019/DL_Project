#!/bin/bash
#SBATCH --job-name=dcpt_finetune_rowB
#SBATCH --partition=phd
#SBATCH --account=p24cs0203
#SBATCH --nodelist=cn07
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/finetune_rowB_%j.out
#SBATCH --error=logs/finetune_rowB_%j.err

# GPU 0 = RTX A6000. Avoid GPUs 1 and 2 (RTX PRO 6000 Blackwell, CUDA 12.1).
export CUDA_VISIBLE_DEVICES=0

module purge
module load python/3.10.pytorch

cd ~/DL_Project
mkdir -p logs

echo "Running on node: $(hostname)"
echo "GPU selected: ${CUDA_VISIBLE_DEVICES} (RTX A6000)"
echo "Job started: $(date)"

echo "=== STARTING ROW B FINE-TUNING (500k iters, batch=32) ==="
./run_experiments.sh row_b_finetune

echo "=== ROW B FINE-TUNING COMPLETE: $(date) ==="
