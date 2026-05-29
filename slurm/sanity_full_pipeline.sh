#!/bin/bash
#SBATCH --job-name=sanity_full_pipeline
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/csehome/p24cs0203/krish/logs/slurm/sanity_full_pipeline_%j.out
#SBATCH --error=/csehome/p24cs0203/krish/logs/slurm/sanity_full_pipeline_%j.err
#SBATCH --time=01:00:00

# ==============================================================================
# Full pipeline sanity test via SLURM
#
# Tests: Pretrain Row C → Finetune Row C → Finetune Row D
# Each stage: run + checkpoint save + resume from checkpoint
#
# Submit:
#   sbatch slurm/sanity_full_pipeline.sh
#
# Monitor:
#   tail -f /csehome/p24cs0203/krish/logs/slurm/sanity_full_pipeline_<jobid>.out
# ==============================================================================

set -euo pipefail

mkdir -p /csehome/p24cs0203/krish/logs/slurm

PROJECT_ROOT="${PROJECT_ROOT:-/csehome/p24cs0203/krish/projects/DL_Project}"
cd "${PROJECT_ROOT}"

echo "================================================================"
echo "  Sanity Full Pipeline"
echo "  Node: $(hostname) | Started: $(date)"
echo "================================================================"

# Source cluster env (GPU detection, venv activation, validation)
source "${PROJECT_ROOT}/slurm/_cluster_env.sh"

echo ""
echo "Node     : $(hostname)"
echo "GPU      : ${DCPT_GPU_NAME}"
echo "GPU class: ${DCPT_GPU_CLASS}"
echo "Job ID   : ${SLURM_JOB_ID}"
echo ""

# Run the full pipeline sanity test
bash "${PROJECT_ROOT}/sanity_full_pipeline.sh"

echo ""
echo "=== SANITY FULL PIPELINE COMPLETE : $(date) ==="
