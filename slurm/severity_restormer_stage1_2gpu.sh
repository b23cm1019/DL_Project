#!/bin/bash
# Independent Proposed-Restormer Stage 1.  Two 24-GB-or-larger GPUs required.
# Global B=32 is preserved as B=16 on each DDP rank.
# Resume: sbatch --export=ALL,SEVERITY_AUTO_RESUME=1 slurm/severity_restormer_stage1_2gpu.sh
#SBATCH --job-name=sevrest_s1
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=5-18:00:00
#SBATCH --output=/scratch/p26cs0003/krish/logs/sevrest_s1_%j.out
#SBATCH --error=/scratch/p26cs0003/krish/logs/sevrest_s1_%j.err
set -euo pipefail

PROJECT_ROOT=/csehome/p26cs0003/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh
[[ "$(python -c 'import torch; print(torch.cuda.device_count())')" == "2" ]] || {
  echo "This run requires exactly two Slurm-visible GPUs." >&2; exit 2;
}
SYNTH_ROOT=/scratch/p26cs0003/krish/severity_cdd11_lsdir_outdoor_v1/data
for required in "$SYNTH_ROOT/manifests/train.jsonl" "$SYNTH_ROOT/manifests/val.jsonl"; do
  [[ -f "$required" ]] || { echo "Missing synthetic manifest: $required" >&2; exit 2; }
done
RESUME=()
if [[ "${SEVERITY_AUTO_RESUME:-0}" == "1" ]]; then RESUME=(--auto_resume); fi
RUN_MASTER_PORT="${SEVERITY_RESTORMER_STAGE1_MASTER_PORT:-$((20000 + (SLURM_JOB_ID % 40000)))}"
python -m torch.distributed.run --nproc_per_node=2 --master-port "$RUN_MASTER_PORT" \
  basicsr/severity_train.py -opt options/cdd_experiments/severity_restormer_stage1_synthetic_2gpu.yml \
  --launcher pytorch "${RESUME[@]}"
