#!/bin/bash
# Train one of the frozen Row-C condition pilots.
# Usage: sbatch --export=ALL,PILOT_STAGE=e1_direct slurm/train_conditioned_pilot.sh
# PILOT_STAGE: e1_direct, e2_clip_text, e3_zero, e4_shuffled, e5_random
#SBATCH --job-name=pilot_train
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/pilot_train_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/pilot_train_%j.err
#SBATCH --time=2-00:00:00
set -euo pipefail

PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh

# _cluster_env.sh supplies a generic default (29500), which collides when
# several pilot jobs run on cn07. Use PILOT_MASTER_PORT for an explicit
# override; otherwise derive a unique rendezvous port from the Slurm job ID.
if [[ -n "${PILOT_MASTER_PORT:-}" ]]; then
  MASTER_PORT="$PILOT_MASTER_PORT"
else
  MASTER_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
fi
export MASTER_PORT
echo "[INFO] Pilot rendezvous port: ${MASTER_PORT}"

case "${PILOT_STAGE:-}" in
  e1_direct)   STAGE=pilot_e1_direct;;
  e2_clip_text) STAGE=pilot_e2_clip_text;;
  e3_zero)     STAGE=pilot_e3_zero;;
  e4_shuffled) STAGE=pilot_e4_shuffled;;
  e5_random)   STAGE=pilot_e5_random;;
  *) echo "Set PILOT_STAGE to e1_direct, e2_clip_text, e3_zero, e4_shuffled, or e5_random" >&2; exit 2;;
esac

# Resume from the latest saved state if a prior pilot run reached a checkpoint.
# This is useful when validation or another transient job error occurs after
# a checkpoint has already been written.
bash run_experiments.sh "$STAGE" --auto_resume
