#!/bin/bash
# Evaluate a trained E1-E5 pilot checkpoint.
# Example:
# sbatch --export=ALL,PILOT_STAGE=e2_clip_text slurm/evaluate_conditioned_pilot.sh
#SBATCH --job-name=pilot_eval
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=cn07
#SBATCH --output=/scratch/p24cs0203/krish/logs/slurm/pilot_eval_%j.out
#SBATCH --error=/scratch/p24cs0203/krish/logs/slurm/pilot_eval_%j.err
#SBATCH --time=1-00:00:00
set -euo pipefail

PROJECT_ROOT=/csehome/p24cs0203/krish/projects/DL_Project
cd "$PROJECT_ROOT"
source slurm/_cluster_env.sh

ROOT="${CONDITIONED_PILOT_ROOT:-/scratch/p24cs0203/krish/conditioned_pilot}"
SPLIT="${PILOT_SPLIT:-val}"
case "${PILOT_STAGE:-}" in
  e1_direct)    NAME=Pilot_E1_RowC_DirectCondition; MODE=direct; DIM=4; MANIFEST=direct;;
  e2_clip_text) NAME=Pilot_E2_RowC_CLIPTextCondition; MODE=clip; DIM=512; MANIFEST=clip;;
  e3_zero)      NAME=Pilot_E3_ZeroCondition; MODE=zero; DIM=512; MANIFEST=clip;;
  e4_shuffled)  NAME=Pilot_E4_ShuffledCondition; MODE=shuffled; DIM=512; MANIFEST=clip;;
  e5_random)    NAME=Pilot_E5_RandomCondition; MODE=random; DIM=512; MANIFEST=clip;;
  *) echo "Set PILOT_STAGE to e1_direct, e2_clip_text, e3_zero, e4_shuffled, or e5_random" >&2; exit 2;;
esac

python scripts/evaluate_conditioned_pilot.py \
  --manifest "$ROOT/manifests/${SPLIT}_${MANIFEST}.jsonl" \
  --checkpoint "$ROOT/checkpoints/${NAME}/net_g_latest.pth" \
  --condition-mode "$MODE" \
  --condition-dim "$DIM" \
  --output "$ROOT/results/${NAME}/${SPLIT}"
