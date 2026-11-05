#!/bin/bash
# T5 step 2/2: render the in-protocol plan (8 CPU shards, existing renderer) and write the final
# manifests (existing finalizer). CPU only. Run after t5_assets.sh, from the repo root:
#   sbatch --job-name=abl_t5_render ablations/slurm/t5_render.sh
#SBATCH --partition=phd
#SBATCH --exclude=cn07,dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=ablations/logs/%x_%j.out
#SBATCH --error=ablations/logs/%x_%j.err
set -euo pipefail
: "${OUT_ROOT:?export OUT_ROOT first, e.g. /scratch/p26cs0003/krish/sevdcpt}"
cd "${SLURM_SUBMIT_DIR}"
if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; else source /etc/profile.d/lmod.sh; fi
module purge 2>/dev/null || true
module load python/3.10.pytorch
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
T5="${OUT_ROOT}/ablations/t5_inprotocol"
[[ -f "${T5}/assets/source_assets.jsonl" ]] || { echo "[ERROR] run ablations/slurm/t5_assets.sh first" >&2; exit 2; }
NUM_SHARDS=8
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  python3 scripts/render_lsdir_cdd11_plan.py --plan "${T5}/plans/train.jsonl" --plan "${T5}/plans/val.jsonl" \
    --asset-manifest "${T5}/assets/source_assets.jsonl" --output-root "${T5}/data" \
    --shard-index "${shard}" --num-shards "${NUM_SHARDS}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
[[ "${status}" == 0 ]] || { echo "[ERROR] a render shard failed; rerun this job (finished images are skipped)" >&2; exit 1; }
python3 scripts/finalize_lsdir_cdd11_render.py --plan-root "${T5}" --data-root "${T5}/data" --num-shards "${NUM_SHARDS}" --overwrite
echo "[INFO] T5 manifests: ${T5}/data/manifests/{train,val}.jsonl"
