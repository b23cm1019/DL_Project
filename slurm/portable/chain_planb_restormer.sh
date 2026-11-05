#!/bin/bash
# One GPU allocation, three phases in a row (starts after Restormer Stage 1 job 46444 ends):
#   Phase 1  NAFNet Plan B full fine-tuning, auto-resumed from p24cs0203's 550k checkpoint (to 750k)
#   Phase 2  Restormer Stage 2 (micro-batch 8 x 2)     -- runs whether Phase 1 succeeded or failed
#   Phase 3  Restormer condition cache + Restormer Stage 3 (Plan B, 750k) until the time limit
#            -- only if Phase 2 produced its checkpoint; auto-resumes in a later job
# Every phase auto-resumes, so resubmitting this script continues where it stopped.
#
# Submit from the repo root (this account), after `export OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt`:
#   sbatch --dependency=afterany:46444 --nodelist=cn07 --job-name=chain_pb_rest slurm/portable/chain_planb_restormer.sh
# Options (via --export=ALL,...):
#   RESUME_ITER=550000   Plan B checkpoint to start from (copied from PLANB_SRC)
#   PHASES="1 2 3"       which phases to run (e.g. PHASES="2 3" = Restormer only)
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5-18:00:00
#SBATCH --output=portable_logs/%x_%j.out
#SBATCH --error=portable_logs/%x_%j.err
set -euo pipefail
: "${OUT_ROOT:?export OUT_ROOT first, e.g. /scratch/p26cs0003/krish/sevdcpt}"
cd "${SLURM_SUBMIT_DIR}"
[[ -z "${VENV_BASE:-}" && -d "${HOME}/krish/envs/dl_project" ]] && export VENV_BASE="${HOME}/krish/envs"
source slurm/portable/env.sh
set +e   # a failing phase must not stop the next one

PHASES="${PHASES:-1 2 3}"
RESUME_ITER="${RESUME_ITER:-550000}"
PLANB_SRC="${PLANB_SRC:-/scratch/p24cs0203/krish/sevdcpt/experiments/SevFull_PlanB_NAFNet_Full750k}"
PLANB_DST="${OUT_ROOT}/experiments/SevFull_PlanB_NAFNet_Full750k"
REST_S2="${OUT_ROOT}/experiments/SeverityRestormer_Stage2_CDDAdapt_1gpu/models"
declare -A STATUS

log() { echo "[CHAIN $(date '+%F %T')] $*"; }

# render + preflight + torchrun, exactly as slurm/portable/train.sh does
train() {
  local config="$1"
  local template="${PROJECT_ROOT}/options/cdd_experiments/portable/${config}"
  local rendered="${OUT_ROOT}/configs/${config}"
  mkdir -p "${OUT_ROOT}/configs"
  sed -e "s#@OUT_ROOT@#${OUT_ROOT}#g" -e "s#@SHARED_ROOT@#${SHARED_ROOT}#g" "${template}" > "${rendered}" || return 2
  if grep -q "@OUT_ROOT@\|@SHARED_ROOT@" "${rendered}"; then log "unrendered placeholder in ${rendered}"; return 2; fi
  python - "${rendered}" <<'PY' || return 2
import sys
from pathlib import Path
from basicsr.utils.options import yaml_load
opt = yaml_load(sys.argv[1])
missing = [d["manifest"] for d in opt["datasets"].values() if "manifest" in d and not Path(d["manifest"]).is_file()]
states = Path(opt["path"]["experiments_root"]) / "training_states"
resuming = states.is_dir() and any(states.glob("*.state"))
if not resuming:
    for key in ("pretrain_network_g", "pretrain_network_extractor"):
        value = opt["path"].get(key)
        if value and not Path(value).is_file():
            missing.append(value)
if missing:
    sys.exit("[ERROR] missing inputs:\n  " + "\n  ".join(missing))
print(f"[INFO] preflight OK ({'resuming' if resuming else 'fresh start'})")
PY
  local port="$((20000 + (SLURM_JOB_ID % 30000) + RANDOM % 9000))"
  python -m torch.distributed.run --nproc_per_node=1 --master-port "${port}" \
    basicsr/severity_train.py -opt "${rendered}" --launcher pytorch --auto_resume
}

# ------------------------------------------------------------ Phase 1: Plan B NAFNet from 550k
if [[ " ${PHASES} " == *" 1 "* ]]; then
  log "Phase 1: NAFNet Plan B, resume from ${RESUME_ITER}"
  mkdir -p "${PLANB_DST}/models" "${PLANB_DST}/training_states"
  if ! ls "${PLANB_DST}"/training_states/*.state >/dev/null 2>&1; then
    # First start here: bring over exactly the requested checkpoint (weights + EMA + optimizer state).
    cp -v "${PLANB_SRC}/models/net_g_${RESUME_ITER}.pth" "${PLANB_DST}/models/" \
      && cp -v "${PLANB_SRC}/training_states/${RESUME_ITER}.state" "${PLANB_DST}/training_states/"
    copy_rc=$?
  else
    copy_rc=0
    log "Plan B already has local states (resuming from the newest): $(ls "${PLANB_DST}/training_states" | tr '\n' ' ')"
  fi
  if [[ "${copy_rc}" -ne 0 ]]; then
    log "Phase 1 SKIPPED: could not copy the ${RESUME_ITER} checkpoint from ${PLANB_SRC} (ACL? not saved yet?)"
    STATUS[1]="skipped (copy failed)"
  else
    train nafnet_planb_full750k.yml
    rc=$?
    STATUS[1]="exit ${rc}"
    log "Phase 1 finished: exit ${rc}"
  fi
fi

# ------------------------------------------------------------ Phase 2: Restormer Stage 2
if [[ " ${PHASES} " == *" 2 "* ]]; then
  REST_S1="${OUT_ROOT}/experiments/SeverityRestormer_Stage1_LSDIR80k_1gpu/models"
  if [[ ! -f "${REST_S1}/net_g_latest.pth" ]]; then
    # Safety net: Stage 1 job ended early (crash / time limit) -> finish it first (auto-resume).
    log "Phase 2a: Restormer Stage 1 is unfinished -> resuming it first"
    train restormer_stage1_synthetic_1gpu.yml
    log "Phase 2a finished: exit $?"
  fi
  log "Phase 2: Restormer Stage 2 (runs regardless of Phase 1)"
  train restormer_stage2_cdd_adapt_1gpu.yml
  rc=$?
  STATUS[2]="exit ${rc}"
  log "Phase 2 finished: exit ${rc}"
fi

# ------------------------------------------------------------ Phase 3: Restormer cache + Stage 3
if [[ " ${PHASES} " == *" 3 "* ]]; then
  if [[ -f "${REST_S2}/net_g_latest.pth" && -f "${REST_S2}/net_extractor_latest.pth" ]]; then
    CACHE_OUT="${OUT_ROOT}/severity_full_restormer"
    if [[ ! -f "${CACHE_OUT}/manifests/train_full.jsonl" ]]; then
      log "Phase 3a: cache Restormer conditions"
      python scripts/cache_severity_full_conditions.py --backbone restormer \
        --generator "${REST_S2}/net_g_latest.pth" --extractor "${REST_S2}/net_extractor_latest.pth" \
        --out "${CACHE_OUT}" --temperature_fit
      rc=$?
    else
      rc=0
      log "Phase 3a: cache already present"
    fi
    if [[ "${rc}" -eq 0 ]]; then
      log "Phase 3b: Restormer Stage 3 (Plan B 750k; continues until the time limit, auto-resumes later)"
      train restormer_planb_full750k.yml
      rc=$?
    fi
    STATUS[3]="exit ${rc}"
    log "Phase 3 finished: exit ${rc}"
  else
    STATUS[3]="skipped (Restormer Stage 2 checkpoint missing)"
    log "Phase 3 SKIPPED: ${REST_S2}/net_g_latest.pth not found"
  fi
fi

log "Summary: phase1=${STATUS[1]:-not run} | phase2=${STATUS[2]:-not run} | phase3=${STATUS[3]:-not run}"
