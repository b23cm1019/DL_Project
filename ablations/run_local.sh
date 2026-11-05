#!/bin/bash
# Run ablation training on a machine WITHOUT Slurm (e.g. the dedicated A5000 server), one config
# after another. Every run auto-resumes, so re-running this script after a crash or reboot continues
# where each run stopped; finished runs exit in about a minute.
#
# One-time setup on that machine: the repo (git pull), a Python env with torch, and the shared data
#   $SHARED_ROOT/datasets/CDD11, $SHARED_ROOT/severity_full/manifests (paths rewritten to this machine),
#   $SHARED_ROOT/experiments/SeverityDCPT_Stage2_CDDAdapt/models/{net_g,net_extractor}_latest.pth
# Run from the repo root inside `screen`, with the env activated:
#   export SHARED_ROOT=/DATA/user/akshat_mishra1/krish/sevdcpt_shared OUT_ROOT=/DATA/user/akshat_mishra1/krish/sevdcpt
#   bash ablations/run_local.sh s3_k4_nokl_100k.yml s3_f1_noskip_100k.yml s3_s4_shared_100k.yml
# Logs: $OUT_ROOT/ablations/logs/<config>.log ; GPU: CUDA_VISIBLE_DEVICES (default 0).
set -uo pipefail
: "${OUT_ROOT:?export OUT_ROOT}" "${SHARED_ROOT:?export SHARED_ROOT}"
[[ $# -ge 1 ]] || { echo "usage: bash ablations/run_local.sh CONFIG.yml [CONFIG.yml ...]" >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PYTHON:-python}"
mkdir -p "${OUT_ROOT}/ablations/configs" "${OUT_ROOT}/ablations/logs"
declare -a SUMMARY

for config in "$@"; do
  template="ablations/configs/${config}"
  rendered="${OUT_ROOT}/ablations/configs/${config}"
  log="${OUT_ROOT}/ablations/logs/${config%.yml}.log"
  if [[ ! -f "${template}" ]]; then SUMMARY+=("${config}: MISSING TEMPLATE"); continue; fi
  sed -e "s#@OUT_ROOT@#${OUT_ROOT}#g" -e "s#@SHARED_ROOT@#${SHARED_ROOT}#g" "${template}" > "${rendered}"
  echo "[$(date '+%F %T')] ${config} -> ${log}"
  "${PY}" - "${rendered}" <<'PY' 2>&1 | tee -a "${log}"
import sys
from pathlib import Path
from basicsr.utils.options import yaml_load
opt = yaml_load(sys.argv[1])
missing = [d[k] for d in opt["datasets"].values() for k in ("manifest", "dataroot_gt", "dataroot_lq")
           if d.get(k) and not Path(d[k]).exists()]
states = Path(opt["path"]["experiments_root"]) / "training_states"
resuming = states.is_dir() and any(states.glob("*.state"))
if not resuming:
    for p in (opt["path"].get("pretrain_network_g"), opt["path"].get("pretrain_network_extractor"),
              opt.get("network_g", {}).get("shared_extractor_path")):
        if p and not Path(p).is_file():
            missing.append(p)
if missing:
    sys.exit("[ERROR] missing inputs:\n  " + "\n  ".join(missing))
print(f"[INFO] preflight OK ({'resuming' if resuming else 'fresh start'})")
PY
  if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then SUMMARY+=("${config}: PREFLIGHT FAILED"); continue; fi
  port=$((20000 + RANDOM % 20000))
  "${PY}" -m torch.distributed.run --nproc_per_node=1 --master-port "${port}" \
    ablations/train.py -opt "${rendered}" --launcher pytorch --auto_resume 2>&1 | tee -a "${log}"
  rc=${PIPESTATUS[0]}
  SUMMARY+=("${config}: exit ${rc}")
  echo "[$(date '+%F %T')] ${config} finished with exit ${rc}"
done
echo; echo "Summary:"; printf '  %s\n' "${SUMMARY[@]}"
