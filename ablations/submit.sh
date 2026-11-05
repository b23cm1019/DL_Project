#!/bin/bash
# Submit ablation training runs as ONE dependency chain: each config runs after the previous one,
# split into --parts consecutive jobs when a partition time limit is shorter than a run (every job
# auto-resumes from the newest 10k checkpoint, so a cut-off job only loses minutes of work).
#
# Run from the repo root on a Slurm cluster, after `export OUT_ROOT=...` (and VENV_BASE if needed):
#   p26cs0003 (phd, no limit problem):
#     ablations/submit.sh s3_c2_constant_100k.yml s3_k2_nogate_100k.yml
#   b23cm1019 (btech, 12 h limit, one running job):
#     ablations/submit.sh --partition btech --time 12:00:00 --parts 2 s3_s1_typeonly_100k.yml s3_k1_nocal_100k.yml
#   continue an existing chain:   --after <jobid>
#   pin a node:                   --nodelist cn07
# Prints every job id; the last one is what a later chain should wait for (--after).
set -euo pipefail
PARTITION=phd; TIME=5-18:00:00; PARTS=1; AFTER=""; NODELIST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME="$2"; shift 2 ;;
    --parts) PARTS="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --nodelist) NODELIST="$2"; shift 2 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    --*) echo "unknown option $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
[[ $# -ge 1 ]] || { echo "usage: ablations/submit.sh [options] CONFIG.yml [CONFIG.yml ...]" >&2; exit 2; }
: "${OUT_ROOT:?export OUT_ROOT first}"
[[ -f ablations/slurm/train.sh ]] || { echo "run from the repo root" >&2; exit 2; }
for config in "$@"; do
  [[ -f "ablations/configs/${config}" ]] || { echo "missing ablations/configs/${config} (run python ablations/make_configs.py)" >&2; exit 2; }
done

prev="${AFTER}"
for config in "$@"; do
  short="${config%.yml}"; short="${short#s3_}"
  for part in $(seq 1 "${PARTS}"); do
    args=(--parsable --partition "${PARTITION}" --time "${TIME}" --job-name "abl_${short}_p${part}"
          --export "ALL,CONFIG=${config}")
    [[ -n "${prev}" ]] && args+=(--dependency "afterany:${prev}")
    [[ -n "${NODELIST}" ]] && args+=(--nodelist "${NODELIST}")
    prev=$(sbatch "${args[@]}" ablations/slurm/train.sh)
    echo "${config} part ${part}/${PARTS}: job ${prev}"
  done
done
echo "last job in this chain: ${prev}"
