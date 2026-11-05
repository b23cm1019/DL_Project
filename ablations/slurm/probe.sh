#!/bin/bash
# GPU job for the analysis scripts: analyzer_probe (S9 + S2) or controllability (X2).
# Run from the repo root, after `export OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt`:
#   sbatch --job-name=abl_s9s2 --export=ALL,SCRIPT=analyzer_probe,CONFIG=s3_ref_planb_200k.yml,ARGS="..." ablations/slurm/probe.sh
# CONFIG is rendered and passed as --config; ARGS is appended unchanged.
#SBATCH --partition=phd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --output=ablations/logs/%x_%j.out
#SBATCH --error=ablations/logs/%x_%j.err
set -euo pipefail
: "${SCRIPT:?set SCRIPT=analyzer_probe|controllability}" "${CONFIG:?set CONFIG=<template>.yml}"
: "${OUT_ROOT:?export OUT_ROOT first, e.g. /scratch/p26cs0003/krish/sevdcpt}"
cd "${SLURM_SUBMIT_DIR}"
[[ -z "${VENV_BASE:-}" && -d "${HOME}/krish/envs/dl_project" ]] && export VENV_BASE="${HOME}/krish/envs"
source slurm/portable/env.sh
source ablations/slurm/_render.sh
render_config "${CONFIG}"
# shellcheck disable=SC2086
python "ablations/scripts/${SCRIPT}.py" --config "${RENDERED}" ${ARGS:-}
