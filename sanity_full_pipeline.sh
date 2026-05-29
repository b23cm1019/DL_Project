#!/bin/bash
# ==============================================================================
# sanity_full_pipeline.sh — End-to-end sanity test for the DCPT experiment pipeline
#
# Tests the COMPLETE chain with checkpoint saving and resume:
#   Pretrain Row C  →  Finetune Row C  →  Finetune Row D
#
# Each stage runs a tiny number of iterations, saves checkpoints, then resumes
# from those checkpoints to verify the full lifecycle works.
#
# Uses SANITY_ prefixed experiment names so real experiments are never touched.
#
# Usage:
#   Submit via SLURM:
#     sbatch slurm/sanity_full_pipeline.sh
#
#   Interactive (after sourcing cluster env):
#     source slurm/_cluster_env.sh
#     bash sanity_full_pipeline.sh
# ==============================================================================

set -euo pipefail

# ── Ensure cluster env is sourced ─────────────────────────────────────────────
if [[ -z "${VENV_ROOT:-}" || -z "${PROJECT_ROOT:-}" ]]; then
    echo "[ERROR] Cluster env not sourced. Run: source slurm/_cluster_env.sh"
    exit 1
fi

cd "${PROJECT_ROOT}"

# ── Configuration ─────────────────────────────────────────────────────────────
SANITY_PRETRAIN="SANITY_Pretrain_RC"
SANITY_FINETUNE_C="SANITY_Finetune_RC"
SANITY_FINETUNE_D="SANITY_Finetune_RD"

ITERS_RUN=10            # iterations for the initial run
ITERS_RESUME=15         # total iterations after resume (runs 5 more)
SAVE_FREQ=5             # save checkpoint every N iters

SCALE_OVERRIDES=(
    "--force_yml"
    "dataloader:batch_size_per_gpu=8"
    "train:optim_g:lr=1e-4"
    "train:optim_dc:lr=5e-5"
    "train:ema_decay=0"
)

export MASTER_PORT="${MASTER_PORT:-29500}"
export PATH="${VENV_ROOT}/bin:${PATH}"

# ── Helpers ───────────────────────────────────────────────────────────────────
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

check_file() {
    local label="$1"
    local filepath="$2"
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    if [[ -f "${filepath}" ]]; then
        local size
        size=$(du -h "${filepath}" 2>/dev/null | cut -f1)
        echo "    [PASS] ${label}: $(basename "${filepath}") (${size})"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo "    [FAIL] ${label}: ${filepath} NOT FOUND"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
}

banner() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
}

section() {
    echo ""
    echo "── $1 ──"
}

# Use venv python for torchrun to avoid system python issues
torchrun_cmd() {
    python -m torch.distributed.run "$@"
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
banner "SANITY TEST: Full Pipeline (Pretrain C → Finetune C → Finetune D)"

echo ""
echo "  GPU             : ${DCPT_GPU_NAME:-unknown}"
echo "  GPU class       : ${DCPT_GPU_CLASS:-unknown}"
echo "  CUDA flavor     : ${DCPT_CUDA_FLAVOR:-unknown}"
echo "  Python          : $(which python)"
echo "  PyTorch         : $(python -c 'import torch; print(torch.__version__)')"
echo "  VENV            : ${VENV_ROOT}"
echo "  PROJECT_ROOT    : ${PROJECT_ROOT}"
echo "  MASTER_PORT     : ${MASTER_PORT}"
echo ""
echo "  Iters per phase : ${ITERS_RUN} (initial) + ${ITERS_RESUME} (with resume)"
echo "  Checkpoint freq : every ${SAVE_FREQ} iters"
echo ""
echo "  Sanity experiments will be saved under:"
echo "    experiments/${SANITY_PRETRAIN}/"
echo "    experiments/${SANITY_FINETUNE_C}/"
echo "    experiments/${SANITY_FINETUNE_D}/"
echo ""

# ── Import check ──────────────────────────────────────────────────────────────
section "Pre-flight: verifying imports"
python -c "import basicsr.all_in_one_train; print('    [OK] basicsr.all_in_one_train')"
python -c "
import torch, torchvision
print(f'    [OK] torch={torch.__version__}  CUDA={torch.version.cuda}  GPU={torch.cuda.get_device_name(0)}')
print(f'    [OK] torchvision={torchvision.__version__}')
"

# ── Clean previous sanity runs ────────────────────────────────────────────────
section "Cleanup: removing previous sanity experiments"
for dir in "experiments/${SANITY_PRETRAIN}" "experiments/${SANITY_FINETUNE_C}" "experiments/${SANITY_FINETUNE_D}"; do
    if [[ -d "${dir}" ]]; then
        echo "    Removing: ${dir}"
        rm -rf "${dir}"
    fi
done
echo "    [OK] Clean slate"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1/6 — Pretrain Row C: Initial Run
# ══════════════════════════════════════════════════════════════════════════════
banner "PHASE 1/6: Pretrain Row C — Initial Run (${ITERS_RUN} iters)"
echo "  Model     : DCPTModel"
echo "  Networks  : net_g (NAFNet) + net_dc (PromptIR_NoImg_DC)"
echo "  Config    : options/cdd_experiments/pretrain_multilabel.yml"
echo "  Expect    : checkpoints at iter ${SAVE_FREQ}, ${ITERS_RUN}, and 'latest'"
echo ""

PHASE1_START=$(date +%s)

torchrun_cmd --master-port "${MASTER_PORT}" --nproc_per_node=1 \
    basicsr/all_in_one_train.py \
    -opt options/cdd_experiments/pretrain_multilabel.yml \
    --launcher pytorch \
    "${SCALE_OVERRIDES[@]}" \
    --force_yml "name=${SANITY_PRETRAIN}" \
                "train:total_iter=${ITERS_RUN}" \
                "logger:print_freq=1" \
                "logger:save_checkpoint_freq=${SAVE_FREQ}" \
                "val:val_freq=999999"

PHASE1_END=$(date +%s)
echo ""
echo "  Phase 1 training took $((PHASE1_END - PHASE1_START))s"

section "Verifying Phase 1 outputs"
EXP_PRETRAIN="experiments/${SANITY_PRETRAIN}"
check_file "net_g checkpoint @iter${SAVE_FREQ}"    "${EXP_PRETRAIN}/models/net_g_${SAVE_FREQ}.pth"
check_file "net_dc checkpoint @iter${SAVE_FREQ}"   "${EXP_PRETRAIN}/models/net_dc_${SAVE_FREQ}.pth"
check_file "Training state @iter${SAVE_FREQ}"      "${EXP_PRETRAIN}/training_states/${SAVE_FREQ}.state"
check_file "net_g checkpoint @iter${ITERS_RUN}"    "${EXP_PRETRAIN}/models/net_g_${ITERS_RUN}.pth"
check_file "net_dc checkpoint @iter${ITERS_RUN}"   "${EXP_PRETRAIN}/models/net_dc_${ITERS_RUN}.pth"
check_file "Training state @iter${ITERS_RUN}"      "${EXP_PRETRAIN}/training_states/${ITERS_RUN}.state"
check_file "net_g final (latest)"                  "${EXP_PRETRAIN}/models/net_g_latest.pth"
check_file "net_dc final (latest)"                 "${EXP_PRETRAIN}/models/net_dc_latest.pth"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2/6 — Pretrain Row C: Resume
# ══════════════════════════════════════════════════════════════════════════════
banner "PHASE 2/6: Pretrain Row C — Resume (iter ${ITERS_RUN} → ${ITERS_RESUME})"
echo "  Using     : --auto_resume"
echo "  Expect    : resumes from ${ITERS_RUN}.state, runs to iter ${ITERS_RESUME}"
echo ""

PHASE2_START=$(date +%s)

torchrun_cmd --master-port "${MASTER_PORT}" --nproc_per_node=1 \
    basicsr/all_in_one_train.py \
    -opt options/cdd_experiments/pretrain_multilabel.yml \
    --launcher pytorch --auto_resume \
    "${SCALE_OVERRIDES[@]}" \
    --force_yml "name=${SANITY_PRETRAIN}" \
                "train:total_iter=${ITERS_RESUME}" \
                "logger:print_freq=1" \
                "logger:save_checkpoint_freq=${SAVE_FREQ}" \
                "val:val_freq=999999"

PHASE2_END=$(date +%s)
echo ""
echo "  Phase 2 resume took $((PHASE2_END - PHASE2_START))s"

section "Verifying Phase 2 outputs"
check_file "Resumed net_g @iter${ITERS_RESUME}"      "${EXP_PRETRAIN}/models/net_g_${ITERS_RESUME}.pth"
check_file "Resumed net_dc @iter${ITERS_RESUME}"     "${EXP_PRETRAIN}/models/net_dc_${ITERS_RESUME}.pth"
check_file "Resumed state @iter${ITERS_RESUME}"      "${EXP_PRETRAIN}/training_states/${ITERS_RESUME}.state"
check_file "Updated net_g (latest)"                  "${EXP_PRETRAIN}/models/net_g_latest.pth"
check_file "Updated net_dc (latest)"                 "${EXP_PRETRAIN}/models/net_dc_latest.pth"

section "Listing all pretrain checkpoints"
echo "    Models:"
ls -lh "${EXP_PRETRAIN}/models/"     2>/dev/null | grep -v '^total' | awk '{print "      "$0}'
echo "    Training states:"
ls -lh "${EXP_PRETRAIN}/training_states/" 2>/dev/null | grep -v '^total' | awk '{print "      "$0}'


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3/6 — Finetune Row C: Initial Run
# ══════════════════════════════════════════════════════════════════════════════
banner "PHASE 3/6: Finetune Row C — Initial Run (${ITERS_RUN} iters)"
echo "  Model     : SRModel (net_g only, no net_dc)"
echo "  Weights   : loading net_g from pretrain → ${EXP_PRETRAIN}/models/net_g_latest.pth"
echo "  Config    : options/cdd_experiments/finetune_multilabel.yml"
echo ""

PHASE3_START=$(date +%s)

torchrun_cmd --master-port "${MASTER_PORT}" --nproc_per_node=1 \
    basicsr/all_in_one_train.py \
    -opt options/cdd_experiments/finetune_multilabel.yml \
    --launcher pytorch \
    "${SCALE_OVERRIDES[@]}" \
    --force_yml "name=${SANITY_FINETUNE_C}" \
                "path:pretrain_network_g=${EXP_PRETRAIN}/models/net_g_latest.pth" \
                "train:total_iter=${ITERS_RUN}" \
                "logger:print_freq=1" \
                "logger:save_checkpoint_freq=${SAVE_FREQ}" \
                "val:val_freq=999999"

PHASE3_END=$(date +%s)
echo ""
echo "  Phase 3 training took $((PHASE3_END - PHASE3_START))s"

section "Verifying Phase 3 outputs"
EXP_FINETUNE_C="experiments/${SANITY_FINETUNE_C}"
check_file "net_g checkpoint @iter${SAVE_FREQ}"    "${EXP_FINETUNE_C}/models/net_g_${SAVE_FREQ}.pth"
check_file "Training state @iter${SAVE_FREQ}"      "${EXP_FINETUNE_C}/training_states/${SAVE_FREQ}.state"
check_file "net_g checkpoint @iter${ITERS_RUN}"    "${EXP_FINETUNE_C}/models/net_g_${ITERS_RUN}.pth"
check_file "Training state @iter${ITERS_RUN}"      "${EXP_FINETUNE_C}/training_states/${ITERS_RUN}.state"
check_file "net_g final (latest)"                  "${EXP_FINETUNE_C}/models/net_g_latest.pth"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4/6 — Finetune Row C: Resume
# ══════════════════════════════════════════════════════════════════════════════
banner "PHASE 4/6: Finetune Row C — Resume (iter ${ITERS_RUN} → ${ITERS_RESUME})"
echo "  Using     : --auto_resume"
echo ""

PHASE4_START=$(date +%s)

torchrun_cmd --master-port "${MASTER_PORT}" --nproc_per_node=1 \
    basicsr/all_in_one_train.py \
    -opt options/cdd_experiments/finetune_multilabel.yml \
    --launcher pytorch --auto_resume \
    "${SCALE_OVERRIDES[@]}" \
    --force_yml "name=${SANITY_FINETUNE_C}" \
                "path:pretrain_network_g=${EXP_PRETRAIN}/models/net_g_latest.pth" \
                "train:total_iter=${ITERS_RESUME}" \
                "logger:print_freq=1" \
                "logger:save_checkpoint_freq=${SAVE_FREQ}" \
                "val:val_freq=999999"

PHASE4_END=$(date +%s)
echo ""
echo "  Phase 4 resume took $((PHASE4_END - PHASE4_START))s"

section "Verifying Phase 4 outputs"
check_file "Resumed net_g @iter${ITERS_RESUME}"    "${EXP_FINETUNE_C}/models/net_g_${ITERS_RESUME}.pth"
check_file "Resumed state @iter${ITERS_RESUME}"    "${EXP_FINETUNE_C}/training_states/${ITERS_RESUME}.state"
check_file "Updated net_g (latest)"                "${EXP_FINETUNE_C}/models/net_g_latest.pth"

section "Listing all finetune Row C checkpoints"
echo "    Models:"
ls -lh "${EXP_FINETUNE_C}/models/"     2>/dev/null | grep -v '^total' | awk '{print "      "$0}'
echo "    Training states:"
ls -lh "${EXP_FINETUNE_C}/training_states/" 2>/dev/null | grep -v '^total' | awk '{print "      "$0}'


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5/6 — Finetune Row D: Initial Run
# ══════════════════════════════════════════════════════════════════════════════
banner "PHASE 5/6: Finetune Row D — Initial Run (${ITERS_RUN} iters)"
echo "  Model     : PromptSRModel (net_g trainable + net_dc frozen for inference)"
echo "  Weights   : net_g from ${EXP_PRETRAIN}/models/net_g_latest.pth"
echo "            : net_dc from ${EXP_PRETRAIN}/models/net_dc_latest.pth (frozen)"
echo "  Config    : options/cdd_experiments/finetune_prompt.yml"
echo ""

PHASE5_START=$(date +%s)

torchrun_cmd --master-port "${MASTER_PORT}" --nproc_per_node=1 \
    basicsr/all_in_one_train.py \
    -opt options/cdd_experiments/finetune_prompt.yml \
    --launcher pytorch \
    "${SCALE_OVERRIDES[@]}" \
    --force_yml "name=${SANITY_FINETUNE_D}" \
                "path:pretrain_network_g=${EXP_PRETRAIN}/models/net_g_latest.pth" \
                "path:pretrain_network_dc=${EXP_PRETRAIN}/models/net_dc_latest.pth" \
                "train:total_iter=${ITERS_RUN}" \
                "logger:print_freq=1" \
                "logger:save_checkpoint_freq=${SAVE_FREQ}" \
                "val:val_freq=999999"

PHASE5_END=$(date +%s)
echo ""
echo "  Phase 5 training took $((PHASE5_END - PHASE5_START))s"

section "Verifying Phase 5 outputs"
EXP_FINETUNE_D="experiments/${SANITY_FINETUNE_D}"
check_file "net_g checkpoint @iter${SAVE_FREQ}"    "${EXP_FINETUNE_D}/models/net_g_${SAVE_FREQ}.pth"
check_file "Training state @iter${SAVE_FREQ}"      "${EXP_FINETUNE_D}/training_states/${SAVE_FREQ}.state"
check_file "net_g checkpoint @iter${ITERS_RUN}"    "${EXP_FINETUNE_D}/models/net_g_${ITERS_RUN}.pth"
check_file "Training state @iter${ITERS_RUN}"      "${EXP_FINETUNE_D}/training_states/${ITERS_RUN}.state"
check_file "net_g final (latest)"                  "${EXP_FINETUNE_D}/models/net_g_latest.pth"


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 6/6 — Finetune Row D: Resume
# ══════════════════════════════════════════════════════════════════════════════
banner "PHASE 6/6: Finetune Row D — Resume (iter ${ITERS_RUN} → ${ITERS_RESUME})"
echo "  Using     : --auto_resume"
echo "  Flags     : classify=false, resume_remove_dc=true, strict_load_g=false"
echo "  (Same flags as real row_d_finetune_resume to test identical resume path)"
echo ""

PHASE6_START=$(date +%s)

torchrun_cmd --master-port "${MASTER_PORT}" --nproc_per_node=1 \
    basicsr/all_in_one_train.py \
    -opt options/cdd_experiments/finetune_prompt.yml \
    --launcher pytorch --auto_resume \
    "${SCALE_OVERRIDES[@]}" \
    --force_yml "name=${SANITY_FINETUNE_D}" \
                "path:pretrain_network_g=${EXP_PRETRAIN}/models/net_g_latest.pth" \
                "path:pretrain_network_dc=${EXP_PRETRAIN}/models/net_dc_latest.pth" \
                "classify=false" \
                "resume_remove_dc=true" \
                "path:strict_load_g=false" \
                "train:total_iter=${ITERS_RESUME}" \
                "logger:print_freq=1" \
                "logger:save_checkpoint_freq=${SAVE_FREQ}" \
                "val:val_freq=999999"

PHASE6_END=$(date +%s)
echo ""
echo "  Phase 6 resume took $((PHASE6_END - PHASE6_START))s"

section "Verifying Phase 6 outputs"
check_file "Resumed net_g @iter${ITERS_RESUME}"    "${EXP_FINETUNE_D}/models/net_g_${ITERS_RESUME}.pth"
check_file "Resumed state @iter${ITERS_RESUME}"    "${EXP_FINETUNE_D}/training_states/${ITERS_RESUME}.state"
check_file "Updated net_g (latest)"                "${EXP_FINETUNE_D}/models/net_g_latest.pth"

section "Listing all finetune Row D checkpoints"
echo "    Models:"
ls -lh "${EXP_FINETUNE_D}/models/"     2>/dev/null | grep -v '^total' | awk '{print "      "$0}'
echo "    Training states:"
ls -lh "${EXP_FINETUNE_D}/training_states/" 2>/dev/null | grep -v '^total' | awk '{print "      "$0}'


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
TOTAL_TIME=$(( $(date +%s) - PHASE1_START ))

banner "SANITY TEST COMPLETE"

echo ""
echo "  ┌──────────────────────────────────────────┐"
echo "  │  Total checks  :  ${TOTAL_CHECKS}                       │"
echo "  │  Passed        :  ${PASSED_CHECKS}                       │"
echo "  │  Failed        :  ${FAILED_CHECKS}                       │"
echo "  │  Total time    :  ${TOTAL_TIME}s                      │"
echo "  └──────────────────────────────────────────┘"
echo ""

echo "  Phase timings:"
echo "    Phase 1 (Pretrain RC run)      : $((PHASE1_END - PHASE1_START))s"
echo "    Phase 2 (Pretrain RC resume)   : $((PHASE2_END - PHASE2_START))s"
echo "    Phase 3 (Finetune RC run)      : $((PHASE3_END - PHASE3_START))s"
echo "    Phase 4 (Finetune RC resume)   : $((PHASE4_END - PHASE4_START))s"
echo "    Phase 5 (Finetune RD run)      : $((PHASE5_END - PHASE5_START))s"
echo "    Phase 6 (Finetune RD resume)   : $((PHASE6_END - PHASE6_START))s"
echo ""

if [[ ${FAILED_CHECKS} -eq 0 ]]; then
    echo "  ============================================"
    echo "  =  ALL CHECKS PASSED                       ="
    echo "  =  Pipeline is ready for full experiments!  ="
    echo "  ============================================"
    echo ""
    echo "  You can now safely submit your real experiments:"
    echo "    sbatch slurm/pretrain_rowC.sh"
    echo "    # (then after pretrain finishes:)"
    echo "    bash slurm/submit_finetune.sh C"
    echo "    bash slurm/submit_finetune.sh D"
else
    echo "  ╔══════════════════════════════════════════╗"
    echo "  ║  SOME CHECKS FAILED — review logs above  ║"
    echo "  ╚══════════════════════════════════════════╝"
fi

echo ""
echo "  Sanity experiment directories (safe to delete):"
echo "    rm -rf experiments/SANITY_*"
echo ""
echo "  Finished at: $(date)"

# Exit with failure if any checks failed
[[ ${FAILED_CHECKS} -eq 0 ]]
