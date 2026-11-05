# Ablations: implementation and run guide

Everything for the ablation study is in this folder. Existing files under `basicsr/`, `scripts/`,
`slurm/` and `options/` are only imported or called, never changed. All ablations use NAFNet on
CDD-11. Each Stage 3 ablation is Plan B with **one** change, trained for **200k updates on Plan B's
750k cosine curve**. It is therefore directly comparable with Plan B's own `net_g_200000.pth` and
DCPT's `net_g_200000.pth`, with no extra reference runs.

## 1. Final list (duplicates merged)

How the requested list was merged:
- **F1 ⊂ F2.** "No skip-FiLM" is the middle+decoder arm of the placement ablation, so F2 = {F1 run, middle-only run, Plan B}.
- **F3 "concatenation".** Concatenating the condition to a feature map and applying a 1×1 conv equals adding a learned shift, so the concatenation arm is implemented as `injection: shift`.
- **C3, C4, S3, K6, X1** need no training. They are test-time probes run in one evaluation job of the trained model.
- **S9 and S2** run in one probe job.
- **T5 and S9** appeared twice in the request; each is done once.
- **K3** (both off) is kept, but only worth reading if K1 and K2 each show a gain.

| # | ID | What changes | What it proves | Kind | Config / script | Needs before it can start |
|---|---|---|---|---|---|---|
| 1 | **REF** | nothing (Plan B @200k, copied from p24cs0203) | reference row | eval | `s3_ref_planb_200k.yml` (architecture only) | copy the checkpoint (§3) |
| 2 | **DCPT** | DCPT @200k (same pretraining data as ours) | baseline row | eval | `MODE=dcpt` | nothing |
| 3 | **C2** | `condition_mode: constant` | gain = per-image information, not FiLM capacity | train 200k | `s3_c2_constant.yml` | nothing |
| 4 | **S1** | type-only condition (severity flattened), train + test | severity helps beyond the type | train 200k | `s3_s1_typeonly.yml` | nothing |
| 5 | **K1** | calibrator off | calibrator helps | train 200k | `s3_k1_nocal.yml` | nothing |
| 6 | **K2** | gate off | gate helps (mainly beyond-range / real) | train 200k | `s3_k2_nogate.yml` | nothing |
| 7 | **K3** | both off | parts add up | train 200k | `s3_k3_nocal_nogate.yml` | nothing (read after K1/K2) |
| 8 | **K4** | KL weight 0 | KL keeps the calibrator near the analyzer | train 200k | `s3_k4_nokl.yml` | nothing |
| 9 | **K5** | null condition fixed uniform | learned fallback helps | train 200k | `s3_k5_fixednull.yml` | nothing |
| 10 | **F1** | FiLM at middle + decoder (no skip-FiLM) | skip-FiLM helps | train 200k | `s3_f1_noskip.yml` | nothing |
| 11 | **F2** | FiLM at middle only | multi-scale conditioning helps | train 200k | `s3_f2_middleonly.yml` | nothing |
| 12 | **F3a** | shift only (= concatenation) | FiLM beats concatenation | train 200k | `s3_f3_shift.yml` | nothing |
| 13 | **F3b** | PromptIR-style prompt blocks (+3.8M params) | FiLM beats prompts | train 200k | `s3_f3_prompt.yml` | nothing |
| 14 | **F4** | FiLM randomly initialized | zero-init keeps Stage 2 knowledge | train 200k | `s3_f4_randinit.yml` | nothing |
| 15 | **S4** | condition read from the restorer's own trainable encoder | the frozen analysis copy is needed | train 200k | `s3_s4_shared.yml` | nothing |
| 16 | **C3** | shuffle / null / severity ±1 bin at test | the restorer uses the condition | probes on REF | `eval.sh PROBES=...` | REF copy + graded set |
| 17 | **C4** | logit noise 0.5 / 1 / 2, with vs without gate | gate makes failures graceful | probes on REF and K2 | `eval.sh` | REF, K2 trained |
| 18 | **S3** | oracle presence (CDD-11) / oracle presence + bin (graded) | cost of analyzer errors | probes on REF (and S1) | `eval.sh` | REF + graded set |
| 19 | **K6** | gate and entropy per image (stored in every eval file) | gate opens less on unfamiliar inputs | read from evals | `compare.py` | REF eval |
| 20 | **X1** | PSNR per severity level, incl. beyond range | gain grows with severity | eval on graded set | `eval.sh` | graded set |
| 21 | **X2** | set one primitive's level by hand (0-3) | controllability; PSNR peaks at the true level | probe on REF | `probe.sh SCRIPT=controllability` | REF + graded set |
| 22 | **S9** | analyzer accuracy / MAE / confusion / ECE | analyzer is accurate and calibrated | probe | `probe.sh SCRIPT=analyzer_probe` | graded set |
| 23 | **S2** | same extractor on fine-tuned Plan B features (50k-200k) | features drift → freeze the analysis path | probe (same job as S9) | `probe.sh SCRIPT=analyzer_probe` | Plan B 50k-200k copies |
| 24 | **X3** | low_haze_snow held out of Stage 2 + Stage 3 | generalizes to an unseen combination | Stage 2 → cache → Stage 3 | `x3_holdout_lhs_stage2.yml` → `s3_x3_holdout_lhs.yml` | nothing |
| 25 | **X3-DCPT** | DCPT with low_haze_snow held out | baseline for X3 | train 200k | `dcpt_x3_holdout_lhs.yml` | nothing |
| 26 | X3-type (optional) | X3 + type-only condition | severity matters on unseen combos | Stage 3 | `s3_x3_holdout_lhs_typeonly.yml` | X3 cache |
| 27 | **S5a** | categorical severity loss (Stage 1+2) | ordinal head is right | Stage 1 → 2 → cache → 3 | `s5_categorical_stage1/2.yml` → `s3_s5_categorical.yml` | nothing |
| 28 | **S5b** | scalar severity head (regression) | ordinal head is right | Stage 1 → 2 → cache → 3 | `s5_scalar_stage1/2.yml` → `s3_s5_scalar.yml` | nothing |
| 29 | **S10** | extractor reads only the middle feature | multi-level analysis helps | Stage 1 → 2 → cache → 3 | `s10_middleonly_stage1/2.yml` → `s3_s10_middleonly.yml` | nothing |
| 30 | **T5** | Stage 1 + replay data rendered from CDD-11 clean **training** scenes | the gain does not need outside data | assets → render → Stage 1 → 2 → cache → 3 | `t5_*.yml`, `t5_assets.sh`, `t5_render.sh` | nothing |

**Independent of the running Plan B 750k job:** every item. No ablation uses that job. Items 1, 16–19, 21 and 23 only
need copies of Plan B checkpoints **that already exist** (50k–200k; Plan B is past 370k).

## 2. Files

| Path | What it is |
|---|---|
| `ablib/networks.py` | `AblationFullNAFNet` (FiLM sites, injection film/shift/prompt, FiLM init, fixed null, shared analysis) and `AblationSeverityExtractor` (feature levels, scalar head) |
| `ablib/datasets.py` | Stage 3 dataset with `condition_transform` / `exclude_settings`; Stage 1/2 dataset with `exclude_settings` |
| `ablib/stage12.py` | Stage 1/2 models with `train.severity_loss: ordinal \| categorical \| regression` |
| `ablib/samplers.py` | Stage 2 sampler that allows deliberately held-out settings (X3) |
| `ablib/conditions.py` | type-only, oracle, severity-shift, noise transforms |
| `ablib/common.py` | analysis copies, config reading, metrics (same PSNR/SSIM code as `scripts/evaluate_cdd11_full.py`) |
| `register.py`, `train.py` | registers the classes; training entry (wraps `basicsr/severity_train.py`) |
| `make_configs.py` → `configs/*.yml` | generates all 30 configs (already generated; re-run only after editing it) |
| `scripts/eval_ablation.py` | CDD-11 + graded evaluation, all test-time probes, per-image results |
| `scripts/build_graded_testset.py` | severity-graded CDD-11 test set (50 test GT images × 11 compositions × 4 levels) |
| `scripts/analyzer_probe.py` | S9 + S2 |
| `scripts/controllability.py` | X2 |
| `scripts/cache_conditions.py` | re-cache logits for ablations with their own Stage 2 |
| `scripts/build_inprotocol_plan.py` | T5 render plan from CDD-11 clean training scenes |
| `scripts/compare.py` | final tables: Δ vs reference, per-image win rate, Wilcoxon p-value |
| `slurm/*.sh` | launchers (`train`, `eval`, `cache`, `probe`, `graded`, `t5_assets`, `t5_render`) |

Checked on CPU with tiny batches and images:
- every Stage 3 config and the Stage 1/2 variants build and train;
- all zero-initialized variants start **exactly** equal to the Stage 2 network (max difference 0);
- S4's own logits equal the analysis copy's at the start;
- the X3 hold-out removes low_haze_snow from Stage 2 and Stage 3;
- the graded set renders with severity rising from level 1 to 4 for every primitive.

## 3. One-time setup (this account)

```bash
cd /csehome/p26cs0003/krish/projects/DL_Project
export OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt          # in every new shell before sbatch
mkdir -p $OUT_ROOT/ablations/reference
```

Copy Plan B checkpoints. First, on **p24cs0203**, give this account read access:
```bash
P=/scratch/p24cs0203/krish/sevdcpt/experiments/SevFull_PlanB_NAFNet_Full750k
setfacl -m u:p26cs0003:x /scratch/p24cs0203 /scratch/p24cs0203/krish /scratch/p24cs0203/krish/sevdcpt /scratch/p24cs0203/krish/sevdcpt/experiments $P
setfacl -m u:p26cs0003:rx $P/models
setfacl -m u:p26cs0003:r $P/models/net_g_50000.pth $P/models/net_g_100000.pth $P/models/net_g_150000.pth $P/models/net_g_200000.pth
```
Then, back on **p26cs0003**:
```bash
P=/scratch/p24cs0203/krish/sevdcpt/experiments/SevFull_PlanB_NAFNet_Full750k/models
cp $P/net_g_{50000,100000,150000,200000}.pth $OUT_ROOT/ablations/reference/
```

Shortcuts used below:
```bash
R=$OUT_ROOT/ablations/reference
E=$OUT_ROOT/ablations/experiments
G=$OUT_ROOT/ablations/graded_cdd11/manifest.jsonl
D=/scratch/p26cs0003/krish/experiments/DCPT_NAFNet_CDD11_Finetune_750k/models
```

## 4. Commands

The QoS allows 2 running and 3 submitted jobs per account. Submit in this order; a job queued with `--dependency` counts as submitted.

### Wave 1: no training, start now
```bash
sbatch --job-name=abl_graded ablations/slurm/graded.sh                                   # graded set (~1 h)
# after it finishes:
sbatch --job-name=ev_ref --export=ALL,CONFIG=s3_ref_planb_200k.yml,CKPT=$R/net_g_200000.pth,TAG=ref_planb_200k,PROBES="base shuffle null sev_up sev_down noise0.5 noise1.0 noise2.0 oracle_type oracle_full" ablations/slurm/eval.sh
sbatch --job-name=ev_dcpt --export=ALL,MODE=dcpt,CKPT=$D/net_g_200000.pth,TAG=dcpt_200k ablations/slurm/eval.sh
sbatch --job-name=abl_s9s2 --export=ALL,SCRIPT=analyzer_probe,CONFIG=s3_ref_planb_200k.yml,ARGS="--graded_manifest $G --restorer_ckpts $R/net_g_50000.pth $R/net_g_100000.pth $R/net_g_150000.pth $R/net_g_200000.pth --out $OUT_ROOT/ablations/probes/analyzer_planb.json" ablations/slurm/probe.sh
sbatch --job-name=abl_x2 --export=ALL,SCRIPT=controllability,CONFIG=s3_ref_planb_200k.yml,ARGS="--ckpt $R/net_g_200000.pth --graded_manifest $G --out $OUT_ROOT/ablations/probes/controllability_planb" ablations/slurm/probe.sh
```

### Wave 2: P0 training (each about 20–25 h on one GPU; A30 is fine)
```bash
for c in s3_c2_constant s3_s1_typeonly s3_k1_nocal s3_k2_nogate s3_f1_noskip; do
  echo sbatch --job-name=abl_${c#s3_} --export=ALL,CONFIG=$c.yml ablations/slurm/train.sh
done   # prints the 5 commands; submit them as QoS slots free up
```

### Wave 3: P1 training
`s3_k3_nocal_nogate`, `s3_k4_nokl`, `s3_k5_fixednull`, `s3_f2_middleonly`, `s3_f3_shift`, `s3_f3_prompt`,
`s3_f4_randinit`, `s3_s4_shared`, `dcpt_x3_holdout_lhs`: same command with `CONFIG=<name>.yml`.

### Chains (one command each; later steps wait for earlier ones)
```bash
# X3: hold-out Stage 2 -> cache -> Stage 3
J=$(sbatch --parsable --job-name=abl_x3_s2 --export=ALL,CONFIG=x3_holdout_lhs_stage2.yml ablations/slurm/train.sh)
J=$(sbatch --parsable --dependency=afterok:$J --job-name=abl_x3_cache --export=ALL,CONFIG=s3_x3_holdout_lhs.yml ablations/slurm/cache.sh)
sbatch --dependency=afterok:$J --job-name=abl_x3_s3 --export=ALL,CONFIG=s3_x3_holdout_lhs.yml ablations/slurm/train.sh

# S5a / S5b / S10: Stage 1 -> Stage 2 -> cache -> Stage 3   (K = s5_categorical | s5_scalar | s10_middleonly)
K=s5_categorical
J=$(sbatch --parsable --job-name=abl_${K}_s1 --export=ALL,CONFIG=${K}_stage1.yml ablations/slurm/train.sh)
J=$(sbatch --parsable --dependency=afterok:$J --job-name=abl_${K}_s2 --export=ALL,CONFIG=${K}_stage2.yml ablations/slurm/train.sh)
J=$(sbatch --parsable --dependency=afterok:$J --job-name=abl_${K}_cache --export=ALL,CONFIG=s3_${K}.yml ablations/slurm/cache.sh)
sbatch --dependency=afterok:$J --job-name=abl_${K}_s3 --export=ALL,CONFIG=s3_${K}.yml ablations/slurm/train.sh

# T5: assets -> render -> then the same 4-step chain with K=t5_inprotocol
J=$(sbatch --parsable --job-name=abl_t5_assets ablations/slurm/t5_assets.sh)
sbatch --dependency=afterok:$J --job-name=abl_t5_render ablations/slurm/t5_render.sh
```
A chain of 4 jobs uses all 3 submitted slots plus one more. Submit the first 3 steps, then the last when a slot frees.
If a Stage 1 at micro-batch 16 runs out of memory on an A30, pin it to cn07 (`--nodelist=cn07`); the result does not change.

### Evaluate every trained ablation (after its training ends)
```bash
# Stage 3 ablations:  TAG = short name, CONFIG = the same template used for training
sbatch --job-name=ev_c2 --export=ALL,CONFIG=s3_c2_constant.yml,CKPT=$E/Abl_C2_Constant/models/net_g_200000.pth,TAG=c2_constant ablations/slurm/eval.sh
# C4 needs the noise probes on K2 too:
sbatch --job-name=ev_k2 --export=ALL,CONFIG=s3_k2_nogate.yml,CKPT=$E/Abl_K2_NoGate/models/net_g_200000.pth,TAG=k2_nogate,PROBES="base noise0.5 noise1.0 noise2.0" ablations/slurm/eval.sh
# S3 on the type-only model (true presence, flat severity):
sbatch --job-name=ev_s1 --export=ALL,CONFIG=s3_s1_typeonly.yml,CKPT=$E/Abl_S1_TypeOnly/models/net_g_200000.pth,TAG=s1_typeonly,PROBES="base oracle_type" ablations/slurm/eval.sh
# DCPT hold-out:
sbatch --job-name=ev_dx3 --export=ALL,MODE=dcpt,CKPT=$E/DCPT_X3_Holdout_LHS/models/net_g_200000.pth,TAG=dcpt_x3_holdout ablations/slurm/eval.sh
```
Experiment folder names are the `name:` in each config, e.g. `Abl_S5_Categorical`, `Abl_X3_Holdout_LHS`, `Abl_T5_InProtocol`.

### Tables (login node, no GPU)
```bash
LD_LIBRARY_PATH=/opt/ohpc/apps/python/3.10.pytorch/lib ~/krish/envs/dl_project/bin/python ablations/scripts/compare.py \
  --eval_root $OUT_ROOT/ablations/eval --ref ref_planb_200k --probe_tags ref_planb_200k k2_nogate s1_typeonly \
  --out $OUT_ROOT/ablations/tables/ablation_table.md
```

## 5. Reading the results

| Ablation | Result that supports the method |
|---|---|
| C2, S1 | clearly below REF (S1 shows how much severity adds beyond the type) |
| K1/K2/K4/K5, F1/F2/F3/F4 | below REF, at least on graded beyond-range or on triples; a part with no gain anywhere should be dropped |
| C3 | shuffle/null drop PSNR; `sev_up`/`sev_down` drop it too (wrong severity hurts) |
| C4 | REF degrades less than K2 as noise grows |
| S3 | oracle gains are small → the analyzer is not the bottleneck |
| K6 | gate lower (or entropy higher) on graded L4 than in range |
| X1 | the gap REF − DCPT grows from L1 to L4 |
| X2 | each row of the PSNR matrix peaks at (or next to) the true level |
| S9 / S2 | high accuracy and low ECE for the frozen copy; accuracy falls and feature similarity drops for fine-tuned Plan B checkpoints |
| S4 | below REF |
| X3 | on low_haze_snow, Ours-holdout loses less vs Ours-full than DCPT-holdout vs DCPT-full |
| S5, S10 | below REF |
| T5 | close to REF (the gain does not depend on LSDIR) |

## 6. Running on another account

`git pull` there, then use the same commands with that account's `OUT_ROOT`. Shared data and Stage 1/2
checkpoints come from `/scratch/p26cs0003/krish` (`SHARED_ROOT`, ACL as in NEW_ACCOUNT_SETUP.md). To reuse
this account's graded set, add `GRADED=/scratch/p26cs0003/krish/sevdcpt/ablations/graded_cdd11/manifest.jsonl`
to `--export` (that folder needs the same read ACL).
