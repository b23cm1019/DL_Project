# Severity-Aware DCPT — Stage 3 Redesign: Proposal and Progress

*Status as of 31 Oct 2026, 11:20 IST.*

## 1. Summary

The old Stage 3 trained more slowly than the paper-faithful DCPT-NAFNet reproduction. The
main reason: it froze **91.5 % of NAFNet** (62.6 M of 68.4 M parameters, including the
28-block encoder), so only the 5.8 M-parameter decoder could learn restoration. DCPT fine-tunes
all 68.4 M.

The redesign keeps Stage 1 and Stage 2 unchanged and splits Stage 3 into two phases:

- **3a, head alignment:** frozen trunk; trains decoder and FiLM.
- **3b, full fine-tuning:** a frozen *analysis copy* (Stage-2 encoder + extractor) produces the
  severity condition, and a separate, fully trainable *restoration copy* does the restoration.

Two variants:

- **Plan A, running now:** reuses the old Stage 3 run (350 k iterations) as Stage 3a, then runs
  Stage 3b for 400 k iterations. Total 750 k updates, the same as DCPT.
- **Plan B, final proposed pipeline:** the same design with a short Stage 3a and a long Stage 3b.
  It is the version to describe in the report. Plan A is how it is executed before the deadline.

## 2. Why the old Stage 3 fell behind DCPT (verified)

| # | Gap | Evidence |
|---|---|---|
| G1 | **Encoder frozen.** 62.6 M of 68.4 M parameters frozen; the encoder was trained only for identity + classification, never for restoration. **Main cause.** | parameter count; `FrozenFeatureConditionedRestorationModel` |
| G2 | No grad clip (DCPT 0.01) and no EMA (DCPT 0.999). | the two YAMLs |
| G3 | Trained on 10,406 of 13,013 CDD-11 images (2,607 held out as val). | manifests |
| G4 | No PSNR / SSIM logged by either run, only L1. | logs |
| G5 | **Train/test condition mismatch.** Condition computed on 128×128 crops in training, on full 720×1080 images at test. Measured on 500 crops: severity distribution differs by 0.22 (L1 per primitive), and type decision flips in 1.75 % of cases. DCPT has no such mismatch (its classifier is discarded after pretraining). | `condition_report.json` |
| G6 | **No TLC.** DCPT's `NAFNet` and our Stage 1/2 use test-time local pooling (`Local_Base`); the old Stage 3 arch did not, so full-image inference used different pooling from training. | `nafnet_arch.py` vs `frozen_feature_conditioned_nafnet_arch.py` |

Stage 1 and Stage 2 are healthy and need no change: type micro-F1 0.999, severity accuracy
0.93, identity L1 ≈ 1e-4.

## 3. What we keep from the previous proposed approach

| Component | Status |
|---|---|
| NAFNet backbone (width 64, enc [1,1,1,28], dec [1,1,1,1]) | kept |
| Stage 1: synthetic severity-aware DCPT pretraining (identity + ordinal severity + type) | **unchanged, not retrained** |
| Stage 2: CDD-11 type adaptation with synthetic severity replay | **unchanged, not retrained** |
| `FeatureSeverityExtractor`: 4 primitives × 4 ordinal bins (16-D **p**) read from encoder/middle features | kept; frozen forever |
| Condition projector (16 → 128 → 128) and FiLM on the middle block and decoder | kept; weights carried over from the old run |
| Old Stage 3 run (`net_g_350000.pth`) | reused as **Stage 3a** |
| Training protocol: AdamW, batch 32, 128×128 crops, cosine LR | kept, matched to DCPT |

## 4. Proposed changes (Stage 3b) and formulas

Notation: frozen extractor logits `z ∈ R^{4×4}`; `q_t = softmax(z_t)`; decoder feature `F_l`;
encoder skip `S_l`; output `ŷ`; ground truth `y`.

| # | Change | Formula / setting | Why |
|---|---|---|---|
| C1 | **Decoupled trunks:** frozen analysis copy + fully trainable restoration copy (all 68.7 M params train) | — | Fixes G1. The extractor always reads the frozen copy, so it cannot forget (ControlNet ICCV'23; DCPT ICLR'25) |
| C2 | **Full-image condition, cached once** | logits from the full image for all 13,013 train / 2,200 test images; every crop uses its image's logits | Fixes G5; also removes the frozen trunk from the training loop |
| C3 | **Restoration-aware severity calibrator** (16 params, identity init) | `π_t = 1 − q_{t,0}` (pinned); `s_t = α_t·z_{t,1:3} + b_t`; `p̂_t = [q_{t,0}, π_t·softmax(s_t)]`; `L_KL = Σ_t KL(p̂_t ‖ q_t)` | CDD-11 has no severity labels. Lets restoration loss re-align severity while type stays fixed |
| C4 | **Learned null condition** | `p_null = softmax(null_logits)`, uniform at init | Fallback for the gate, dropout and the control run |
| C5 | **Learned confidence gate** (blend form) | `g = 1 − σ(λ)·mean_t H_t(p̂)`; `e = g·e(p̂) + (1−g)·e(p_null)`; `σ(λ)` starts at 0.0025 | Trust the condition only as much as the extractor is confident (UniRestorer ICLR'26, MoFME AAAI'24) |
| C6 | **Condition dropout** | 10 % of samples use `e(p_null)`, ramped in over 20 k iterations | Robust to wrong **p** (OneRestore ECCV'24) |
| C7 | **Skip-connection FiLM** (zero init) | `S_l ← S_l ⊙ (1+γ^s_l(e)) + β^s_l(e)` | Rain/snow streaks live in the high-resolution skips (SFT CVPR'18, AdaLN-Zero ICCV'23) |
| C8 | **FFT amplitude loss** | `L = L1 + 1.0·‖ |FFT(ŷ)| − |FFT(y)| ‖₁ + 0.01·L_KL` (ortho FFT) | Haze/low light are global low-frequency errors; streaks show as spectral peaks (MIMO-UNet ICCV'21, MoCE-IR CVPR'25) |
| C9 | **DCPT training parity** | grad clip 0.01, EMA 0.999, all 13,013 images, linear LR warmup 5 k | Fixes G2, G3 |
| C10 | **TLC** in the restoration network | `Local_Base` conversion, as in DCPT's `NAFNet` | Fixes G6 |
| C11 | **Per-category PSNR/SSIM validation** | every 25 k on 20 test images × 11 categories (EMA weights) | Fixes G4 |

### Design adjustments found during testing

The old Stage 3a FiLM learned scale ≈ −0.95, which shrinks the frozen encoder features about 20×.
Removing FiLM sends L1 from 0.05 to 242. Hence:

- The gate and dropout **blend toward a null condition** instead of switching FiLM off.
- The control run uses a **constant** condition, not "no FiLM".
- The condition LR is ×1, not ×5: the ×5 rule is for freshly initialized FiLM.
- The FFT weight is 1.0, not 0.05: with ortho normalization, 0.05 was about 0.5 % of the loss.

## 5. Plan A — continue from the current checkpoint (running)

| Phase | What trains | Iterations | Notes |
|---|---|---|---|
| Stage 1 | NAFNet + extractor | 100 k (done) | unchanged |
| Stage 2 | encoder + extractor | 50 k (done) | unchanged |
| Stage 3a | decoder + FiLM (trunk frozen) | 350 k (done, old run) | crop-based **p**, no clip/EMA, 10,406 images |
| **Stage 3b** | **everything** | **400 k** (running) | C1–C11; warm start from `net_g_350000.pth` |

The total is 750 k updates, equal to DCPT. The LR is 2e-4 with cosine decay to 1e-6, close to
where DCPT's own cosine is at 350 k.

## 6. Plan B — final proposed full pipeline (for the report)

1. **Stage 1:** synthetic severity-aware DCPT pretraining (unchanged).
2. **Stage 2:** CDD-11 type adaptation with synthetic severity replay (unchanged). Optional
   post-hoc temperature scaling. The fitted temperatures (0.64–0.88) show the extractor is slightly
   under-confident, so this is not needed.
3. **Stage 3a:** short head alignment (≈ 30–50 k iterations) with full-image cached **p**.
4. **Stage 3b:** full fine-tuning with C1–C11 for the rest of the 750 k budget.
5. **Inference:** analysis copy → **p** → calibrator → restorer.

Plan A differs from Plan B only in its Stage 3a: 350 k iterations instead of about 50 k,
crop-based **p**, no clip/EMA, and fewer images. Measured evidence of the cost:

- At ≈ 55 k full fine-tuning iterations, Stage 3b's training L1 (0.034) equals DCPT's at
  ≈ 140 k full fine-tuning iterations.
- So per full-network iteration, Stage 3b learns faster than DCPT.
- But the 350 k frozen iterations were worth only ≈ 85 k full-network iterations.

## 7. Implementation status

All new code is in separate files; no existing file was modified.

| File | Purpose | Status |
|---|---|---|
| `basicsr/archs/severity_conditioned_full_nafnet_arch.py` | Stage 3b network (C3–C7, C10) | done, tested |
| `basicsr/models/severity_full_finetune_model.py` | Training/validation (C8, C9, C11), param groups, save `params` + `params_ema` | done, tested |
| `basicsr/data/severity_conditioned_paired_dataset.py` | Crops + cached full-image logits | done, tested |
| `basicsr/utils/severity_full_analysis.py` | Frozen analysis copy (exact Stage 2 path: TLC NAFNet, unpadded full image) | done, tested |
| `scripts/cache_severity_full_conditions.py` | Logit cache, crop-gap diagnostic, temperature fit | done; **ran** (job 46030) |
| `scripts/evaluate_cdd11_full.py` | One evaluator for DCPT / old Stage 3 / Stage 3b, with shuffled / no-calibrator / no-gate / constant-condition flags | done, tested |
| `options/cdd_experiments/severity_full_stage3b.yml` | Main run | **running** |
| `options/cdd_experiments/severity_full_stage3b_nocond.yml` | Constant-condition control | ready, not launched |
| `slurm/cache_severity_full_conditions.sh`, `slurm/severity_full_stage3b.sh`, `slurm/severity_full_stage3b_nocond.sh`, `slurm/evaluate_cdd11_full.sh` | Launchers | ready |

**Smoke tests passed:**

- The 350 k checkpoint loads (only new modules missing), and the output equals the old Stage 3 to 3.6e-7.
- The calibrator is an exact identity at init, and presence probability is unchanged after updates.
- Training steps, validation and saving work for both configs. The evaluator works in all modes.
- On real images, shuffled **p** raises L1 from 0.047 to 0.18, so the model really uses the condition.

## 8. Current progress

| Job | Node / GPU | Progress |
|---|---|---|
| DCPT reproduction (45342) | cn07, A6000 | ≈ 355 k / 750 k |
| Stage 3b (46058) | **cn05, A30** (moved from cn07) | resumed at 50 k / 400 k |

The move: on cn07, another user's training process shared each of our GPUs, which halved our
speed (0.42 → 0.9 s/iter). Our process uses 17.7 GB, so it fits on the 23.5 GB A30. The job
resumed from the 50 k state with weights, EMA, optimizer and scheduler restored.

**Training L1 at the same total number of updates** (Stage 3b iteration *i* = 350 k + *i*):

| Total updates | DCPT | Old Stage 3 (frozen) | Stage 3b |
|---|---|---|---|
| 350 k | 0.0254 | ≈ 0.0365 (plateau) | 0.0383 (start) |
| 375 k | — | — | 0.0365 |
| 400 k | — | — | 0.0343 |

**Stage 3b validation** (EMA, 220 test images):

| Iteration | Avg PSNR / SSIM | L+H+R | L+H+S |
|---|---|---|---|
| 25 k | 26.46 / 0.856 | 23.23 / 0.758 | 23.63 / 0.769 |
| 50 k | 26.59 / 0.859 | 23.28 / 0.762 | 23.71 / 0.773 |

**Stage 3b vs the DCPT reproduction** on the same 220 test images (same evaluator, RGB, crop 0):

| Category | DCPT @ 350 k | Stage 3b @ 50 k (400 k total) | Δ |
|---|---|---|---|
| low | 25.78 / 0.826 | 24.71 / 0.812 | −1.07 |
| haze | 30.76 / 0.988 | 29.84 / 0.988 | −0.92 |
| rain | 32.19 / 0.961 | 31.64 / 0.945 | −0.55 |
| snow | 31.87 / 0.971 | 31.41 / 0.965 | −0.46 |
| low_haze | 25.17 / 0.803 | 24.20 / 0.788 | −0.97 |
| low_rain | 25.16 / 0.792 | 24.59 / 0.769 | −0.57 |
| low_snow | 24.45 / 0.771 | 23.84 / 0.750 | −0.61 |
| haze_rain | 27.66 / 0.957 | 26.81 / 0.939 | −0.85 |
| haze_snow | 29.46 / 0.967 | 28.49 / 0.955 | −0.97 |
| **L+H+R** | **24.32 / 0.787** | 23.28 / 0.762 | −1.04 |
| **L+H+S** | **24.22 / 0.795** | 23.71 / 0.773 | −0.51 |
| **11-cat avg** | **27.37 / 0.874** | 26.59 / 0.859 | **−0.78** |

Stage 3b is still **0.78 dB behind** the reproduction (DCPT has had 350 k full-network
iterations vs Stage 3b's 50 k). The gap is largest on low-light categories. The reproduction itself
is about 1.9–2.2 dB below the DCPT paper on the triples (24.32 vs 26.23, 24.22 vs 26.40), so the
paper numbers are not a realistic target for this setup. The fair target is the reproduction.

## 9. Experiments

**Running now**

1. **Stage 3b main run** (Plan A): validation PSNR/SSIM every 25 k.
2. **DCPT reproduction** to 750 k (untouched).

**Next, evaluation only (no new training):**

3. DCPT checkpoints (350 k now, 750 k at the end) on the same 220 images, then on the full 2,200-image test set.
4. Old Stage 3 (`net_g_350000.pth`, frozen encoder) on the same images, as the "before" row.

**Ablations after the main run** (evaluator flags on the final Stage 3b checkpoint, no retraining):

5. Shuffled **p** (does the model use the condition?).
6. No calibrator.
7. No gate.

**Needs one more training run (only if time and a GPU slot allow):**

8. Constant-condition control (`severity_full_stage3b_nocond.yml`): same architecture, init and
   budget, no per-image information. Isolates the gain from severity conditioning versus unfreezing.

**Final results table:** DCPT paper (26.23 / 26.40 on L+H+R / L+H+S) | DCPT reproduction |
old Stage 3 | Stage 3b | Stage 3b ablations; plus parameters (restorer 68.7 M + analysis 68.8 M)
and per-category CDD-11 PSNR/SSIM.

## 10. Risks and caveats

- The DCPT paper used 4× L40 GPUs and its own pretraining. The fair target is our reproduction under the same evaluator.
- Stage 3b's ETA depends on GPU sharing; a full 400 k may not finish before the deadline. Use the
  best validated checkpoint reached.
- The analysis copy roughly doubles inference parameters. Report this; distilling or sharing the
  analysis trunk is future work.
- Expected gains from the calibrator and gate are small; their value is interpretability and robustness.
