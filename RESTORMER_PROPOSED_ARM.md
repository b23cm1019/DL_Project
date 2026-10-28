# Proposed-Restormer arm

This is an independent backbone ablation of the proposed severity-aware
method.  It does **not** replace the paper-faithful DCPT-NAFNet baseline or
the completed/running proposed-NAFNet experiments.

## What remains invariant

- The finalized 80k/15k synthetic corpus and CDD-11 manifests.
- Stage 1 clean-to-clean identity, ordinal severity loss, balanced type loss,
  constant learning rates, and 100k global updates.
- Stage 2 CDD type adaptation plus synthetic severity replay: exactly eight
  CDD and eight synthetic patches per global update for 50k updates.
- Stage 3 CDD LQ-to-GT L1 restoration for 750k global updates, with the
  feature extractor and feature-tap analysis path frozen.
- Learning rates, schedules, augmentation, validation, and checkpoint
  cadences of the proposed-NAFNet arm.

Only the restoration backbone is changed: NAFNet becomes the standard
Restormer configuration used by Restormer experts in the unmodified official
UniRestorer clone: `dim=48`, blocks `[4,6,6,8]`, four refinement blocks,
heads `[1,2,4,8]`, and `WithBias` layer normalization.

## Backbone adapter contract

The method is portable to a restoration backbone only when it provides:

1. an unconditioned encoder/latent feature tap for the extractor;
2. an unconditioned restoration path for Stage 1/2 identity training;
3. a post-feature-tap conditioning location for Stage 3; and
4. a meaningful analysis/restoration parameter partition.

`FeatureTapRestormer` and `FrozenFeatureConditionedRestormer` implement that
contract.  The extractor sees Restormer features `[48,96,192]` plus latent
`384`.  Stage-3 FiLM modules are zero initialized, so loading the Stage-2
backbone initially reproduces its unconditioned output exactly.

This makes the approach backbone-agnostic in principle, but not a YAML-only
plug-in: each new backbone needs an adapter satisfying this contract and
shape/freeze/memory tests.

## Two-GPU policy

Two 24-GB GPUs are DDP replicas, not a pooled 48-GB memory space.  The
configs preserve the original global batches as follows:

| Stage | Global batch | DDP ranks | Batch per rank |
|---|---:|---:|---:|
| Stage 1 | 32 | 2 | 16 |
| Stage 2 | 16 = 8 CDD + 8 synthetic | 2 | 8 = 4 CDD + 4 synthetic |
| Stage 3 | 32 | 2 | 16 |

The DDP fixed-mix sampler shards complete batches, so every rank maintains
the required four CDD/four synthetic Stage-2 mixture.

Request one node with two comparable GPUs each having at least 24 GB VRAM,
20 CPU cores, and 64 GB host RAM.  Do not use mixed GPU models for this run.

## Required launch sequence

First run the mandatory 45-minute B16-per-rank memory probe on the selected
two-GPU node:

```bash
sbatch slurm/smoke_severity_restormer_2gpu.sh
```

The log must end with both:

```text
SEVERITY_RESTORMER_CPU_SMOKE_OK
RESTORMER_STAGE1_DDP_B16_PROBE_OK max_allocated_gib=... max_reserved_gib=...
```

Proceed only if it exits zero and `max_reserved_gib` leaves at least roughly
2 GiB below the physical per-GPU capacity.  Then submit Stage 1:

```bash
sbatch slurm/severity_restormer_stage1_2gpu.sh
```

After successful Stage 1 and Stage 2 respectively, submit:

```bash
sbatch slurm/severity_restormer_stage2_2gpu.sh
sbatch slurm/severity_restormer_stage3_2gpu.sh
```

Use `SEVERITY_AUTO_RESUME=1` for a manual Stage-1/2 recovery submission;
Stage 3 always auto-resumes from its latest 50k state.

## Interpretation boundary

Proposed-Restormer controls the backbone family when comparing with
UniRestorer.  It is not a reproduction of UniRestorer: UniRestorer uses a
multi-granularity routed mixture of heterogeneous experts.  Before making a
numerical claim against its reported results, align CDD-11 split, preprocessing
and evaluation protocol in addition to using the same dataset source.
