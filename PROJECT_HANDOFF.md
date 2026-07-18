# Project Handoff

## Purpose

This document is a quick handoff guide for any new teammate joining the project.

It explains:

- what this project is about
- what has already been implemented
- what the current code structure looks like
- what results we have so far
- what we recently diagnosed
- what we now believe the main bottlenecks are
- what experiments should be done next

The goal is to help a new contributor start making useful progress quickly.

## Current Branch

- current branch: `claude`

At the moment, the working tree is clean.

## Project Idea

This project studies mixed-degradation image restoration on `CDD-11`, using the `DCPT` idea of degradation-classification-based pretraining and extending it in two main ways:

- `Row C`: replace the original flat 11-class supervision with a `4D multi-label BCE` target over primitive degradations:
  - `low-light`
  - `haze`
  - `rain`
  - `snow`

- `Row D`: reuse the pretrained classifier from `Row C` and inject its predicted degradation vector back into the restoration model as a `soft prompt`

So the core project question has been:

- can better weak supervision and semantic conditioning improve restoration quality on mixed degradations?

## Repo Layout

The most important locations in the repo are:

- [README.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/README.md>)
  Main project summary and reported results.

- [EXPERIMENTAL_SETUP.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/EXPERIMENTAL_SETUP.md>)
  Initial methodology and reduced-compute experiment design.

- [options/cdd_experiments](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/options/cdd_experiments>)
  Main YAML configs for `Row B`, `Row C`, and `Row D`.

- [basicsr/losses/classify_loss.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/basicsr/losses/classify_loss.py>)
  Contains the custom `MultiLabelBCELoss`.

- [basicsr/models/prompt_sr_model.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/basicsr/models/prompt_sr_model.py>)
  Contains `Row D` prompt-guided finetuning model.

- [basicsr/all_in_one_train.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/basicsr/all_in_one_train.py>)
  Main stage-wise training entry point used for these experiments.

- [run_experiments.sh](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/run_experiments.sh>)
  Main runner for pretraining, finetuning, testing, and resume flows.

- [scripts/analyze_classifier.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/scripts/analyze_classifier.py>)
  New diagnosis script added to inspect `Row C` classifier behavior.

- [NEXT_EXPERIMENTAL_PLAN.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/NEXT_EXPERIMENTAL_PLAN.md>)
  Planned next-stage experiments.

- [ROW_C_CLASSIFIER_DIAGNOSIS_REPORT.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/ROW_C_CLASSIFIER_DIAGNOSIS_REPORT.md>)
  Detailed interpretation of the latest classifier diagnosis.

## Current Experimental Rows

### Row B

- baseline using original-style single-label degradation supervision

### Row C

- uses `4D primitive-aware multi-label BCE`
- intended to make degradation supervision more compositional

### Row D

- takes the classifier from `Row C`
- injects classifier predictions back into restoration as a prompt

## Dataset Setup

The main dataset is `CDD-11`.

It contains `11` degradation settings:

- single:
  - `low`
  - `haze`
  - `rain`
  - `snow`

- mixed:
  - `low_haze`
  - `low_rain`
  - `low_snow`
  - `haze_rain`
  - `haze_snow`
  - `low_haze_rain`
  - `low_haze_snow`

### Test Size

From the latest full diagnosis, the test split currently used has:

- `200` images per setting
- `2200` total test images

So:

- single-degradation test images: `800`
- mixed-degradation test images: `1400`

### Important Current Training Detail

Right now, both `Row C` and `Row D` use all `11` settings during:

- pretraining
- finetuning

Also, current finetuning is **not balanced** between single and mixed settings.

Since there are `4` single folders and `7` mixed folders, and all currently use `dataset_enlarge_ratio: 1`, the finetuning stage is biased toward mixed degradations.

## What Has Already Been Built

The following project pieces are already implemented in the codebase:

- multi-label primitive loss for `Row C`
- prompt-guided restoration for `Row D`
- stage-wise training pipeline for pretrain / finetune / test
- resume-ready SLURM and shell runners
- unseen synthetic 4-way degradation evaluation
- plotting and qualitative asset generation
- classifier diagnosis script for `Row C`

## Existing Reported Results

There are two result references in the repo right now:

- the metrics in [README.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/README.md>) are the earlier / initial summary values
- the latest comparison numbers are the newer results and should be treated as the current reference

### Latest Standard CDD-11 Results

The latest `Row C` vs `Row D` results are:

- `Row C`: `28.0424 PSNR`, `0.8806 SSIM`
- `Row D`: `28.0405 PSNR`, `0.8806 SSIM`
- `Paper DCPT-NAFNet`: `30.6591 PSNR`, `0.8912 SSIM`

### Latest Gap Summary

- `Row C vs Paper`: `-2.6167 dB`, `-0.0106 SSIM`
- `Row D vs Paper`: `-2.6186 dB`, `-0.0106 SSIM`
- `Row D vs Row C`: `-0.0019 dB`, approximately no SSIM change

### Latest Experimental Setting Comparison

- `Your Row C (Multilabel BCE)`:
  - pretraining: `100k`
  - finetuning: `500k`
  - test set: `CDD test`
  - evaluated from `net_g_latest.pth`

- `Paper DCPT-NAFNet`:
  - pretraining: `100k`
  - finetuning: `750k`
  - test set: `CDD test`

### Interpretation So Far

- the latest numbers show `Row C` and `Row D` are now almost identical on the standard test average
- `Row D` is no longer giving a meaningful gain over `Row C`
- the remaining gap to paper is still large enough that further work should focus on restoration-side improvement, not just small prompt variations
- the README results should be treated as an earlier project snapshot, not the final current benchmark

## What We Recently Investigated

A key concern was raised:

- if both pretraining and finetuning happen on mixed-degradation images, maybe the classifier starts assuming multiple degradations too often
- if that happens, restoration on single-degradation images may suffer

To test that, we added and ran:

- [scripts/analyze_classifier.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/scripts/analyze_classifier.py>)

This script was run on the full `CDD-11` test split using the current `Row C` pretrained encoder and classifier.

## What The Diagnosis Showed

The most important finding is:

- the `Row C` classifier is actually already very strong

### Full Diagnosis Metrics

- exact multi-hot match accuracy: `98.0%`
- cardinality accuracy: `98.0%`
- micro F1: `0.9945`
- macro F1: `0.9954`

### Single vs Mixed

- single-only exact-match accuracy: `99.5%`
- mixed-only exact-match accuracy: `97.14%`

### Primitive-Level Result

- `low`: weakest primitive
- `haze`: near-perfect
- `rain`: perfect
- `snow`: perfect

### Important Updated Conclusion

This diagnosis means:

- hard classifier prediction is **not** the main bottleneck anymore
- the model does distinguish single vs mixed degradations well
- the remaining classification weakness is mainly around `low-light`

## What This Changed In Our Understanding

Before the diagnosis, we were mainly worried that:

- mixed training made the classifier fundamentally unreliable

After the diagnosis, the better interpretation is:

- `Row C` classifier quality is already high
- the bigger remaining bottlenecks are more likely:
  - restoration feature quality
  - finetuning distribution bias
  - soft prompt leakage in `Row D`

This is especially important because `Row D` uses:

- `sigmoid(cls_output)`

directly as a soft prompt.

So even when thresholded classification is correct, nonzero soft scores can still influence restoration.

The main suspicious case is:

- nonzero `low-light` soft activation on non-low settings

## Current Direction

We are now moving away from the earlier idea that:

- "the classifier is broadly failing"

and toward the stronger idea that:

- "the classifier is already good, but the restoration framework and prompt usage can still be improved"

## Planned Next Experimental Direction

The main future plan is documented in [NEXT_EXPERIMENTAL_PLAN.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/NEXT_EXPERIMENTAL_PLAN.md>).

The intended direction is:

### 1. Single-First Pretraining

Start with only:

- `low`
- `haze`
- `rain`
- `snow`

Goal:

- learn cleaner primitive degradation signatures first

### 2. Mixed Calibration Stage

Then reintroduce all `11` settings.

Goal:

- teach the model how primitive degradations compose in mixed cases

### 3. Balanced Finetuning

Still finetune on all `11` settings, but oversample single degradations more than mixed ones.

Goal:

- reduce bias toward mixed degradation assumptions

### 4. Better Prompt Control

Re-test prompt-guided `Row D` using:

- a better-calibrated classifier
- possibly confidence-gated prompt injection

Goal:

- reduce harmful soft prompt contamination

## Immediate Next Priorities

If a teammate is starting now, the most useful next contributions are:

1. Create new YAML variants for:
   - single-only pretraining
   - mixed calibration
   - balanced finetuning
   - gated prompting

2. Split restoration evaluation into:
   - single-only average
   - mixed-only average
   - overall average
   - unseen 4-way average

3. Investigate whether soft `low-light` prompt leakage correlates with worse restoration results in `Row D`

4. Improve documentation and reproducibility for cluster runs

## Suggested New Config Files

These are the most likely next files to add:

- `pretrain_baseline_single.yml`
- `pretrain_multilabel_single.yml`
- `pretrain_multilabel_mixed_calibration.yml`
- `finetune_multilabel_balanced.yml`
- `finetune_prompt_calibrated.yml`
- `finetune_prompt_calibrated_gated.yml`

## Cluster Workflow Notes

The main experiments are run on the cluster, not locally.

Typical workflow:

1. push local code changes
2. pull on cluster
3. activate env via `slurm/_cluster_env.sh`
4. run experiments through `run_experiments.sh` or SLURM wrappers

The latest classifier diagnosis was run on the cluster and then copied back locally for analysis.

## Recommended Reading Order For A New Teammate

If someone is joining now, they should read in this order:

1. [README.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/README.md>)
2. [NEXT_EXPERIMENTAL_PLAN.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/NEXT_EXPERIMENTAL_PLAN.md>)
3. [ROW_C_CLASSIFIER_DIAGNOSIS_REPORT.md](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/ROW_C_CLASSIFIER_DIAGNOSIS_REPORT.md>)
4. `options/cdd_experiments/*.yml`
5. [basicsr/losses/classify_loss.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/basicsr/losses/classify_loss.py>)
6. [basicsr/models/prompt_sr_model.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/basicsr/models/prompt_sr_model.py>)
7. [scripts/analyze_classifier.py](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/scripts/analyze_classifier.py>)

## Bottom Line

This project already has:

- a working baseline
- a working multi-label extension
- a working prompt-guided extension
- initial results and stress-test evaluation
- a completed classifier diagnosis

The project is no longer in the stage of asking:

- "does the classifier work at all?"

It is now in the stage of asking:

- "how do we turn a good classifier signal into better restoration quality?"

That is the main direction new contributors should work toward.
