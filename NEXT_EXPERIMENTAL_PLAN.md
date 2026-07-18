# Next Experimental Plan

## Objective

Build a cleaner training framework that improves restoration on both:

- single degradations
- mixed degradations

while also helping us understand:

- whether the encoder is learning better degradation-aware features
- whether the classifier is over-predicting mixed degradations
- whether prompt guidance is helping or hurting restoration

## What We Will Change

We will move from the current `all-11-settings-from-the-start` training setup to a staged training strategy:

- first learn from only single degradations
- then calibrate on mixed degradations
- then finetune with a better balance between single and mixed samples
- then re-evaluate prompt guidance using a better-calibrated classifier

## Why We Are Changing It

In the current setup, the model sees mixed degradations too early and too often.

Because of this:

- the classifier may start predicting multiple degradations even for single-degradation images
- the encoder may not learn clean primitive degradation signatures
- the restoration network may become biased toward mixed degradation patterns
- this can limit `PSNR` and `SSIM`, especially on single-degradation cases

## Expected Benefit

This new plan is expected to help us:

- learn cleaner primitive degradation features first
- better distinguish single and mixed degradation images
- improve classifier calibration for multi-hot prediction
- reduce false prompt signals during restoration
- improve restoration quality on both single and mixed degradations

In short, the goal is to make the model:

- understand degradations more clearly
- combine them more reliably
- restore images more accurately

## Current Problem

Right now `Row C` and `Row D` are trained on all `CDD-11` settings during both pretraining and finetuning.

Because of this:

- the model sees mixed degradations very often
- the classifier may start assuming multiple degradations even for single-degradation inputs
- the restoration network may become biased toward mixed patterns
- this can hurt `PSNR` and `SSIM` on single-degradation test sets

## Main Questions We Want To Answer

1. Is mixed-degradation pretraining hurting single-degradation restoration?
2. Does single-first pretraining help the encoder learn cleaner primitive features?
3. Do we need a mixed-calibration stage after single-first pretraining?
4. Is finetuning data balance causing the model to drift back toward mixed assumptions?
5. Does prompt guidance help only when the classifier is well calibrated?

## Stage 0: Diagnose Current Row C

### Change

Add a classifier analysis script that compares:

- predicted 4D degradation vector
- ground-truth 4D vector

for all `CDD-11` settings.

### What We Measure

- per-primitive precision, recall, and F1
- exact multi-hot match accuracy
- degradation-count accuracy
- false positives on single-degradation folders

### What We Learn

- whether the current classifier is over-activating extra degradations
- which primitive is most frequently confused
- whether the current issue is really classifier calibration or something deeper

## Stage 1: Remove Mixed Bias From Pretraining

### Experiment: `B-single`

### Pretraining

- use only `low`, `haze`, `rain`, `snow`
- keep baseline single-label supervision

### Finetuning

- normal finetuning on all `11` settings

### Question Answered

If we remove mixed images from pretraining, does restoration improve even without multi-label learning?

### Possible Finding

If `B-single` beats current `Row B`, then mixed pretraining was harming feature purity.

## Stage 2: Primitive-First Curriculum

### Experiment: `C-curriculum`

### Pretraining Stage A

- use only `low`, `haze`, `rain`, `snow`
- use multi-label BCE

### Pretraining Stage B

- continue pretraining on all `11` settings
- keep multi-label BCE
- use this as a short classifier calibration stage

### Finetuning

- normal finetuning on all `11` settings

### Question Answered

Can we first learn clean primitive signatures and then explicitly teach the model how mixed degradations compose?

### Possible Finding

If `C-curriculum` beats `B-single`, then primitive-aware supervision is giving better encoder features, not just better classifier calibration.

## Stage 3: Fix Finetuning Bias

### Experiment: `C-curriculum-balanced-ft`

### Pretraining

- same as `C-curriculum`

### Finetuning

- still use all `11` settings
- oversample single-degradation folders more than mixed folders

Suggested idea:

- singles: weight `2x` or `3x`
- mixed: weight `1x`

### Question Answered

Is finetuning itself pulling the model back toward mixed-degradation assumptions?

### Possible Finding

If this model improves on single settings without losing too much on mixed settings, then the main problem is finetuning distribution, not just pretraining.

## Stage 4: Re-test Prompt Guidance

### Experiment: `D-calibrated`

### Setup

- take the best classifier from Stage 2 or Stage 3
- freeze it
- use prompt-guided finetuning

### Question Answered

Does prompt guidance help when the classifier is properly calibrated?

### Possible Finding

If `D-calibrated` beats the best `C` variant, then prompting was not the problem. The real issue was classifier quality.

## Stage 5: Safer Prompt Injection

### Experiment: `D-calibrated-gated`

### Setup

- same as `D-calibrated`
- inject prompt only when classifier confidence is above a threshold

### Question Answered

Can we stop wrong prompt signals from hurting restoration on single-degradation images?

### Possible Finding

If this improves singles while keeping mixed performance strong, then prompt noise was a major source of error.

## Evaluation Protocol

For every important row, report results separately for:

- `single-only average`: `low`, `haze`, `rain`, `snow`
- `mixed-only average`: remaining `7` folders
- `overall average`: all `11` folders
- `unseen 4-way average`

Also record classifier diagnostics for all classifier-based variants.

This is important because a single overall average can hide whether we improved:

- single restoration only
- mixed restoration only
- both

## Minimal High-Value Run Order

If compute is limited, follow this order:

1. Diagnose current `Row C`
2. Run `B-single`
3. Run `C-curriculum`
4. Run `C-curriculum-balanced-ft`
5. Run `D-calibrated` using the best `C` model
6. Run `D-calibrated-gated` only if prompt noise still looks harmful

## How To Interpret The Results

- If `B-single` improves over current `Row B`, mixed pretraining is hurting purity.
- If `C-curriculum` improves over `B-single`, primitive-aware supervision is helping encoder learning.
- If `C-curriculum-balanced-ft` improves over `C-curriculum`, finetuning balance is a major issue.
- If `D-calibrated` improves over the best `C` variant, prompt guidance is useful when classifier quality is fixed.
- If `D-calibrated-gated` improves over `D-calibrated`, prompt confidence control is needed.

## Proposed Future Configs

These are the experiment variants we should create next:

- `pretrain_baseline_single.yml`
- `pretrain_multilabel_single.yml`
- `pretrain_multilabel_mixed_calibration.yml`
- `finetune_multilabel_balanced.yml`
- `finetune_prompt_calibrated.yml`
- `finetune_prompt_calibrated_gated.yml`
- `scripts/analyze_classifier.py`

## Final Direction

The most promising path is:

`single-first pretraining -> mixed calibration -> balanced finetuning -> optional gated prompt guidance`

This path is simple, explainable, and directly targets the current weakness:

- learn pure degradations first
- learn composition next
- avoid finetuning bias
- use classifier guidance only when it is trustworthy
