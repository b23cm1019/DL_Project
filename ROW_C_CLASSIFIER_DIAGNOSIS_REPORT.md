# Row C Classifier Diagnosis Report

## Purpose

This report summarizes what we observed after running the `Row C` degradation-classifier diagnosis on the full `CDD-11` test split.

The main goal of this diagnosis was to verify an important concern:

- whether training on mixed degradations causes the model to wrongly assume multiple degradations even for single-degradation images
- whether this classifier behavior is the main reason restoration quality is not beating stronger baselines or paper-reported results

## What We Evaluated

We ran the pretrained `Row C` encoder and classifier on the full `CDD-11` test set and compared:

- predicted 4D degradation vector
- thresholded multi-hot prediction
- ground-truth primitive vector

The diagnosis outputs are stored in:

- [summary.json](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/row_c_classifier_full_20260718/summary.json>)
- [per_setting_summary.csv](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/row_c_classifier_full_20260718/per_setting_summary.csv>)
- [primitive_metrics.csv](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/row_c_classifier_full_20260718/primitive_metrics.csv>)
- [per_image_predictions.csv](</C:/Users/krish/Desktop/btp/DPU/DL_Project_modified/DL_Project/row_c_classifier_full_20260718/per_image_predictions.csv>)

Note:
The files are currently available in the local folder `row_c_classifier_full_20260718/`.

## Initial Hypothesis

Before running this diagnosis, our working hypothesis was:

- because `Row C` sees mixed degradations during pretraining, the classifier may over-predict multiple degradations
- this may make the model treat single-degradation images as mixed ones
- this could be a major reason for weaker restoration performance on single-degradation cases

## Main Result

The diagnosis does **not** support the idea that the `Row C` classifier is broadly failing.

Instead, the classifier is very strong overall.

## Overall Metrics

From the full diagnosis on `2200` test images:

- exact multi-hot match accuracy: `98.0%`
- cardinality accuracy: `98.0%`
- micro precision: `0.9955`
- micro recall: `0.9935`
- micro F1: `0.9945`
- macro precision: `0.9962`
- macro recall: `0.9946`
- macro F1: `0.9954`

These numbers show that the classifier is learning the primitive decomposition of degradations very well.

## Single vs Mixed Performance

The most useful split is:

- `single settings`: `low`, `haze`, `rain`, `snow`
- `mixed settings`: the remaining `7` folders

Average exact-match accuracy:

- single-only: `99.5%`
- mixed-only: `97.14%`
- all settings: `98.0%`

Interpretation:

- the classifier is extremely reliable on single-degradation inputs
- it is slightly weaker on mixed degradations, but still very strong
- therefore, `Row C` already distinguishes single and mixed degradations well at the classification level

## Primitive-Level Findings

Primitive-wise results show a very clear pattern.

- `low`: F1 `0.9833`
- `haze`: F1 `0.9983`
- `rain`: F1 `1.0000`
- `snow`: F1 `1.0000`

### Key Observation

The weakest primitive is `low-light`.

This is the only channel where most of the remaining mistakes are concentrated.

That means the issue is **not general multi-label confusion**.
It is much more specific:

- the classifier sometimes adds `low` when it should not
- the classifier sometimes misses `low` when it should be present

## Where the Errors Happen

There are only `44` wrong exact matches out of `2200` images.

They are concentrated in a few settings:

- `haze_snow`: `10` errors
- `low_snow`: `6`
- `haze_rain`: `6`
- `low_rain`: `5`
- `low_haze_snow`: `5`

The important part is that these errors are not random.

They follow two structured patterns.

### Pattern 1: False Positive `low`

Examples:

- `haze -> low + haze`
- `haze_rain -> low + haze + rain`
- `haze_snow -> low + haze + snow`

This means the classifier sometimes interprets haze-heavy scenes as also having low-light characteristics.

### Pattern 2: False Negative `low`

Examples:

- `low_haze -> haze`
- `low_rain -> rain`
- `low_snow -> snow`
- `low_haze_rain -> haze + rain`
- `low_haze_snow -> haze + snow`

This means that in some scenes the classifier underestimates the low-light component.

## Important Image-Level Observation

Some image IDs repeatedly appear in the errors across several settings, especially:

- `00371`
- `00730`
- `04181`
- `04566`

This is very useful.

It suggests that the problem is not only the degradation label itself, but also certain scene characteristics.
In other words, some scenes consistently make low-light attribution harder across different degradation combinations.

This can be used later for qualitative examples in the report or evaluator discussion.

## Why This Changes Our Earlier Understanding

Before the diagnosis, we were mainly worried that:

- mixed-degradation training may be making the classifier unreliable

After the diagnosis, the stronger conclusion is:

- the `Row C` classifier is already reliable
- hard multi-hot prediction is **not** the main bottleneck
- the remaining weakness is mostly around the `low` primitive

So our earlier story should be updated.

Instead of saying:

- "the classifier cannot distinguish single vs mixed degradation properly"

we should now say:

- "the classifier distinguishes single vs mixed degradations well overall, but low-light calibration is the main remaining classification weakness"

## Most Important Hidden Insight: Soft Prompt Contamination

Although the thresholded predictions are very accurate, the soft probabilities still matter a lot for `Row D`.

This is because `PromptSRModel` uses:

- `sigmoid(cls_output)`

directly as the restoration prompt.

That means even when a primitive is not predicted after thresholding, a non-trivial soft score can still influence restoration.

Examples from the diagnosis:

- on `haze`, mean `low` probability is about `0.24`
- on `rain`, mean `low` probability is about `0.17`
- on `snow`, mean `low` probability is about `0.16`
- on `haze_rain`, mean `low` probability is about `0.30`
- on `haze_snow`, mean `low` probability is about `0.27`

These values are below `0.5`, so classification remains correct.
But they are still not close to zero.

### Why This Matters

For `Row C`, this is less harmful because the classifier is mainly used as a weak-supervision signal during pretraining.

For `Row D`, this can be harmful because:

- the restoration network receives a soft prompt
- the low-light expert signal may be partially injected even when low-light is absent

This is a much stronger explanation for prompt-related degradation than the earlier assumption of complete classifier failure.

## Final Interpretation

The diagnosis suggests the following:

1. `Row C` classification is already strong.
2. The classifier is not broadly confusing single and mixed degradations.
3. The main classification weakness is `low-light` calibration.
4. The bigger remaining bottleneck is likely **not** hard classification accuracy.
5. The more likely bottlenecks now are:
   - restoration feature quality
   - finetuning distribution bias
   - soft prompt contamination in `Row D`

## What We Should Focus On Now

Based on this diagnosis, our next focus should shift.

### 1. Stop treating classifier failure as the primary problem

This diagnosis shows that the classifier itself is already very strong.

So future improvements should focus less on:

- "how do we make the classifier stop predicting mixed degradations everywhere"

and more on:

- "how do we use the classifier signal more effectively for restoration"

### 2. Prioritize finetuning and restoration-side improvements

The likely improvement areas now are:

- better pretraining curriculum for encoder quality
- better finetuning balance between single and mixed degradations
- more controlled prompt usage during restoration

### 3. Add prompt control for Row D

Since soft low-light scores are non-zero even on some non-low settings, the next high-value change is:

- confidence-gated prompting

This should help reduce incorrect restoration conditioning from weak but nonzero prompt signals.

### 4. Keep low-light as a special focus

Because `low` is the weakest primitive, it deserves special attention in future experiments:

- low-light-aware calibration
- stronger low-light-specific sampling
- more analysis of scenes where low-light is missed or over-added

## Recommended Next Experimental Priorities

Based on these results, the most meaningful next steps are:

1. `single-first -> mixed-calibration -> balanced finetuning`
   Reason:
   the classifier is already good, so now we want to improve encoder and restoration behavior rather than only classification accuracy.

2. `confidence-gated prompt injection for Row D`
   Reason:
   thresholded predictions are strong, but soft prompt contamination may still hurt restoration.

3. `single vs mixed restoration metric split`
   Reason:
   we now know classifier performance is strong, so we should check whether restoration quality itself is where the single/mixed gap really appears.

4. `image-level correlation analysis`
   Reason:
   compare low-light prompt strength with poor restoration cases to verify whether prompt leakage is truly harming performance.

## Evaluator-Facing Conclusion

The diagnosis gave a very important correction to our initial assumption.

We started with the concern that the model may be fundamentally misclassifying single-degradation images as mixed ones because of training on `CDD-11`.

After the full analysis, we found:

- the `Row C` classifier is already highly accurate
- it distinguishes single and mixed degradations very well
- the remaining classification weakness is mainly confined to the `low-light` primitive
- therefore, the next improvements should focus less on classifier correctness and more on:
  - restoration-side feature learning
  - finetuning balance
  - soft prompt control

In short:

- the diagnosis reduces uncertainty around the classifier
- narrows the problem to a much smaller failure mode
- gives a clearer and more defensible direction for the next experiments
