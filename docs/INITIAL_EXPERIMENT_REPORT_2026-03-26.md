# Initial Experiment Report

**Date:** 2026-03-26  
**Workspace:** `C:\Users\kaust\PycharmProjects\DataScience`

## Scope

This report summarizes an initial execution pass over the implemented algorithmic solvability benchmark after reading the project documentation in `docs/`.

The following documented workflows were executed from the checked-in virtual environment:

- `.\.venv\Scripts\python.exe main.py smoke --output-root results`
- `.\.venv\Scripts\python.exe main.py sequence --output-root results`
- `.\.venv\Scripts\python.exe main.py classification --output-root results`
- `.\.venv\Scripts\python.exe main.py bonus --output-root results`

An attempt was also made to run:

- `.\.venv\Scripts\python.exe main.py diagnostic --output-root results`

However, the diagnostic workflow did not finish within a 20-minute shell timeout and was still running after an additional extended wait. No `results/EXP-D1` through `results/EXP-D5` artifact directories were produced during this run, so diagnostics are treated as **not completed** for this report.

## Executive Summary

The benchmark currently shows a strong split between the two task families:

- **Classification tasks are largely learnable and extrapolatable** under the current harness. Most C1-C3 tasks reached `MODERATE` solvability evidence, often with near-perfect IID and OOD performance.
- **Sequence tasks are mostly not solved** by the current model set. Only `S1.4_count_symbol` and `S2.2_balanced_parens` achieved `WEAK` evidence; the rest of the implemented S1-S3 tasks were `NEGATIVE` or `INCONCLUSIVE`.
- **Controls behaved as expected** in the smoke/control suite, supporting that the evaluation pipeline is not trivially over-calling solvability.
- **Bonus symbolic recovery is much stronger than sequence learning performance suggests.** Rule extraction succeeded on most eligible classification tasks, and DSL program search recovered exact programs for most searched sequence tasks. This suggests the benchmark tasks themselves are genuinely algorithmic, but the current sequence model family is not learning them reliably from examples alone.

## What Ran Successfully

### Smoke and control calibration

- `EXP-0.1` smoke sequence run completed.
- `EXP-0.2` smoke classification run completed.
- `EXP-0.3` control tasks completed.

Observed control behavior was directionally correct:

- `S0.1_random_labels`: `NEGATIVE`, score `0.1750`
- `C0.1_random_class`: `NEGATIVE`, score `0.2215`
- `C1.1_numeric_threshold` smoke calibration: `MODERATE`, score `0.7247`

This is a good sign that the pipeline distinguishes trivial deterministic rules from non-algorithmic controls.

### Main benchmark suites

Completed artifact groups:

- `results/EXP-S1`
- `results/EXP-S2`
- `results/EXP-S3`
- `results/EXP-C1`
- `results/EXP-C2`
- `results/EXP-C3`

### Bonus suites

Completed artifact groups:

- `results/EXP-B1`
- `results/EXP-B2`

## Main Quantitative Picture

Across the completed baseline/control suites (`EXP-0.3`, `EXP-S1` to `EXP-S3`, `EXP-C1` to `EXP-C3`), there are 30 task verdicts:

- **Sequence:** 16 tasks
- **Classification:** 14 tasks

### Sequence verdict distribution

- `NEGATIVE`: 12
- `WEAK`: 2
- `INCONCLUSIVE`: 2

Aggregate sequence performance:

- Mean best IID accuracy: `0.3195`
- Mean best OOD accuracy: `0.2135`

Most common winning models:

- `lstm`: 11 tasks
- `mlp`: 5 tasks

### Classification verdict distribution

- `MODERATE`: 11
- `WEAK`: 1
- `INCONCLUSIVE`: 1
- `NEGATIVE`: 1

Aggregate classification performance:

- Mean best IID accuracy: `0.9355`
- Mean best OOD accuracy: `0.9702`

Most common winning models:

- `decision_tree`: 6 tasks
- `random_forest`: 4 tasks
- `logistic_regression`: 2 tasks
- `gradient_boosted_trees`: 1 task
- `mlp`: 1 task

## Sequence Track Interpretation

The current sequence stack does **not** provide convincing evidence of broad algorithmic solvability for the implemented S1-S3 tasks.

### Positive pockets

- `S1.4_count_symbol`: `WEAK`
  - Best IID: `0.9870`
  - Best OOD: `0.8116`
  - Best model: `lstm`
- `S2.2_balanced_parens`: `WEAK`
  - Best IID: `0.9907`
  - Best OOD: `0.9927`
  - Best model in verdict artifact: `mlp`

These two tasks show that some algorithmic sequence behavior is accessible to the current pipeline, but neither task cleared the full baseline-separation requirement needed for `MODERATE`.

### Broad failure pattern

The majority of sequence tasks remained `NEGATIVE`, including:

- `S1.1_reverse`
- `S1.2_sort`
- `S1.3_rotate`
- `S1.6_prefix_sum`
- `S1.7_deduplicate`
- `S2.1_cumulative_xor`
- `S2.3_running_min`
- `S2.5_checksum`
- `S3.1_dedup_sort_count`
- `S3.2_filter_sort_sum`
- `S3.4_rle_encode`

This matters because many of these are canonical low-to-mid complexity algorithmic transformations. Under the design document, we would have hoped at least some simple transforms such as reverse, sort, and prefix sum would show substantially stronger evidence.

### Interpretation

The most likely interpretation is not that these tasks are non-algorithmic, but that the **current sequence representation/model mix is underpowered or mismatched** for systematic generalization. That interpretation is reinforced by the bonus program search results: several sequence tasks that models failed to learn were nonetheless exactly recoverable by symbolic search.

In plain terms: the tasks appear to be algorithmically structured, but the learning setup is mostly not extracting that structure from examples.

## Classification Track Interpretation

The classification track is the strongest part of the current system.

### Clear successes

The following tasks achieved `MODERATE` evidence with very high OOD performance:

- `C1.1_numeric_threshold`: OOD `1.0000`
- `C1.2_range_binning`: OOD `1.0000`
- `C1.3_categorical_match`: OOD `0.9319`
- `C1.5_numeric_comparison`: OOD `0.9916`
- `C2.2_or_rule`: OOD `0.9459`
- `C2.3_nested_if_else`: OOD `1.0000`
- `C2.5_k_of_n`: OOD `1.0000`
- `C2.6_categorical_gate`: OOD `1.0000`
- `C3.1_xor`: OOD `0.9990`
- `C3.3_rank_based`: OOD `0.9898`
- `C3.5_interaction_poly`: OOD `0.9807`

This is a strong empirical result. It suggests that for deterministic tabular rule systems, the benchmark can already produce meaningful evidence of learnability and extrapolation.

### Edge cases

- `C2.1_and_rule`: `WEAK`
  - IID and OOD were both essentially perfect, but it missed `criterion_3_baseline_separation`.
  - This suggests the task may be so easy that simple baselines also succeed, weakening its usefulness as a discrimination task.

- `C1.6_modular_class`: `INCONCLUSIVE`
  - Best IID: `0.8289`
  - Best OOD: `0.7733`
  - Best model: `random_forest`
  - Main issue: insufficient IID strength and seed instability

This is the one clearly challenging algorithmic classifier in the current C1-C3 implementation. Modular arithmetic seems to be a meaningful stress test that the current tabular models do not solve cleanly.

### Model-family takeaway

The winning models are mostly classical tabular learners:

- decision trees for simple threshold/gating tasks
- random forests for more interaction-heavy rules
- logistic regression for some linearly expressible relationships
- gradient boosting for selected combinational logic tasks

This is a sensible pattern and increases confidence that the benchmark is capturing real structure rather than arbitrary noise.

## Bonus Experiments

### EXP-B1: Rule extraction from classification models

Summary:

- Tasks evaluated: `12`
- Tasks passing the >99% hard-test threshold: `9`
- Pass rate: `75.0%`

Successful high-fidelity extraction included:

- `C1.1_numeric_threshold`
- `C1.2_range_binning`
- `C1.3_categorical_match`
- `C2.1_and_rule`
- `C2.2_or_rule`
- `C2.3_nested_if_else`
- `C2.5_k_of_n`
- `C2.6_categorical_gate`
- `C3.1_xor`

Notably, all extracted trees reported `uses_only_relevant = true`, which is encouraging. The recovered rules are not leaning on irrelevant features in the logged structural summaries.

The misses were:

- `C1.5_numeric_comparison` at `98.08%`
- `C3.3_rank_based` at `95.23%`
- `C3.5_interaction_poly` at `98.83%`

Interpretation:

- Simple and medium-complexity classification rules are often recoverable as compact symbolic structures.
- Some richer numeric interaction tasks still require deeper trees and do not compress cleanly into a near-perfect extracted rule.

### EXP-B2: DSL program search for sequence tasks

Summary:

- Tasks evaluated: `9`
- Tasks with recovered program >99%: `7`
- Pass rate: `77.8%`

Recovered exactly:

- `S1.1_reverse`
- `S1.2_sort`
- `S1.5_parity`
- `S1.6_prefix_sum`
- `S1.7_deduplicate`
- `S3.1_dedup_sort_count`
- `S3.2_filter_sort_sum`

Not recovered:

- `S1.4_count_symbol` best hard-test accuracy `0.391`
- `S1.8_extrema` best hard-test accuracy `0.123`

This result is one of the most interesting findings in the entire run. Several sequence tasks that the learned models failed to solve are nonetheless recoverable by explicit DSL search. That strongly suggests:

- the tasks are appropriately algorithmic,
- the DSL is expressive enough for much of the benchmark,
- the main bottleneck is model learning/generalization rather than task construction.

## Execution Caveats and Issues

### 1. Diagnostic suite did not complete

The biggest operational issue in this run is the inability to finish `main.py diagnostic` within a practical window. Because no `EXP-D1` to `EXP-D5` directories were produced, this report cannot comment on:

- sample efficiency,
- distractor robustness,
- noise robustness,
- feature-importance alignment,
- calibrated verdict updates.

Those are important missing pieces, especially because the current baseline verdicts leave all criteria 6-9 unmet.

### 2. One split is structurally inapplicable

The classification run logged:

`Empty train or test set for C1.3_categorical_match/value_extrapolation/... Skipping.`

This is expected conceptually: a pure categorical match rule does not naturally support value-range extrapolation. The skip should be treated as a **split-design limitation**, not a task failure.

### 3. Training warnings appeared

Observed warnings included:

- multiple `MLPClassifier` convergence warnings
- multiple `LogisticRegression` convergence warnings
- repeated scikit-learn warnings about high class cardinality in some sequence-output settings

These did not crash the runs, but they are relevant when interpreting marginal outcomes, especially on the weaker sequence tasks.

## Overall Assessment

### What the project already demonstrates well

- The benchmark framework can generate, run, score, and report on a substantial set of deterministic synthetic tasks.
- The control tasks behave sensibly.
- The classification track already provides credible evidence that many deterministic tabular rules are learnable and extrapolatable from examples.
- The bonus modules show that symbolic recovery is often feasible once the task class is right.

### What remains unresolved

- The sequence track is far weaker than the classification track under the current model family and split setup.
- Diagnostic evidence is missing from this run because the diagnostic pipeline did not complete.
- Several of the most algorithmically canonical sequence tasks remain decisively unsolved by the learned models, even when symbolic search can recover them exactly.

## Recommended Next Steps

1. Make `TASK-14` diagnostics runnable within a predictable wall-clock budget.
2. Prioritize sequence-track improvements before expanding the benchmark further.
3. Add split-selection logic so non-applicable extrapolation regimes are excluded up front rather than skipped mid-run.
4. Investigate whether sequence feature engineering or architecture choice is the main cause of failure on reverse/sort/prefix-sum class tasks.
5. Use the bonus-search successes as a guide for targeted sequence-model redesign: the tasks are solvable in the DSL, so the learning objective or representation is likely the bottleneck.

## Bottom Line

The initial run supports a nuanced conclusion:

- **Yes, the framework can already detect algorithmic solvability in many tabular classification settings.**
- **No, it does not yet show broad empirical algorithmic solvability for the implemented sequence tasks under the current learned model stack.**
- **Symbolic recovery results indicate the negative sequence outcomes are more likely a modeling limitation than a benchmark-design limitation.**

That makes the project promising, but incomplete: the classification side is already informative, while the sequence side still needs substantial iteration before the broader research claim is well supported.
