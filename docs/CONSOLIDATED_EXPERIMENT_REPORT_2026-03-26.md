# Consolidated Experiment Report

**Date:** 2026-03-26  
**Workspace:** `C:\Users\kaust\PycharmProjects\DataScience`

## Objective

This report consolidates the engineering cleanup, validation status, refreshed experiment execution, and post-fix interpretation of the algorithmic solvability benchmark.

It supersedes the earlier initial report by incorporating:

- code fixes for experiment reliability and reporting clarity,
- a full passing test run,
- a complete baseline + diagnostic + bonus rerun,
- updated empirical conclusions after `EXP-D1` through `EXP-D5` completed successfully.

## What Was Fixed

### 1. Diagnostic runtime bottlenecks

The diagnostic suite was previously not completing in practice. The following changes were made to make it runnable:

- `src/diagnostic_experiments.py`
  - cached IID splits in `EXP-D1` instead of regenerating datasets for every sample-size point,
  - cached distractor/noise/feature-alignment datasets and splits so models reuse the same generated data,
  - reduced the default `EXP-D1` sample-size ladder from `[100, 500, 1000, 2000, 5000, 10000]` to `[100, 250, 500, 1000, 2000]`,
  - reduced the default `EXP-D1` test size from `2000` to `1000`.

Outcome:

- the full diagnostic stack now completes successfully in about **530.6 seconds** on this machine.

### 2. Split validation for inapplicable value extrapolation

The old value-extrapolation logic silently placed nonnumeric/missing tabular features into the training set, which produced empty test splits for tasks like `C1.3_categorical_match`.

This was corrected in `src/splits.py` so value extrapolation now fails fast with an explicit error when the requested feature is not present or not numeric.

Outcome:

- the classification rerun now logs a clean, intentional skip:
  - `Skipping split value_extrapolation for task C1.3_categorical_match ...`
- this is now a clear split-applicability issue rather than a confusing empty-split artifact.

### 3. Reporting consistency

The report layer previously mixed several notions of “best model,” which made some summaries look inconsistent.

This was corrected in `src/reporting.py` by adding:

- `best_iid_model`
- `best_ood_model`

and aligning the persisted `best_model` field with the selected IID winner used for primary solvability interpretation.

Outcome:

- verdict JSON and markdown summaries now make it explicit which model won on IID versus OOD.

## Validation Status

The full automated test suite passes:

- `460 passed`
- `17 warnings`

Command used:

- `.\.venv\Scripts\python.exe -m pytest -q`

The remaining warnings are training/runtime warnings rather than test failures:

- `MLPClassifier` convergence warnings
- `LogisticRegression` convergence warnings
- scikit-learn “many unique classes” warnings on some sequence-output settings

## Full Rerun Coverage

The following workflows were rerun successfully:

- `main.py smoke`
- `main.py sequence`
- `main.py classification`
- `main.py diagnostic`
- `main.py bonus`

Approximate wall-clock times from this refreshed pass:

- smoke: `59.7s`
- sequence: `161.0s`
- classification: `186.3s`
- diagnostic: `530.6s`
- bonus: `67.3s`

Total experiment execution time was about **1005 seconds** or **16.7 minutes**.

## Baseline Results

### Control behavior

The controls behaved sensibly:

- `S0.1_random_labels`: `NEGATIVE`
- `C0.1_random_class`: `NEGATIVE`

This continues to support that the evaluation pipeline is not trivially over-attributing solvability.

### Sequence track

Across 16 sequence/control tasks:

- `NEGATIVE`: 12
- `WEAK`: 2
- `INCONCLUSIVE`: 2

Mean best accuracies:

- IID: `0.3195`
- OOD: `0.2135`

The best sequence tasks remain:

- `S1.4_count_symbol`: `WEAK`
- `S2.2_balanced_parens`: `WEAK`

The broad pattern remains the same as in the initial run: current learned sequence models still do not provide strong evidence of general algorithmic generalization across the implemented S1-S3 benchmark.

### Classification track

Across 14 classification/control tasks:

- `MODERATE`: 11
- `WEAK`: 1
- `INCONCLUSIVE`: 1
- `NEGATIVE`: 1

Mean best accuracies:

- IID: `0.9355`
- OOD: `0.9702`

The classification track remains the strongest part of the benchmark. Most deterministic rule-based tabular tasks are learned with near-perfect IID and OOD performance.

Representative strong baseline tasks include:

- `C1.1_numeric_threshold`
- `C1.2_range_binning`
- `C1.5_numeric_comparison`
- `C2.3_nested_if_else`
- `C2.6_categorical_gate`
- `C3.1_xor`
- `C3.3_rank_based`
- `C3.5_interaction_poly`

The main unresolved baseline classification case remains:

- `C1.6_modular_class`: `INCONCLUSIVE`

## Diagnostic Results

### EXP-D1: Sample efficiency

Key D1 results:

- Controls fail the sample-efficiency criterion, as intended.
- All six algorithmic comparison tasks pass `criterion_8`.

Notable AUCs:

- `C1.1_numeric_threshold`: `0.9985`
- `C2.3_nested_if_else`: `0.9988`
- `C3.3_rank_based`: `0.9911`
- `S1.4_count_symbol`: `0.9640`
- `S2.2_balanced_parens`: `0.9863`

Interpretation:

- the sample-efficiency diagnostic strongly separates algorithmic tasks from control tasks,
- even some sequence tasks that still lack strong final solvability labels exhibit nontrivial algorithmic sample-efficiency signatures.

### EXP-D2: Distractor robustness

Passes:

- `C1.1_numeric_threshold`
- `C2.1_and_rule`
- `C2.6_categorical_gate`
- `C3.5_interaction_poly`

Failure:

- `C3.1_xor` selected `decision_tree`, score `0.5659`

Interpretation:

- most tested classification tasks are robust to appended irrelevant features,
- `C3.1_xor` is the clearest distractor-sensitivity failure in the diagnostic suite.

### EXP-D3: Noise robustness

All tested tasks passed:

- `C1.1_numeric_threshold`
- `C2.1_and_rule`
- `C3.1_xor`
- `C3.5_interaction_poly`

Interpretation:

- moderate input perturbation does not meaningfully break the strongest classification models on these tasks.

### EXP-D4: Feature-importance alignment

All tested task/model combinations reached:

- `Precision@k = 1.0000`

for:

- `C2.1_and_rule`
- `C2.6_categorical_gate`
- `C3.1_xor`

across:

- decision tree
- gradient boosted trees
- MLP

Interpretation:

- on the tested subset, model salience aligns perfectly with the true relevant feature set.

### EXP-D5: Calibrated solvability labels

Calibration checks passed:

- controls negative/weak: `True`
- trivial tasks strong: `True`

Label changes after diagnostics:

- `C1.1_numeric_threshold`: `MODERATE -> STRONG`

No other tasks changed label category under the current calibration rule.

This matters for interpretation:

- diagnostics strengthened confidence in several tasks numerically,
- but only `C1.1_numeric_threshold` crossed the threshold into a new final label.

## Bonus Results

### EXP-B1: Rule extraction

- tasks evaluated: `12`
- tasks passing >99% on hard test: `9`
- pass rate: `75.0%`

This remains a strong result for symbolic recovery on the classification side.

Misses were:

- `C1.5_numeric_comparison`
- `C3.3_rank_based`
- `C3.5_interaction_poly`

### EXP-B2: DSL program search

- tasks evaluated: `9`
- recovered exactly (>99%): `7`
- pass rate: `77.8%`

Recovered exactly:

- `S1.1_reverse`
- `S1.2_sort`
- `S1.5_parity`
- `S1.6_prefix_sum`
- `S1.7_deduplicate`
- `S3.1_dedup_sort_count`
- `S3.2_filter_sort_sum`

Not recovered:

- `S1.4_count_symbol`
- `S1.8_extrema`

This continues to support the key interpretation that many sequence tasks are algorithmically well-formed even when the current learned models fail to generalize well.

## Consolidated Interpretation

### What is now solid

- the repo is test-clean,
- the reporting artifacts are more internally consistent,
- the split behavior is cleaner on inapplicable extrapolation cases,
- the full diagnostic stack now completes successfully,
- the classification benchmark provides credible evidence of algorithmic solvability on a wide range of deterministic tabular tasks,
- symbolic recovery remains strong on both the classification and sequence bonus tracks.

### What is still weak

- the implemented sequence learning track still does not show broad algorithmic solvability under the current learned model stack,
- `C1.6_modular_class` remains a meaningful unresolved classification challenge,
- `C3.1_xor` shows distractor fragility despite strong baseline accuracy,
- convergence warnings remain in the training stack and should still be treated as technical debt.

## Bottom Line

After the cleanup pass, the project is in a much stronger state operationally:

- tests pass,
- diagnostics run,
- artifacts are clearer,
- the full benchmark can be executed end to end in a reasonable time.

Empirically, the conclusions are now sharper:

- **classification results are robust and convincing,**
- **sequence-learning results are still substantially behind,**
- **bonus symbolic recovery indicates that the sequence failures are more likely due to model/representation limitations than benchmark design flaws.**

## Key Artifacts

- [Initial report](C:\Users\kaust\PycharmProjects\DataScience\docs\INITIAL_EXPERIMENT_REPORT_2026-03-26.md)
- [D1 summary](C:\Users\kaust\PycharmProjects\DataScience\results\EXP-D1\summary.md)
- [D2 summary](C:\Users\kaust\PycharmProjects\DataScience\results\EXP-D2\summary.md)
- [D3 summary](C:\Users\kaust\PycharmProjects\DataScience\results\EXP-D3\summary.md)
- [D4 summary](C:\Users\kaust\PycharmProjects\DataScience\results\EXP-D4\summary.md)
- [D5 summary](C:\Users\kaust\PycharmProjects\DataScience\results\EXP-D5\summary.md)
- [B1 summary](C:\Users\kaust\PycharmProjects\DataScience\results\EXP-B1\summary.md)
- [B2 summary](C:\Users\kaust\PycharmProjects\DataScience\results\EXP-B2\summary.md)
