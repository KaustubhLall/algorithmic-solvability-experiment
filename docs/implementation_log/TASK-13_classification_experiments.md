# TASK-13: Classification Experiments

- **Date completed:** 2026-03-26
- **Scope:** Implement and run the classification-track experiment suite supported by the current registry/model/split stack.
- **Primary spec:** `docs/EXPERIMENT_CATALOG.md` Part 2 (`EXP-C1` to `EXP-C5`) and Part 4 (`TASK-13`)

## What Was Added

- Added `src/classification_experiments.py` with runnable TASK-13 experiment specs and helpers:
  - `build_classification_experiment_specs()`
  - `run_classification_experiment()`
  - `run_all_classification_experiments()`
- Extended `main.py` with a `classification` command so TASK-13 can be executed via:

```bash
.\.venv\Scripts\python.exe main.py classification --output-root results
```

- Added `tests/test_classification_experiments.py` to validate:
  - the spec builder covers the supported C1-C3 classification tiers,
  - the chosen model families and split menus are as expected,
  - experiment artifact generation works end-to-end.
- Updated `src/splits.py` and `src/runner.py` so tabular `NOISE` splits can use the task schema to flip categorical features within their valid domain during test-time perturbation.
- Added a regression test in `tests/test_splits.py` confirming categorical-only classification tasks now receive real noisy test inputs while labels remain tied to the clean inputs.

## Experiment Definitions Implemented

### EXP-C1

- **Tasks:** all registered C1 classification tasks
  - `C1.1_numeric_threshold`
  - `C1.2_range_binning`
  - `C1.3_categorical_match`
  - `C1.5_numeric_comparison`
  - `C1.6_modular_class`
- **Models:** `majority_class`, `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosted_trees`, `mlp`
- **Splits:** `iid`, `value_extrapolation`, `noise`
- **Samples:** 900
- **Seeds:** 42, 123, 456, 789, 1024

### EXP-C2

- **Tasks:** all registered C2 classification tasks
  - `C2.1_and_rule`
  - `C2.2_or_rule`
  - `C2.3_nested_if_else`
  - `C2.5_k_of_n`
  - `C2.6_categorical_gate`
- **Models:** `majority_class`, `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosted_trees`, `mlp`
- **Splits:** `iid`, `value_extrapolation`, `noise`
- **Samples:** 900
- **Seeds:** 42, 123, 456, 789, 1024

### EXP-C3

- **Tasks:** all registered C3 classification tasks
  - `C3.1_xor`
  - `C3.3_rank_based`
  - `C3.5_interaction_poly`
- **Models:** `majority_class`, `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosted_trees`, `mlp`
- **Splits:** `iid`, `value_extrapolation`, `noise`
- **Samples:** 1000
- **Seeds:** 42, 123, 456, 789, 1024

## Results Generated

Artifacts were written to:

- `results/EXP-C1`
- `results/EXP-C2`
- `results/EXP-C3`

Each experiment includes:

- `config.json`
- `summary.md`
- `comparison.md`
- `solvability_verdicts.json`
- per-task `metrics.json`
- per-task `errors.json`
- per-task `extrap_curve.png`
- per-task `confusion.png`

## Observed Outcomes

### Overall suite

- Runtime: 160.25s total
- 13 tasks, 1,140 single runs, 228 aggregated groups
- Verdict distribution:
  - `MODERATE`: 11 tasks
  - `WEAK`: 1 task (`C2.1_and_rule`)
  - `INCONCLUSIVE`: 1 task (`C1.6_modular_class`)

### EXP-C1

- Runtime: 52.29s
- 5 tasks, 420 single runs, 84 aggregated groups
- Best outcomes:
  - `C1.1_numeric_threshold` -> `MODERATE` (`best_iid_accuracy=1.0000`, `best_ood_accuracy=1.0000`, best model `decision_tree`)
  - `C1.2_range_binning` -> `MODERATE` (`best_iid_accuracy=1.0000`, `best_ood_accuracy=1.0000`, best model `decision_tree`)
  - `C1.3_categorical_match` -> `MODERATE` (`best_iid_accuracy=1.0000`, `best_ood_accuracy=0.9319`, best model `decision_tree`)
  - `C1.5_numeric_comparison` -> `MODERATE` (`best_iid_accuracy=0.9978`, `best_ood_accuracy=0.9916`, best model `logistic_regression`)
- Hard case:
  - `C1.6_modular_class` -> `INCONCLUSIVE` (`best_iid_accuracy=0.8289`, `best_ood_accuracy=0.7733`, best model `random_forest`)
- `C1.3_categorical_match` skipped `value_extrapolation` on all 5 seeds because the task has no numeric feature `x1`; it still received IID + schema-guided categorical noise evaluation.

### EXP-C2

- Runtime: 67.92s
- 5 tasks, 450 single runs, 90 aggregated groups
- `C2.2_or_rule`, `C2.3_nested_if_else`, `C2.5_k_of_n`, and `C2.6_categorical_gate` all reached `MODERATE`
- `C2.1_and_rule` landed at `WEAK` despite near-perfect IID/OOD accuracy because the best IID model did not clear the reporting layer's baseline-separation threshold

### EXP-C3

- Runtime: 40.05s
- 3 tasks, 270 single runs, 54 aggregated groups
- All implemented C3 tasks reached `MODERATE`
  - `C3.1_xor` best model: `random_forest`
  - `C3.3_rank_based` best model: `logistic_regression`
  - `C3.5_interaction_poly` best model: `random_forest`

## Validation Performed

### Targeted tests

- `tests/test_classification_experiments.py`
- `tests/test_splits.py`
- `tests/test_runner.py`
- `tests/test_reporting.py`

### Focused validation result

- Ran `.\.venv\Scripts\python.exe -m pytest -q tests/test_classification_experiments.py tests/test_splits.py tests/test_runner.py tests/test_reporting.py`
- Result: **85 passed**

### Full suite

- Ran `.\.venv\Scripts\python.exe -m pytest -q`
- Result: **418 passed**

### End-to-end reruns

- Generated `results/EXP-C1` through `results/EXP-C3`
- Verified per-experiment summaries and solvability verdicts after artifact generation

## Key Implementation Decisions

- Used task-schema-guided categorical flips inside the `NOISE` split so categorical classification tasks get a real robustness regime instead of an IID duplicate.
- Used a shared mid-range value training band (`x1 in [20, 80]`) for the implemented classification experiments so threshold-like tasks retain class support in training while still exposing unseen tails at test time.
- Kept the TASK-13 suite on the currently validated SR-5/SR-4 surface: classical tabular models plus `iid`, `value_extrapolation`, and `noise`.

## Deviations Logged

- `DEV-012`: TASK-13 runs the implemented C1-C3 classification tiers only; the remaining catalog-only C-track tasks plus `EXP-C4` and `EXP-C5` remain deferred.
- `DEV-013`: TASK-13 uses the currently validated classification model families and available OOD splits instead of the catalog's broader architecture/split matrix.

## Files Changed

- `main.py`
- `src/classification_experiments.py`
- `src/runner.py`
- `src/splits.py`
- `tests/test_classification_experiments.py`
- `tests/test_runner.py`
- `tests/test_splits.py`
- `docs/PROJECT_STATUS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/EXPERIMENT_CATALOG.md`
- `docs/implementation_log/TASK-13_classification_experiments.md`

## Follow-on Task

- TASK-14: Diagnostic Experiments
