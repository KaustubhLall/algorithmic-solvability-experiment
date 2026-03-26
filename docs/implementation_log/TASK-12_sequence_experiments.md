# TASK-12: Sequence Experiments

- **Date completed:** 2026-03-26
- **Scope:** Implement and run the sequence-track experiment suite supported by the current registry/model/split stack.
- **Primary spec:** `docs/EXPERIMENT_CATALOG.md` Part 2 (`EXP-S1` to `EXP-S5`) and Part 4 (`TASK-12`)

## What Was Added

- Added `src/sequence_experiments.py` with runnable TASK-12 experiment specs and helpers:
  - `build_sequence_experiment_specs()`
  - `run_sequence_experiment()`
  - `run_all_sequence_experiments()`
- Extended `main.py` with a `sequence` command so TASK-12 can be executed via:

```bash
.\.venv\Scripts\python.exe main.py sequence --output-root results
```

- Added `tests/test_sequence_experiments.py` to validate:
  - the spec builder covers the supported S1-S3 sequence tiers,
  - the chosen model families and split menus are as expected,
  - experiment artifact generation works end-to-end.
- Updated `src/models/harness.py` so `LSTMSequenceModel` can safely encode unseen test-time input tokens via an unknown bucket during value-range extrapolation.
- Retuned the TASK-11 smoke LSTM config in `src/smoke_tests.py` so the existing smoke gate stays above 90% exact match after the unknown-token change.

## Experiment Definitions Implemented

### EXP-S1

- **Tasks:** all registered S1 sequence tasks
  - `S1.1_reverse`
  - `S1.2_sort`
  - `S1.3_rotate`
  - `S1.4_count_symbol`
  - `S1.5_parity`
  - `S1.6_prefix_sum`
  - `S1.7_deduplicate`
  - `S1.8_extrema`
- **Models:** `majority_class`, `sequence_baseline`, `mlp`, `lstm`
- **Splits:** `iid`, `length_extrapolation`, `value_extrapolation`
- **Samples:** 600
- **Seeds:** 42, 123, 456

### EXP-S2

- **Tasks:** all registered S2 sequence tasks
  - `S2.1_cumulative_xor`
  - `S2.2_balanced_parens`
  - `S2.3_running_min`
  - `S2.5_checksum`
- **Models:** `majority_class`, `sequence_baseline`, `mlp`, `lstm`
- **Splits:** `iid`, `length_extrapolation`
- **Samples:** 600
- **Seeds:** 42, 123, 456

### EXP-S3

- **Tasks:** all registered S3 sequence tasks
  - `S3.1_dedup_sort_count`
  - `S3.2_filter_sort_sum`
  - `S3.4_rle_encode`
- **Models:** `majority_class`, `sequence_baseline`, `mlp`, `lstm`
- **Splits:** `iid`, `length_extrapolation`, `value_extrapolation`
- **Samples:** 500
- **Seeds:** 42, 123, 456

## Results Generated

Artifacts were written to:

- `results/EXP-S1`
- `results/EXP-S2`
- `results/EXP-S3`

Each experiment includes:

- `config.json`
- `summary.md`
- `comparison.md`
- `solvability_verdicts.json`
- per-task `metrics.json`
- per-task `errors.json`
- per-task `extrap_curve.png`
- per-task `confusion.png` when applicable

## Observed Outcomes

### EXP-S1

- Runtime: 81.80s
- 8 tasks, 276 single runs, 92 aggregated groups
- Best outcomes:
  - `S1.4_count_symbol` -> `WEAK` (`best_iid_accuracy=0.9833`, `best_ood_accuracy=0.8438`, best model `lstm`)
  - `S1.5_parity` -> `INCONCLUSIVE`
  - `S1.8_extrema` -> `INCONCLUSIVE`
- All other S1 tasks were `NEGATIVE`

### EXP-S2

- Runtime: 36.91s
- 4 tasks, 96 single runs, 32 aggregated groups
- Best outcome:
  - `S2.2_balanced_parens` -> `WEAK` (`best_iid_accuracy=0.9870`, `best_ood_accuracy=0.9927`, best model `mlp`)
- `S2.1_cumulative_xor`, `S2.3_running_min`, and `S2.5_checksum` were `NEGATIVE`

### EXP-S3

- Runtime: 23.28s
- 3 tasks, 108 single runs, 36 aggregated groups
- All S3 tasks were `NEGATIVE`

## Validation Performed

### Targeted tests

- `tests/test_sequence_experiments.py`
- `tests/test_model_harness.py`
- `tests/test_runner.py`
- `tests/test_reporting.py`

### Full suite

- Ran `.\.venv\Scripts\python.exe -m pytest -q`
- Result: **414 passed**

### End-to-end reruns

- Regenerated `results/EXP-S1` through `results/EXP-S3`
- Refreshed `results/EXP-0.1` through `results/EXP-0.3` after the LSTM compatibility change

## Key Implementation Decisions

- Added an explicit unknown-token bucket for sequence LSTM inference so value-range extrapolation cannot crash on unseen test-time inputs.
- Kept the unknown bucket input-only; the output vocabulary still reflects train-time targets so the model interface and smoke behavior remain stable.
- Restricted `EXP-S2` to IID + length extrapolation because value-range extrapolation is not meaningful for the currently registered binary/stateful tasks in that tier.

## Deviations Logged

- `DEV-010`: TASK-12 runs the implemented S1-S3 sequence tiers only; `EXP-S4` and `EXP-S5` remain deferred.
- `DEV-011`: TASK-12 uses the currently validated sequence model families and substitutes value-range extrapolation for the cataloged EXP-S3 composition split until composition/transformer support is implemented.

## Files Changed

- `main.py`
- `src/models/harness.py`
- `src/sequence_experiments.py`
- `src/smoke_tests.py`
- `tests/test_model_harness.py`
- `tests/test_sequence_experiments.py`
- `docs/PROJECT_STATUS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/EXPERIMENT_CATALOG.md`
- `docs/implementation_log/TASK-12_sequence_experiments.md`

## Follow-on Task

- TASK-13: Classification Experiments
