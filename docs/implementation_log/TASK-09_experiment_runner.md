# TASK-09: Experiment Runner

> **Status:** COMPLETE
> **Builds:** SR-7
> **Validation:** V-7 (36 tests passing)
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-7), Part 3 (V-7), Part 4 (TASK-09)
> **Date Started:** 2026-03-25
> **Date Completed:** 2026-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `ExperimentSpec` dataclass | `src/runner.py` | DONE |
| Result dataclasses: `SingleRunResult`, `AggregatedResult`, `ExperimentReport` | `src/runner.py` | DONE |
| Split dispatch helper (`_apply_split`) | `src/runner.py` | DONE |
| Single-run executor (`_run_single`) | `src/runner.py` | DONE |
| `run_experiment()` multi-seed orchestrator | `src/runner.py` | DONE |
| Aggregation across seeds (mean +/- std) | `src/runner.py` | DONE |
| Progress logging | `src/runner.py` | DONE |
| JSON-ready serialization helpers | `src/runner.py` | DONE |
| V-7 test suite | `tests/test_runner.py` | DONE (36 tests) |

---

## Implementation Notes

### Architecture

- **One generated dataset per (task, seed).** For each task/seed pair, the runner generates the dataset once and reuses it across all requested split strategies. This keeps split comparisons paired and avoids introducing extra variance from re-sampling the task data.
- **Split dispatch is centralized.** `_apply_split()` owns the mapping from `SplitStrategy` to the concrete split helper and validates strategy-specific parameters like `length_threshold`, `value_feature`, and `value_train_range`.
- **Aggregation groups by task/model/split.** `_aggregate_results()` computes mean and standard deviation across seeds for each `(task_id, model_name, split_strategy)` group, and also carries track-specific metrics like macro F1, exact match, and token accuracy when available.
- **Execution metadata is preserved for reporting.** `SingleRunResult` stores train/test sizes, train time, and `split_metadata`; `ExperimentReport` stores `seeds_used`. The serialization helpers expose the full experiment spec plus detailed run results so TASK-10 can emit files without reconstructing state.
- **Seed override support was added at the runner boundary.** `run_experiment()` accepts an optional `seeds` override while leaving `ExperimentSpec.seeds` as the default source of truth. This makes quick smoke runs easy without rebuilding the spec object.

### Edge Cases Handled

- Length and value extrapolation splits raise clear `ValueError`s if required parameters are missing.
- Empty train or test splits are skipped with a warning instead of crashing the whole experiment.
- Per-seed aggregation correctly yields `std = 0.0` for single-seed runs.
- Multi-task and multi-model experiments keep results separated by explicit grouping keys, preventing cross-contamination in the aggregated report.

---

## Acceptance Criteria Results

V-7 validation: **36 tests, all passing** (`tests/test_runner.py`, 0.95s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| End-to-end integrity | Mini experiments run and return well-formed reports | Classification + sequence mini experiments complete | YES |
| Seed variation | Different seeds produce distinct runs; same seed is reproducible | Seed override, reproducibility, and task-definition checks pass | YES |
| Multi-seed aggregation | Mean and std computed correctly per group | Aggregation tests pass for 1-seed and multi-seed cases | YES |
| No cross-contamination | Task/model/split results remain isolated | Multi-task and multi-model grouping tests pass | YES |
| Serialization | Report objects are JSON-serializable and include execution metadata | `single_result_to_dict`, `aggregated_result_to_dict`, `experiment_report_to_dict` pass | YES |

Additional validation: **full test suite passes** (`pytest`, 370 tests, 1.95s)

---

## Deviations from Plan

### DEV-007: `ExperimentSpec` uses total sample count plus train fraction

- **Type:** INTERFACE_CHANGE
- **What changed:** The SR-7 plan specified separate `n_train_samples` and `n_test_samples` fields. The implementation follows the existing SR-3/SR-4 pipeline and uses `n_samples` plus `train_fraction`, with split-specific logic deriving train/test sizes from the generated dataset.
- **Why:** The current Data Generator and Split Generator operate on whole datasets rather than separate pre-sized train/test pools. Keeping SR-7 aligned with those interfaces avoided duplicate sampling logic and preserved paired comparisons across split strategies.
- **Impact:** Future reporting code should read `n_samples`, `train_fraction`, and `split_metadata` from the experiment report. If exact per-split sample counts become important later, SR-7 can be extended without breaking the current interface.

---

## Completion Summary

TASK-09 delivered the Experiment Runner in `src/runner.py` with a complete in-memory orchestration flow: generate data, apply splits, run models, evaluate predictions, aggregate across seeds, and serialize the resulting report structure for downstream consumers. The implementation was already largely present on the branch; this session validated it against the full docs set, tightened the report metadata, extended V-7 coverage to 36 tests, and reconciled the project documentation to match the real code state. The key thing for TASK-10 is that SR-7 now returns enough structured information (`spec`, `seeds_used`, per-run metrics, and split metadata) for a report generator to write disk artifacts without re-running experiments.
