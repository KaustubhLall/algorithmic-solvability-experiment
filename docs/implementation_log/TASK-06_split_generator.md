# TASK-06: Split Generator

> **Status:** COMPLETE ✓
> **Builds:** SR-4
> **Validation:** V-4
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-4), Part 3 (V-4), Part 4 (TASK-06)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25
> **Milestone:** DATA PIPELINE (TASK-05–06 complete)

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `SplitGenerator` class | `src/splits.py` | DONE |
| `SplitResult` dataclass | `src/splits.py` | DONE |
| `SplitStrategy` enum | `src/splits.py` | DONE |
| IID split | `src/splits.py` | DONE |
| Length extrapolation split | `src/splits.py` | DONE |
| Value extrapolation split | `src/splits.py` | DONE |
| Noise split (test-only noise) | `src/splits.py` | DONE |
| Convenience functions: `split_iid`, `split_length`, `split_value`, `split_noise` | `src/splits.py` | DONE |
| V-4 test suite | `tests/test_splits.py` | DONE (29 tests) |

---

## Implementation Notes

### Design Choices
- **`SplitStrategy` enum** — Four strategies: `IID`, `LENGTH_EXTRAPOLATION`, `VALUE_EXTRAPOLATION`, `NOISE`. Each produces a `SplitResult` with metadata about the split parameters.
- **`SplitResult` has accessors** — `train_inputs`, `train_outputs`, `test_inputs`, `test_outputs` convenience properties that the Model Harness consumes directly.
- **Length extrapolation** — Sequences with `len <= threshold` go to train, `len > threshold` to test. Tests whether models generalize to longer inputs.
- **Value extrapolation** — For tabular: samples where a specific feature falls within `train_range` go to train, others to test. For sequences: all values must be within range. Tests whether models generalize to unseen value ranges.
- **Noise split** — IID split first, then perturb test inputs only. Labels are unchanged (they reference clean data, per ADR-004). Tests robustness to input perturbation.
- **Validation in `split_iid`** — Rejects `train_fraction` outside (0, 1).

### Edge Cases Handled
- Length threshold that puts all samples in train (threshold > max length) or all in test (threshold = 0).
- Value range that captures all or no samples.
- Zero noise level produces clean test set.

---

## Acceptance Criteria Results

V-4 validation: **29 tests, all passing** (pytest, 0.12s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| IID split | Correct sizes, no overlap, reproducible | 10 tests pass | YES |
| Length extrapolation | Train short, test long | 6 tests pass | YES |
| Value extrapolation | Train narrow, test wide | 5 tests pass | YES |
| Noise split | Train clean, test noisy, labels correct | 6 tests pass | YES |
| Classification splits | IID + noise work on tabular | 2 tests pass | YES |

---

## Deviations from Plan

### DEV-005: DistractorSplit deferred

- **Type:** SCOPE_DEFERRAL
- **What changed:** The `DISTRACTOR` split strategy (adding irrelevant features to test set) is defined in the `SplitStrategy` enum but not implemented yet.
- **Why:** Requires `TabularInputSchema.with_extra_irrelevant()` method to generate distractor features at split time. Will implement when TASK-12/13 experiments need it.
- **Impact:** No impact on pipeline development. The enum value exists for forward compatibility.

---

## Completion Summary

TASK-06 delivered the Split Generator in `src/splits.py`. Four split strategies are implemented: IID random, length extrapolation, value extrapolation, and noise injection. All 29 V-4 tests pass. This completes the DATA PIPELINE milestone (TASK-05–06). One deviation: DistractorSplit deferred to experiment phase. The Split Generator feeds directly into the Model Harness (TASK-07) via `SplitResult` accessors.
