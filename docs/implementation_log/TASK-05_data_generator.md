# TASK-05: Data Generator

> **Status:** COMPLETE ✓
> **Builds:** SR-3
> **Validation:** V-3
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-3), Part 3 (V-3), Part 4 (TASK-05)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `DataGenerator` class | `src/data_generator.py` | DONE |
| `Sample` dataclass | `src/data_generator.py` | DONE |
| `Dataset` dataclass | `src/data_generator.py` | DONE |
| Label re-verification at generation time | `src/data_generator.py` | DONE |
| Noise injection (inputs only) | `src/data_generator.py` | DONE |
| Multi-task generation | `src/data_generator.py` | DONE |
| `compute_class_balance()` utility | `src/data_generator.py` | DONE |
| V-3 test suite | `tests/test_data_generator.py` | DONE (23 tests) |

---

## Implementation Notes

### Design Choices
- **`verify_labels=True` by default** — Every generated label is re-verified against the reference algorithm at generation time. This catches bugs in reference implementations early (per ADR-003).
- **Noise on inputs only, never labels (ADR-004)** — When `noise_level > 0`, inputs are perturbed but labels are always computed from the clean input. Noisy inputs have "stale" labels — intentional.
- **Seed arithmetic** — Sample `i` uses seed `base_seed + i`. Multi-task generation offsets by `task_index * n_samples_per_task` to avoid seed overlap.
- **`Sample` carries metadata** — Each sample records its seed, noise level, and task complexity metadata. This enables downstream analysis of per-sample properties.
- **`Dataset` has `.inputs` and `.outputs` accessors** — Convenience properties that extract lists from samples, used by downstream modules (Split Generator, Model Harness).
- **Noise implementation** — Sequence noise: randomly perturb int elements by ±2 with probability = noise_level. Tabular noise: add Gaussian noise to float features with probability = noise_level, scale = max(|val| × 0.1, 1.0).

---

## Acceptance Criteria Results

V-3 validation: **23 tests, all passing** (pytest, 0.23s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| Label correctness | All labels match reference | 3 tests pass (200 samples + all 32 tasks × 50 samples) | YES |
| Reproducibility | Same seed → same dataset | 3 tests pass | YES |
| Noise injection | Inputs modified, labels unchanged | 4 tests pass | YES |
| Metadata | Complete per-sample and per-dataset metadata | 4 tests pass | YES |
| Class balance | Correct distribution computation | 3 tests pass | YES |
| Dataset properties | Accessors and len work | 2 tests pass | YES |
| Multi-task generation | Multiple tasks, non-overlapping seeds | 3 tests pass | YES |

---

## Deviations from Plan

_None. Implementation follows SR-3 spec._

---

## Completion Summary

TASK-05 delivered the Data Generator in `src/data_generator.py`. The generator produces labeled datasets from any registered task, with label re-verification at generation time, noise injection (inputs only), multi-task support, and class balance computation. All 23 V-3 tests pass. The generator is the primary input to TASK-06 (Split Generator) and feeds all downstream pipeline components.
