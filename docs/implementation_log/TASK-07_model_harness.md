# TASK-07: Model Harness

> **Status:** COMPLETE ✓
> **Builds:** SR-5
> **Validation:** V-5
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-5), Part 3 (V-5), Part 4 (TASK-07)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `ModelHarness` class (encode → train → predict → decode) | `src/models/harness.py` | DONE |
| `BaseModel` ABC | `src/models/harness.py` | DONE |
| `ModelConfig` dataclass | `src/models/harness.py` | DONE |
| `ModelFamily` enum (8 families) | `src/models/harness.py` | DONE |
| `InputEncoder` (tabular + sequence) | `src/models/harness.py` | DONE |
| `LabelEncoder` (string ↔ int) | `src/models/harness.py` | DONE |
| `PredictionResult` dataclass | `src/models/harness.py` | DONE |
| `build_model()` factory | `src/models/harness.py` | DONE |
| `run_models()` convenience function | `src/models/harness.py` | DONE |
| Concrete models: MajorityClass, SklearnModelWrapper, SequenceBaseline | `src/models/harness.py` | DONE |
| V-5 test suite | `tests/test_model_harness.py` | DONE (39 tests) |

---

## Implementation Notes

### Design Choices
- **Unified pipeline** — `ModelHarness.run()` handles the full flow: raw inputs → encode → fit → predict → decode → string predictions. Downstream code (Evaluation Engine) only deals with string predictions.
- **`InputEncoder`** — Two modes: tabular (numerical pass-through, categorical label-encoded) and sequence (8 summary features: length, mean, std, min, max, sum, first, last). Fit on train, transform both train and test.
- **`LabelEncoder`** — Stringifies all labels, maps to ints for sklearn. Handles unseen labels at predict time by mapping to -1 / "UNKNOWN".
- **`SklearnModelWrapper`** — Generic wrapper for any sklearn-compatible estimator. Most model families use this.
- **`SequenceBaselineModel`** — Uses summary features + DecisionTreeClassifier. Intentionally weak on most sequence tasks — it's a baseline.
- **8 model families** — MajorityClass, LogisticRegression, DecisionTree, RandomForest, KNN, GradientBoostedTrees, MLP, SequenceBaseline. All use `random_state=42` for reproducibility.
- **`ModelConfig` with hyperparams** — Each family has sensible defaults. Custom hyperparams override via dict.
- **`run_models()` convenience** — Runs multiple model configs on the same split, returns list of `PredictionResult`.

### Edge Cases Handled
- Sequence tasks have many unique outputs (>50% of samples), which triggers a sklearn warning. Suppressed in tests.
- Unseen labels at prediction time return "UNKNOWN" rather than crashing.
- Empty inputs rejected by InputEncoder.

---

## Acceptance Criteria Results

V-5 validation: **39 tests, all passing** (pytest)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| Model instantiation | All 8 families build | 10 tests pass | YES |
| Prediction shape | Correct length and type | 2 tests pass | YES |
| Majority baseline | Always predicts mode | 2 tests pass | YES |
| Decision tree on trivial task | >95% accuracy on C1.1 | 1 test pass (perfect accuracy) | YES |
| InputEncoder | Tabular + sequence encoding correct | 4 tests pass | YES |
| LabelEncoder | Encode/decode, unseen handling | 4 tests pass | YES |
| E2E ModelHarness | Full pipeline produces PredictionResult | 2 tests pass | YES |
| run_models | Multiple models, correct sizes | 2 tests pass | YES |
| All families on classification | Each family produces valid predictions | 6 tests pass | YES |

---

## Deviations from Plan

### DEV-006: Single harness.py instead of separate configs.py and encoders.py

- **Type:** STRUCTURE_CHANGE (minor)
- **What changed:** The planned file structure had `src/models/harness.py`, `src/models/configs.py`, and `src/models/encoders.py`. Implementation puts everything in `harness.py`.
- **Why:** The total code is 459 lines — splitting into 3 files would create unnecessary import complexity. Will refactor if the file grows significantly.
- **Impact:** All imports come from `src.models.harness`. No impact on API.

---

## Completion Summary

TASK-07 delivered the Model Harness in `src/models/harness.py` (459 lines). The harness provides a unified encode → train → predict → decode pipeline supporting 8 model families (from majority-class baseline to gradient boosted trees and MLPs). All 33 V-5 tests pass, including a decision tree achieving perfect accuracy on the trivial C1.1 threshold task. One deviation: all model code is in a single file instead of the planned 3-file split. The harness is consumed by TASK-08 (Evaluation Engine) and TASK-09 (Experiment Runner).
