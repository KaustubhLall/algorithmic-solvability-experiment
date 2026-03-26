# TASK-14: Diagnostic Experiments — Implementation Log

> **Date:** 2026-03-26
> **Status:** COMPLETE
> **Dependencies:** TASK-12 (Sequence Experiments), TASK-13 (Classification Experiments)
> **Branch:** `codex/task-14-diagnostic-experiments`

---

## Scope

Implement the five diagnostic experiments (EXP-D1 through EXP-D5) defined in `EXPERIMENT_CATALOG.md` Part 2. These experiments go beyond the baseline Phase 2/3 results to measure sample efficiency, distractor robustness, noise robustness, feature-importance alignment, and solvability verdict calibration.

---

## Deliverables

| Artifact | Path | Description |
|---|---|---|
| Diagnostic experiment module | `src/diagnostic_experiments.py` | 1271 lines. Five experiment runners + helpers. |
| CLI support | `main.py` | Added `diagnostic` command. |
| Test suite | `tests/test_diagnostic_experiments.py` | 21 tests covering all 5 experiments + utilities. |
| Harness extensions | `src/models/harness.py` | `InputEncoder.feature_names` property, `SklearnModelWrapper.estimator` property. |
| Architecture decisions | `docs/ARCHITECTURE_DECISIONS.md` | ADR-022 through ADR-025. |
| Deviation log entries | `docs/EXPERIMENT_CATALOG.md` | DEV-014, DEV-015. |

---

## Experiment Implementations

### EXP-D1: Sample Efficiency Comparison

**Function:** `run_sample_efficiency_experiment()`

- Loads baseline task/model pairings from existing `results/EXP-*/` artifacts via `collect_baseline_task_records()`.
- Generates learning curves across configurable sample sizes (default: 100, 500, 1000, 2000, 5000, 10000) with 5 seeds.
- Computes normalized AUC under the learning curve (log-scale x-axis) via `_curve_auc()`.
- Compares each task's AUC against control task AUC to produce a delta and criterion-8 pass/fail.
- Outputs: `sample_efficiency.json`, `config.json`, `summary.md`, `sequence_learning_curves.png`, `classification_learning_curves.png`.

**Task selection:** 6 benchmark tasks (`S1.4`, `S2.2`, `S3.1`, `C1.1`, `C2.3`, `C3.3`) + 2 controls (`S0.1`, `C0.1`).

### EXP-D2: Distractor Feature Robustness

**Function:** `run_distractor_robustness_experiment()`

- Uses `_clone_task_with_distractors()` to inject 0-20 irrelevant features into task schemas.
- Wraps the reference algorithm to filter out distractor features before computing labels.
- Trains 3 model families (decision tree, GBT, MLP) at each distractor count across 5 seeds.
- Measures accuracy drop and computes a robustness score (criterion-7 pass if drop ≤ 0.05).
- Outputs: `distractor_robustness.json`, `config.json`, `summary.md`, per-task `distractor_curve.png`.

**Task selection:** 5 classification tasks (`C1.1`, `C2.1`, `C2.6`, `C3.1`, `C3.5`).

### EXP-D3: Noise Robustness

**Function:** `run_noise_robustness_experiment()`

- Applies noise levels 0.0–0.2 to test inputs via `split_noise()` with schema-aware categorical perturbations.
- Trains 3 model families at each noise level across 5 seeds.
- Checks for smooth degradation (no spike > 0.05 between consecutive levels).
- Outputs: `noise_robustness.json`, `config.json`, `summary.md`, per-task `noise_curve.png`.

**Task selection:** 4 classification tasks (`C1.1`, `C2.1`, `C3.1`, `C3.5`).

### EXP-D4: Feature Importance Alignment

**Function:** `run_feature_importance_alignment_experiment()`

- Adds 5 distractor features to each task, trains sklearn models, and computes permutation importance.
- Uses `InputEncoder.feature_names` for column mapping and `SklearnModelWrapper.estimator` for sklearn access.
- Computes precision@k and Jaccard@k alignment between top-k important features and known relevant features.
- Outputs: `feature_importance_alignment.json`, `config.json`, `summary.md`, per-task `feature_alignment.png` (heatmap).

**Task selection:** 3 classification tasks (`C2.1`, `C2.6`, `C3.1`) with known relevant features.

### EXP-D5: Solvability Verdict Calibration

**Function:** `run_solvability_calibration_experiment()`

- Merges baseline evidence from TASK-12/13 with diagnostic evidence from D1-D4.
- Updates criterion-7 (distractor robustness) and criterion-8 (sample efficiency) flags.
- Uses `_calibrated_label()` to compute refined verdicts based on all available evidence.
- Applies calibrated score adjustment: `base_score + 0.10 * sample_score + 0.10 * distractor_score`.
- Validates: controls should be NEGATIVE/WEAK, trivial tasks should be STRONG.
- Outputs: `solvability_calibration.json`, `config.json`, `summary.md`.

---

## Key Design Decisions

| Decision | ADR | Summary |
|---|---|---|
| Reuse baseline artifacts | ADR-022 | Load from `results/EXP-*/` instead of re-running Phase 2/3. |
| Task-level distractor injection | ADR-023 | Augment `TaskSpec` schema instead of split-level injection. |
| Expose harness internals | ADR-024 | `feature_names` and `estimator` properties for diagnostic access. |
| Extracted label function | ADR-025 | `_calibrated_label()` is testable independently from the experiment loop. |

---

## Bug Fixes

- **DEV-015:** `np.trapz` removed in NumPy 2.0+. Fixed with `getattr(np, 'trapezoid', None) or np.trapz` fallback in `_curve_auc()`.

---

## Test Coverage

21 tests in `tests/test_diagnostic_experiments.py`:

| Test | What it covers |
|---|---|
| `test_collect_baseline_task_records_loads_best_model_config` | Baseline artifact loading and model config parsing. |
| `test_clone_task_with_distractors_preserves_labels` | Distractor injection preserves reference algorithm correctness. |
| `test_run_sample_efficiency_experiment_writes_artifacts` | EXP-D1 end-to-end with synthetic baseline artifacts. |
| `test_run_distractor_robustness_experiment_writes_artifacts` | EXP-D2 end-to-end with small config. |
| `test_run_solvability_calibration_promotes_strong_with_d1_and_d2` | EXP-D5 promotes MODERATE → STRONG when D1+D2 evidence added. |
| `test_curve_auc_monotone_perfect` | AUC = 1.0 for perfect flat curve. |
| `test_curve_auc_increasing` | AUC is reasonable for increasing curve. |
| `test_curve_auc_single_point` | Single-point edge case. |
| `test_curve_auc_length_mismatch` | Raises ValueError for mismatched inputs. |
| `test_calibrated_label_strong` | STRONG when all minimum + 2 optional criteria met. |
| `test_calibrated_label_moderate` | MODERATE when all minimum but <2 optional. |
| `test_calibrated_label_weak` | WEAK when criterion-1 passes but others fail. |
| `test_calibrated_label_negative_low_iid` | NEGATIVE for low IID accuracy. |
| `test_calibrated_label_inconclusive` | NEGATIVE/INCONCLUSIVE for mixed evidence. |
| `test_run_noise_robustness_experiment_writes_artifacts` | EXP-D3 end-to-end artifact creation. |
| `test_noise_robustness_per_task_plot_created` | EXP-D3 per-task plot files exist. |
| `test_run_feature_importance_alignment_writes_artifacts` | EXP-D4 end-to-end artifact creation. |
| `test_feature_importance_alignment_per_task_plot_created` | EXP-D4 per-task heatmap files exist. |
| `test_feature_importance_precision_at_k_nonzero` | EXP-D4 precision@k is > 0 for a simple task. |
| `test_clone_task_with_zero_distractors_keeps_original_schema` | Zero-distractor edge case. |
| `test_solvability_calibration_controls_are_negative` | EXP-D5 control tasks remain NEGATIVE/WEAK. |

---

## Files Changed

| File | Change |
|---|---|
| `src/diagnostic_experiments.py` | **NEW** — 1271 lines, all 5 experiment runners. |
| `tests/test_diagnostic_experiments.py` | **NEW** — 21 tests. |
| `main.py` | Added `diagnostic` CLI command and import. |
| `src/models/harness.py` | Added `InputEncoder.feature_names` property (+4 lines), `SklearnModelWrapper.estimator` property (+4 lines). |
| `docs/PROJECT_STATUS.md` | Updated for TASK-14 completion. |
| `docs/IMPLEMENTATION_LOG_SUMMARY.md` | Added TASK-14 entry + lessons learned. |
| `docs/ARCHITECTURE_DECISIONS.md` | Added ADR-022 through ADR-025. |
| `docs/EXPERIMENT_CATALOG.md` | Added DEV-014, DEV-015 to deviation log. |

---

## Acceptance Criteria

| Criterion | Status |
|---|---|
| All diagnostic experiments (EXP-D1 through EXP-D5) complete without error | ✅ |
| Solvability scores are well-calibrated (controls WEAK/NEGATIVE, trivial tasks STRONG) | ✅ |
| Results written to `results/EXP-D{1-5}/` directories | ✅ (test-verified) |
| Learning curves, distractor/noise degradation plots generated | ✅ (test-verified) |
| Feature importance comparisons generated | ✅ (test-verified) |
| Solvability calibration report generated | ✅ (test-verified) |
| 21 tests pass | ✅ |
| No regressions in existing test suite | ✅ (437 pass, 2 pre-existing torch failures) |
