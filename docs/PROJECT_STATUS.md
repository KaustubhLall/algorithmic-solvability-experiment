# PROJECT STATUS: Algorithmic Solvability Experiment

> **PURPOSE:** This is the single entry-point document for any new implementation chat.
> Read this first, then follow the links to detailed documents.
> Update this document at the end of every chat session.
>
> **Last Updated:** 2026-03-27 (TASK-20 complete)
> **Current Phase:** Methodology execution is underway; the classification baseline-separation repair is complete, and baseline distractor-split support is now the next queued credibility upgrade

---

## Quick Context

**Goal:** Determine whether ML models can detect that a task is governed by a compact deterministic algorithm, by training on synthetic input/output pairs and testing for systematic generalization beyond the training distribution.

**Two task tracks:**
- **Sequence track (S-tiers):** Variable-length discrete token sequences -> transformed sequences
- **Classification track (C-tiers):** Mixed numerical + categorical tabular inputs -> categorical label

**Two authoritative design documents:**
- `docs/EXPERIMENT_DESIGN.md` - *what* to build and *why* (task tiers, models, metrics, evidence criteria)
- `docs/EXPERIMENT_CATALOG.md` - *how* to build it (shared resources SR-1-10, experiments, validation V-1-10, execution plan TASK-01-21, deviation log)

---

## Current Task

| Field | Value |
|---|---|
| **Next task to implement** | TASK-21 - Baseline Distractor Split Support (Phase 1.4 follow-up after TASK-20 repair) |
| **Status** | READY |
| **Blocked by** | Nothing; TASK-20 resolved the last classification baseline-separation edge case, refreshed the affected experiment stack, and regenerated publication assets |
| **Relevant spec** | `docs/METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md`, `docs/PREPUBLICATION_ANALYSIS_2026-03-26.md`, the refreshed publication assets in `output/publication_assets/`, the second-paper manuscript in `output/pdf/`, and the TASK-18/TASK-20 implementation logs |

---

## Implementation Progress

| Task | Scope | Status | Notes |
|---|---|---|---|
| TASK-01 | Input Schema (SR-2) | **COMPLETE** | 54 V-2 tests pass. See [log](implementation_log/TASK-01_input_schema.md) |
| TASK-02 | Classification Rule DSL (SR-9) | **COMPLETE** | 58 V-9 tests pass. See [log](implementation_log/TASK-02_classification_dsl.md) |
| TASK-03 | Sequence DSL (SR-10) | **COMPLETE** | 57 V-10 tests pass. See [log](implementation_log/TASK-03_sequence_dsl.md) |
| TASK-04 | Task Registry (SR-1) | **COMPLETE** | 36 V-1 tests pass. 32 tasks registered. See [log](implementation_log/TASK-04_task_registry.md) |
| TASK-05 | Data Generator (SR-3) | **COMPLETE** | 24 V-3 tests pass. See [log](implementation_log/TASK-05_data_generator.md) |
| TASK-06 | Split Generator (SR-4) | **COMPLETE** | 33 V-4 tests pass. See [log](implementation_log/TASK-06_split_generator.md) |
| TASK-07 | Model Harness (SR-5) | **COMPLETE** | 38 V-5 tests pass. 9 model families including a raw-sequence LSTM path. See [log](implementation_log/TASK-07_model_harness.md) |
| TASK-08 | Evaluation Engine (SR-6) | **COMPLETE** | 53 V-6 tests pass. See [log](implementation_log/TASK-08_evaluation_engine.md) |
| TASK-09 | Experiment Runner (SR-7) | **COMPLETE** | 36 V-7 tests pass. Multi-seed orchestration, aggregation, and serialization helpers in `src/runner.py`. See [log](implementation_log/TASK-09_experiment_runner.md) |
| TASK-10 | Report Generator (SR-8) | **COMPLETE** | 14 V-8 tests pass. Structured artifacts, plots, markdown summaries, and solvability verdict logic in `src/reporting.py`. See [log](implementation_log/TASK-10_report_generator.md) |
| TASK-11 | Smoke Tests (EXP-0.x) | **COMPLETE** | **GATE CLEARED.** `src/smoke_tests.py`, `main.py`, 7 V-Global tests, and `results/EXP-0.1` through `results/EXP-0.3` artifacts. LSTM reaches 90.5% exact match on bounded sort; C1.1 smoke is MODERATE with perfect tree/logistic accuracy; control tasks are NEGATIVE. See [log](implementation_log/TASK-11_smoke_tests.md) |
| TASK-12 | Sequence Experiments | **COMPLETE** | Added `src/sequence_experiments.py`, CLI support in `main.py`, `tests/test_sequence_experiments.py`, refreshed smoke-compatible LSTM handling for unseen tokens, and generated `results/EXP-S1` through `results/EXP-S3`. After the TASK-18 rerun, the implemented sequence/control benchmark stands at 11 `NEGATIVE`, 1 `MODERATE`, 2 `WEAK`, and 2 `INCONCLUSIVE`, with `S1.4_count_symbol` promoted to `MODERATE` and `S2.3_running_min` promoted to `INCONCLUSIVE`. See [log](implementation_log/TASK-12_sequence_experiments.md) |
| TASK-13 | Classification Experiments | **COMPLETE** | Added `src/classification_experiments.py`, CLI support in `main.py`, `tests/test_classification_experiments.py`, schema-guided categorical noise for tabular `NOISE` splits, and generated `results/EXP-C1` through `results/EXP-C3`. After the TASK-20 rerun, the implemented classification/control benchmark stands at 3 `STRONG`, 9 `MODERATE`, 1 `INCONCLUSIVE`, and 1 `NEGATIVE`, with `C2.1_and_rule` promoted from `WEAK` to `STRONG` after repairing its class-prior skew. See [log](implementation_log/TASK-13_classification_experiments.md) |
| TASK-14 | Diagnostic Experiments | **COMPLETE** | Added `src/diagnostic_experiments.py`, CLI support in `main.py`, `tests/test_diagnostic_experiments.py` (22 tests), `InputEncoder.feature_names` and `SklearnModelWrapper.estimator` properties. EXP-D1 through EXP-D5 runners implemented: sample-efficiency learning curves, distractor robustness, noise robustness, feature-importance alignment, and solvability verdict calibration combining baseline + diagnostic evidence. See [log](implementation_log/TASK-14_diagnostic_experiments.md) |
| TASK-15 | Bonus: Algorithm Discovery | **COMPLETE** | Added `src/bonus_experiments.py`, CLI support in `main.py`, `tests/test_bonus_experiments.py` (20 tests). EXP-B1 extracts decision tree rules from classification models and evaluates functional equivalence on hard test sets. EXP-B2 searches over SR-10 DSL programs for sequence tasks using the reference algorithm as oracle. See [log](implementation_log/TASK-15_bonus_algorithm_discovery.md) |
| TASK-16 | Methodology Feedback Planning | **COMPLETE** | Cross-checked the methodology review against the fresh rerun, `PREPUBLICATION_ANALYSIS`, and the compiled preprint; corrected stale counts and claims; documented a prioritized Phase 1-3 follow-up plan; and updated the status/catalog/ADR docs so the next task starts from the implemented `S0-S3` and `C0-C3` scope rather than the aspirational roadmap. See [log](implementation_log/TASK-16_methodology_feedback_planning.md) |
| TASK-17 | Methodology Feedback Execution | **COMPLETE** | Wired EXP-D1/EXP-D2/EXP-D4 evidence into the baseline SR-8 verdict path, aligned EXP-D5 to the same resolver, added regression coverage in `tests/test_reporting.py` and `tests/test_diagnostic_experiments.py`, reran the full `smoke`/`classification`/`sequence`/`diagnostic`/`bonus` stack, and refreshed `output/publication_assets/`. Baseline reporting now yields two classification `STRONG` tasks and EXP-D5 is a consistency check rather than the only place `STRONG` appears. See [log](implementation_log/TASK-17_methodology_feedback_execution.md) |
| TASK-18 | Sequence Training Protocol Upgrade | **COMPLETE** | Upgraded the LSTM sequence protocol to 5 seeds and 1000-sample `EXP-S1`/`EXP-S2` runs, added weight decay + `ReduceLROnPlateau`, logged per-epoch training curves into SR-7/SR-8 artifacts, reran `sequence`, `diagnostic`, `bonus`, `smoke`, and `classification`, and regenerated `output/publication_assets/` with full 16/16 runtime coverage plus a new sequence-training-dynamics figure. Sequence/control results improved to 11 `NEGATIVE`, 1 `MODERATE`, 2 `WEAK`, and 2 `INCONCLUSIVE`. See [log](implementation_log/TASK-18_sequence_training_protocol_upgrade.md) |
| TASK-19 | Methodology Synthesis and Second-Paper Draft | **COMPLETE** | Rewrote the methodology review and publication-facing analysis around the fresh TASK-18 artifacts, converted the refreshed evidence bundle into a submission-style second-paper manuscript with training-dynamics figures, compiled the PDF, rendered page images for QA, and updated the task/status/catalog docs so the next execution task starts from the current rerun-backed narrative rather than from stale prepublication text. See [log](implementation_log/TASK-19_methodology_synthesis_second_paper.md) |
| TASK-20 | Classification Baseline Separation Repair | **COMPLETE** | Added a task-local balanced sampler for `C2.1_and_rule` in `src/registry.py`, locked it in with new registry/data-generator coverage, reran `classification`, `diagnostic`, and `bonus`, regenerated `output/publication_assets/`, and updated the paper-facing docs so classification now stands at 3 `STRONG`, 9 `MODERATE`, 1 `INCONCLUSIVE`, and 1 `NEGATIVE`. See [log](implementation_log/TASK-20_classification_baseline_separation_repair.md) |

**Milestone Gates:**
- `[x]` FOUNDATION complete (TASK-01-04 done + V-1, V-2, V-9, V-10 passing)
- `[x]` DATA PIPELINE complete (TASK-05-06 done + V-3, V-4 passing)
- `[x]` FULL PIPELINE complete (TASK-07-10 done + V-5 through V-8 passing)
- `[x]` SMOKE TEST GATE cleared (TASK-11 done + V-G1-G4 passing)
- `[x]` SEQUENCE BASELINE SUITE complete (TASK-12 done for implemented S1-S3 tiers + `results/EXP-S1` through `results/EXP-S3`)
- `[x]` CLASSIFICATION BASELINE SUITE complete (TASK-13 done for implemented C1-C3 tiers + `results/EXP-C1` through `results/EXP-C3`)
- `[x]` DIAGNOSTIC SUITE complete (TASK-14 done for EXP-D1 through EXP-D5 + 22 tests passing)
- `[x]` BONUS ALGORITHM DISCOVERY complete (TASK-15 done for EXP-B1 + EXP-B2 + 20 tests passing)
- `[x]` METHODOLOGY FEEDBACK PLAN complete (TASK-16 done; review docs and publication-facing claims reconciled to the latest rerun/preprint)
- `[x]` METHODOLOGY FEEDBACK EXECUTION 1.1 complete (TASK-17 wired criteria 6-8 into baseline reporting, aligned EXP-D5, reran affected suites, and refreshed publication assets)
- `[x]` METHODOLOGY FEEDBACK EXECUTION 1.2 complete (TASK-18 upgraded the sequence LSTM protocol, logged training curves, refreshed the affected experiment stack, and regenerated publication assets with complete runtime coverage)
- `[x]` METHODOLOGY FEEDBACK EXECUTION 1.3 complete (TASK-20 repaired `C2.1_and_rule` label balance, reran the affected classification/diagnostic/bonus stack, and removed the last `WEAK` classification verdict)
- `[x]` SECOND-PAPER SYNTHESIS complete (TASK-19 rewrote the methodology-facing docs, produced a manuscript sourced from the refreshed asset bundle, and QA'd the compiled PDF via rendered pages plus page-structure checks)

---

## Actual File Structure

```text
DataScience/
|-- docs/
|   |-- EXPERIMENT_DESIGN.md
|   |-- EXPERIMENT_CATALOG.md
|   |-- PROJECT_STATUS.md
|   |-- IMPLEMENTATION_LOG_SUMMARY.md
|   |-- ARCHITECTURE_DECISIONS.md
|   |-- METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md
|   |-- PREPUBLICATION_ANALYSIS_2026-03-26.md
|   `-- implementation_log/
|       |-- TASK-01_input_schema.md
|       |-- TASK-02_classification_dsl.md
|       |-- TASK-03_sequence_dsl.md
|       |-- TASK-04_task_registry.md
|       |-- TASK-05_data_generator.md
|       |-- TASK-06_split_generator.md
|       |-- TASK-07_model_harness.md
|       |-- TASK-08_evaluation_engine.md
|       |-- TASK-09_experiment_runner.md
|       |-- TASK-10_report_generator.md
|       |-- TASK-11_smoke_tests.md
|       |-- TASK-12_sequence_experiments.md
|       |-- TASK-13_classification_experiments.md
|       |-- TASK-14_diagnostic_experiments.md
|       |-- TASK-15_bonus_algorithm_discovery.md
|       |-- TASK-16_methodology_feedback_planning.md
|       |-- TASK-17_methodology_feedback_execution.md
|       |-- TASK-18_sequence_training_protocol_upgrade.md
|       |-- TASK-19_methodology_synthesis_second_paper.md
|       `-- TASK-20_classification_baseline_separation_repair.md
|-- src/
|   |-- __init__.py
|   |-- schemas.py                    # SR-2 built
|   |-- registry.py                   # SR-1 built (32 tasks)
|   |-- data_generator.py             # SR-3 built
|   |-- splits.py                     # SR-4 built
|   |-- evaluation.py                 # SR-6 built
|   |-- runner.py                     # SR-7 built
|   |-- reporting.py                  # SR-8 built
|   |-- smoke_tests.py                # TASK-11 smoke experiment specs + runners
|   |-- sequence_experiments.py       # TASK-12 sequence experiment specs + runners
|   |-- classification_experiments.py # TASK-13 classification experiment specs + runners
|   |-- diagnostic_experiments.py     # TASK-14 diagnostic experiment specs + runners
|   |-- bonus_experiments.py          # TASK-15 bonus algorithm discovery runners
|   |-- dsl/
|   |   |-- __init__.py
|   |   |-- classification_dsl.py     # SR-9 built
|   |   `-- sequence_dsl.py           # SR-10 built
|   `-- models/
|       |-- __init__.py
|       `-- harness.py                # SR-5 built (9 families)
|-- tests/                            # Validation suite (465 tests total)
|   |-- __init__.py
|   |-- test_schemas.py               # V-2: 54 tests
|   |-- test_classification_dsl.py    # V-9: 58 tests
|   |-- test_sequence_dsl.py          # V-10: 57 tests
|   |-- test_registry.py              # V-1: 36 tests
|   |-- test_data_generator.py        # V-3: 24 tests
|   |-- test_splits.py                # V-4: 33 tests
|   |-- test_model_harness.py         # V-5: 38 tests
|   |-- test_evaluation.py            # V-6: 53 tests
|   |-- test_runner.py                # V-7: 36 tests
|   |-- test_reporting.py             # V-8: 14 tests
|   |-- test_smoke_tests.py           # V-G1..V-G4: 7 tests
|   |-- test_sequence_experiments.py  # TASK-12 sequence experiment coverage
|   |-- test_classification_experiments.py # TASK-13 classification experiment coverage
|   |-- test_diagnostic_experiments.py    # TASK-14 diagnostic experiment coverage (22 tests)
|   `-- test_bonus_experiments.py         # TASK-15 bonus experiment coverage (20 tests)
|-- results/
|-- output/
|   |-- pdf/
|   `-- publication_assets/
|-- scripts/
|-- conftest.py
|-- requirements.txt
`-- main.py
```

---

## Known Deviations from Plan

- **DEV-001:** Added `Distribution` and `ElementType` enums (not in original SR-2 spec) for type safety.
- **DEV-002:** Used tuples instead of lists for frozen dataclass compatibility.
- **DEV-003:** Sequence DSL reducers return `[int]` instead of `int` for uniform composability.
- **DEV-004:** S4/S5/C4/C5 task tiers deferred - only S0-S3 and C0-C3 registered initially.
- **DEV-005:** DistractorSplit defined in enum but not yet implemented.
- **DEV-006:** Single `harness.py` instead of planned `harness.py` + `configs.py` + `encoders.py`.
- **DEV-007:** `ExperimentSpec` uses `n_samples` + `train_fraction` rather than separate train/test sample counts.
- **DEV-008:** Solvability verdicts are operationalized from currently available SR-7 metrics and, after TASK-17, can also absorb EXP-D1/EXP-D2/EXP-D4 evidence for criteria 6-8 when those artifacts exist alongside the baseline results; criterion 9 remains unmet until new transfer splits land.
- **DEV-009:** TASK-11 classification smoke calibration expands EXP-0.2 beyond the original single-seed IID spec so the current verdict logic can evaluate baseline separation, extrapolation, and seed stability; under the current operationalization, V-G3 is satisfied by `MODERATE` or better rather than requiring `STRONG`.
- **DEV-010:** TASK-12 executes the currently implemented sequence tiers (`EXP-S1` through `EXP-S3`) and keeps `EXP-S4`/`EXP-S5` deferred until their registry tasks and split strategies exist.
- **DEV-011:** TASK-12 sequence runs use the currently validated model families (`majority_class`, `sequence_baseline`, `mlp`, `lstm`) and replace EXP-S3's cataloged composition split with the available value-range extrapolation split until transformer/composition support lands.
- **DEV-012:** TASK-13 executes the currently implemented classification tiers (`EXP-C1` through `EXP-C3`) and keeps the remaining catalog-only classification tasks plus `EXP-C4`/`EXP-C5` deferred until their registry coverage exists.
- **DEV-013:** TASK-13 classification runs use the currently validated model families (`majority_class`, `logistic_regression`, `decision_tree`, `random_forest`, `gradient_boosted_trees`, `mlp`) and the available OOD splits (`value_extrapolation`, `noise`) instead of the catalog's broader split/architecture matrix.
- **DEV-014:** TASK-14 selects D1 task/model pairings from existing baseline artifacts rather than requiring a live Phase 2/3 rerun. D2 uses `_clone_task_with_distractors` to inject schema-aware distractor features at the task level. D3 noise robustness runs classification tasks only (numeric Gaussian noise). D4 feature-importance alignment uses `sklearn.inspection.permutation_importance`. D5 calibration combines baseline evidence with D1-D4 diagnostic signals and uses a refined `_calibrated_label` function.
- **DEV-015:** `np.trapz` replaced with `np.trapezoid` (NumPy 2.0+ compat) in `_curve_auc`.
- **DEV-016:** TASK-15 selects classification tasks with MODERATE or better verdicts for EXP-B1 rule extraction (spec says STRONG, but current suite peaks at MODERATE). EXP-B2 searches all implemented S1/S3 sequence tasks rather than limiting to S5-tier STRONG tasks (S5 is deferred). Random DSL program search replaces model-guided search since the reference algorithm itself is used as the oracle.
- **DEV-017:** TASK-15 uses `InputEncoder` + `DecisionTreeClassifier` directly for EXP-B1 rather than going through the full `ModelHarness.run()` pipeline, to access the fitted tree structure for rule extraction and structural comparison.
- **DEV-018:** TASK-16 introduces an explicit post-implementation methodology/planning phase. Publication-facing documents are now required to stay scoped to the implemented `S0-S3` and `C0-C3` benchmark and to be cross-checked against the latest rerun-backed analysis before new execution tasks are queued.
- **DEV-019:** TASK-17 centralizes diagnostic verdict wiring in SR-8 and EXP-D5. Baseline reports now auto-load EXP-D1/EXP-D2/EXP-D4 artifacts from the results root when present, so the refreshed baseline and calibrated labels agree on the current rerun.
- **DEV-020:** TASK-18 upgrades the executed sequence protocol beyond the original TASK-12 baseline: `EXP-S1` and `EXP-S2` now run with 5 seeds and 1000 samples, the LSTM trains for 200 epochs with weight decay and `ReduceLROnPlateau`, checkpoint selection uses an internal validation slice, and per-epoch held-out monitoring curves are stored in `training_curve`.
- **DEV-021:** TASK-19 adds an explicit second-paper synthesis task. Publication-facing markdown analysis, manuscript source, compiled PDF, and figure QA are now tracked deliverables rather than ad hoc outputs.
- **DEV-022:** TASK-20 repairs `C2.1_and_rule` baseline separation by changing the task-local input sampler instead of weakening the SR-8 baseline-gap criterion. The rule is unchanged; only the sampled class prior is rebalanced so majority-class performance no longer masks learnability.

See `EXPERIMENT_CATALOG.md` Part 5 (Deviation Log) for structured entries.

---

## How to Continue in a New Chat

Paste the following as your first message in a new implementation chat:

```text
Continue implementing the algorithmic solvability experiment from where we left off.

Read these documents in order:
1. docs/PROJECT_STATUS.md
2. docs/ARCHITECTURE_DECISIONS.md
3. docs/IMPLEMENTATION_LOG_SUMMARY.md
4. docs/EXPERIMENT_CATALOG.md
5. docs/EXPERIMENT_DESIGN.md

Check the Deviation Log in EXPERIMENT_CATALOG.md Part 5 before starting.
Then implement the next incomplete task shown in PROJECT_STATUS.md.
After completing the task, update PROJECT_STATUS.md, IMPLEMENTATION_LOG_SUMMARY.md,
and write a detailed log in docs/implementation_log/TASK-XX_<name>.md.
```
