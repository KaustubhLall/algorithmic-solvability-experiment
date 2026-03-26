# PROJECT STATUS: Algorithmic Solvability Experiment

> **PURPOSE:** This is the single entry-point document for any new implementation chat.
> Read this first, then follow the links to detailed documents.
> Update this document at the end of every chat session.
>
> **Last Updated:** 2026-03-26
> **Current Phase:** Implementation - TASK-12 complete (sequence baseline suite landed, TASK-13 next)

---

## Quick Context

**Goal:** Determine whether ML models can detect that a task is governed by a compact deterministic algorithm, by training on synthetic input/output pairs and testing for systematic generalization beyond the training distribution.

**Two task tracks:**
- **Sequence track (S-tiers):** Variable-length discrete token sequences -> transformed sequences
- **Classification track (C-tiers):** Mixed numerical + categorical tabular inputs -> categorical label

**Two authoritative design documents:**
- `docs/EXPERIMENT_DESIGN.md` - *what* to build and *why* (task tiers, models, metrics, evidence criteria)
- `docs/EXPERIMENT_CATALOG.md` - *how* to build it (shared resources SR-1-10, experiments, validation V-1-10, execution plan TASK-01-15, deviation log)

---

## Current Task

| Field | Value |
|---|---|
| **Next task to implement** | TASK-13: Classification Experiments |
| **Status** | READY TO START |
| **Blocked by** | Nothing - TASK-12 shipped and validation is green |
| **Relevant spec** | `EXPERIMENT_CATALOG.md` Part 2 (EXP-C1 to EXP-C5), Part 4 (TASK-13) |

---

## Implementation Progress

| Task | Scope | Status | Notes |
|---|---|---|---|
| TASK-01 | Input Schema (SR-2) | **COMPLETE** | 54 V-2 tests pass. See [log](implementation_log/TASK-01_input_schema.md) |
| TASK-02 | Classification Rule DSL (SR-9) | **COMPLETE** | 58 V-9 tests pass. See [log](implementation_log/TASK-02_classification_dsl.md) |
| TASK-03 | Sequence DSL (SR-10) | **COMPLETE** | 57 V-10 tests pass. See [log](implementation_log/TASK-03_sequence_dsl.md) |
| TASK-04 | Task Registry (SR-1) | **COMPLETE** | 35 V-1 tests pass. 28 tasks registered. See [log](implementation_log/TASK-04_task_registry.md) |
| TASK-05 | Data Generator (SR-3) | **COMPLETE** | 23 V-3 tests pass. See [log](implementation_log/TASK-05_data_generator.md) |
| TASK-06 | Split Generator (SR-4) | **COMPLETE** | 33 V-4 tests pass. See [log](implementation_log/TASK-06_split_generator.md) |
| TASK-07 | Model Harness (SR-5) | **COMPLETE** | 38 V-5 tests pass. 9 model families including a raw-sequence LSTM path. See [log](implementation_log/TASK-07_model_harness.md) |
| TASK-08 | Evaluation Engine (SR-6) | **COMPLETE** | 53 V-6 tests pass. See [log](implementation_log/TASK-08_evaluation_engine.md) |
| TASK-09 | Experiment Runner (SR-7) | **COMPLETE** | 36 V-7 tests pass. Multi-seed orchestration, aggregation, and serialization helpers in `src/runner.py`. See [log](implementation_log/TASK-09_experiment_runner.md) |
| TASK-10 | Report Generator (SR-8) | **COMPLETE** | 9 V-8 tests pass. Structured artifacts, plots, markdown summaries, and solvability verdict logic in `src/reporting.py`. See [log](implementation_log/TASK-10_report_generator.md) |
| TASK-11 | Smoke Tests (EXP-0.x) | **COMPLETE** | **GATE CLEARED.** `src/smoke_tests.py`, `main.py`, 7 V-Global tests, and `results/EXP-0.1` through `results/EXP-0.3` artifacts. LSTM reaches 90.5% exact match on bounded sort; C1.1 smoke is MODERATE with perfect tree/logistic accuracy; control tasks are NEGATIVE. See [log](implementation_log/TASK-11_smoke_tests.md) |
| TASK-12 | Sequence Experiments | **COMPLETE** | Added `src/sequence_experiments.py`, CLI support in `main.py`, `tests/test_sequence_experiments.py`, refreshed smoke-compatible LSTM handling for unseen tokens, and generated `results/EXP-S1` through `results/EXP-S3`. Current sequence verdicts are mostly NEGATIVE, with WEAK evidence on `S1.4_count_symbol` and `S2.2_balanced_parens`. See [log](implementation_log/TASK-12_sequence_experiments.md) |
| TASK-13 | Classification Experiments | NOT STARTED | Depends on TASK-11 |
| TASK-14 | Diagnostic Experiments | NOT STARTED | Depends on TASK-12-13 |
| TASK-15 | Bonus: Algorithm Discovery | NOT STARTED | Depends on TASK-14 |

**Milestone Gates:**
- `[x]` FOUNDATION complete (TASK-01-04 done + V-1, V-2, V-9, V-10 passing)
- `[x]` DATA PIPELINE complete (TASK-05-06 done + V-3, V-4 passing)
- `[x]` FULL PIPELINE complete (TASK-07-10 done + V-5 through V-8 passing)
- `[x]` SMOKE TEST GATE cleared (TASK-11 done + V-G1-G4 passing)
- `[x]` SEQUENCE BASELINE SUITE complete (TASK-12 done for implemented S1-S3 tiers + `results/EXP-S1` through `results/EXP-S3`)

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
|       `-- TASK-11_smoke_tests.md
|-- src/
|   |-- __init__.py
|   |-- schemas.py                    # SR-2 built
|   |-- registry.py                   # SR-1 built (28 tasks)
|   |-- data_generator.py             # SR-3 built
|   |-- splits.py                     # SR-4 built
|   |-- evaluation.py                 # SR-6 built
|   |-- runner.py                     # SR-7 built
|   |-- reporting.py                  # SR-8 built
|   |-- smoke_tests.py                # TASK-11 smoke experiment specs + runners
|   |-- sequence_experiments.py       # TASK-12 sequence experiment specs + runners
|   |-- dsl/
|   |   |-- __init__.py
|   |   |-- classification_dsl.py     # SR-9 built
|   |   `-- sequence_dsl.py           # SR-10 built
|   `-- models/
|       |-- __init__.py
|       `-- harness.py                # SR-5 built (9 families)
|-- tests/                            # Validation suite (403 tests total)
|   |-- __init__.py
|   |-- test_schemas.py               # V-2: 54 tests
|   |-- test_classification_dsl.py    # V-9: 58 tests
|   |-- test_sequence_dsl.py          # V-10: 57 tests
|   |-- test_registry.py              # V-1: 35 tests
|   |-- test_data_generator.py        # V-3: 23 tests
|   |-- test_splits.py                # V-4: 33 tests
|   |-- test_model_harness.py         # V-5: 38 tests
|   |-- test_evaluation.py            # V-6: 53 tests
|   |-- test_runner.py                # V-7: 36 tests
|   |-- test_reporting.py             # V-8: 9 tests
|   `-- test_smoke_tests.py           # V-G1..V-G4: 7 tests
|   `-- test_sequence_experiments.py  # TASK-12 sequence experiment coverage
|-- results/
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
- **DEV-008:** Solvability verdicts are operationalized from currently available SR-7 metrics (IID/OOD accuracy, baseline gap, seed stability, robustness splits) while unavailable evidence criteria remain explicitly marked unmet.
- **DEV-009:** TASK-11 classification smoke calibration expands EXP-0.2 beyond the original single-seed IID spec so the current verdict logic can evaluate baseline separation, extrapolation, and seed stability; under the current operationalization, V-G3 is satisfied by `MODERATE` or better rather than requiring `STRONG`.
- **DEV-010:** TASK-12 executes the currently implemented sequence tiers (`EXP-S1` through `EXP-S3`) and keeps `EXP-S4`/`EXP-S5` deferred until their registry tasks and split strategies exist.
- **DEV-011:** TASK-12 sequence runs use the currently validated model families (`majority_class`, `sequence_baseline`, `mlp`, `lstm`) and replace EXP-S3's cataloged composition split with the available value-range extrapolation split until transformer/composition support lands.

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
