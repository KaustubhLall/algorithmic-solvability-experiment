# TASK-17 Implementation Log: Methodology Feedback Execution

- **Date:** 2026-03-26
- **Task:** TASK-17
- **Status:** COMPLETE
- **Branch:** `codex/task-17-methodology-feedback-execution`

## Goal

Execute the first methodology follow-up from TASK-16 by closing the verdict-wiring gap between baseline SR-8 reporting and diagnostic calibration, then refresh the rerun-backed artifacts and documentation so the repo is ready for the next second-paper improvement step.

## Inputs Reviewed

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md`
- `docs/PREPUBLICATION_ANALYSIS_2026-03-26.md`
- `src/reporting.py`
- `src/diagnostic_experiments.py`
- `tests/test_reporting.py`
- `tests/test_diagnostic_experiments.py`
- Existing diagnostic artifacts in `results/EXP-D1/`, `results/EXP-D2/`, and `results/EXP-D4/`

## Problem Statement

TASK-16 identified a methodology mismatch:

- baseline `solvability_verdicts.json` artifacts still hardcoded criteria 6 and 8 as unmet and only recognized criterion 7 when a distractor split was embedded in the same baseline run,
- EXP-D5 separately folded in D1/D2/D4 evidence,
- the paper-facing story therefore depended on whether one looked at the baseline report or the calibration pass.

That was a real prepublication issue because the repo had the evidence, but not one authoritative interpretation path.

## Implementation Summary

### 1. Centralized diagnostic evidence resolution in SR-8

Added shared helpers in `src/reporting.py`:

- `build_diagnostic_evidence_index(...)`
- `_select_alignment_model(...)`
- `resolve_task_diagnostic_support(...)`

These helpers:

- load EXP-D1 sample-efficiency evidence,
- load EXP-D2 distractor-robustness evidence,
- load EXP-D4 feature-alignment evidence,
- resolve that evidence per task and preferred best-model name,
- emit consistent criterion flags plus transparent diagnostic notes.

### 2. Wired baseline verdicts to EXP-D1 / EXP-D2 / EXP-D4

Updated `compute_solvability_verdict()` and `compute_solvability_verdicts()` so baseline verdicts can consume the diagnostic evidence index. `generate_report()` now points SR-8 at the results root automatically, which means baseline reruns pick up diagnostic evidence when those artifacts already exist.

This closed the specific TASK-16 gap:

- criterion 6 is now proxied from EXP-D4 feature-alignment precision@k,
- criterion 7 can now come from EXP-D2 summaries even when no distractor split exists in the baseline report,
- criterion 8 can now come from EXP-D1 sample-efficiency curves.

### 3. Aligned EXP-D5 to the same resolver

Updated `run_solvability_calibration_experiment()` in `src/diagnostic_experiments.py` to reuse the same diagnostic resolver instead of partially re-implementing the logic inline.

This means:

- baseline SR-8 reports and EXP-D5 now evaluate criteria 6-8 the same way,
- EXP-D5 notes and `diagnostic_support` fields are consistent with the baseline notes,
- D5 now behaves as a consistency/calibration check rather than a second competing label path.

### 4. Expanded regression coverage

Added tests for:

- external diagnostic evidence being consumed by baseline verdict logic,
- `generate_report()` auto-loading D1/D2/D4 from the results root,
- EXP-D5 wiring criterion 6 from EXP-D4 and surfacing the corresponding diagnostic support.

## Files Changed

- `src/reporting.py`
- `src/diagnostic_experiments.py`
- `tests/test_reporting.py`
- `tests/test_diagnostic_experiments.py`
- `docs/PROJECT_STATUS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/EXPERIMENT_CATALOG.md`
- `docs/METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md`
- `docs/PREPUBLICATION_ANALYSIS_2026-03-26.md`
- `docs/implementation_log/TASK-17_methodology_feedback_execution.md`

## Validation Performed

### Targeted validation

- `.\\.venv\\Scripts\\python.exe -m pytest tests/test_reporting.py tests/test_diagnostic_experiments.py`
- Result: `36 passed, 8 warnings`

### Full regression suite

- `.\\.venv\\Scripts\\python.exe -m pytest`
- Result: `463 passed, 17 warnings`

### Experiment reruns

- `.\\.venv\\Scripts\\python.exe main.py smoke --output-root results`
- `.\\.venv\\Scripts\\python.exe main.py classification --output-root results`
- `.\\.venv\\Scripts\\python.exe main.py sequence --output-root results`
- `.\\.venv\\Scripts\\python.exe main.py diagnostic --output-root results`
- `.\\.venv\\Scripts\\python.exe main.py bonus --output-root results`
- `.\\.venv\\Scripts\\python.exe scripts/generate_publication_assets.py`

All completed successfully. Diagnostics still finish in about `545s`, so the refactor did not reintroduce the earlier hang.

## Empirical Outcome

### Refreshed baseline verdict distribution

From the TASK-17 rerun:

- **Classification/control:** 2 `STRONG`, 9 `MODERATE`, 1 `WEAK`, 1 `INCONCLUSIVE`, 1 `NEGATIVE`
- **Sequence/control:** 12 `NEGATIVE`, 2 `WEAK`, 2 `INCONCLUSIVE`

New baseline `STRONG` tasks:

- `C1.1_numeric_threshold`
- `C2.6_categorical_gate`

### EXP-D5 outcome after alignment

After TASK-17, EXP-D5 shows:

- `0` upgrades
- `0` downgrades
- `30` unchanged tasks
- `controls_negative_or_weak = True`
- `trivial_tasks_strong = True`

That is the desired result. It means the baseline and calibrated paths now agree on the same evidence instead of drifting.

### Interpretation

TASK-17 improved the paper story in a meaningful but bounded way:

- it removed the strongest reporting objection from the first prepublication pass,
- it strengthened the baseline classification story by surfacing two true `STRONG` tasks directly in SR-8 artifacts,
- it did not change the central scientific bottleneck, which remains the weak sequence-learning stack.

## Decisions Captured

This task produced:

- `ADR-030` in `docs/ARCHITECTURE_DECISIONS.md`
- `DEV-019` in `docs/EXPERIMENT_CATALOG.md`

## Remaining Gaps / Next Task

TASK-17 completes Action 1.1 from the methodology plan. The next queued work is:

- **TASK-18 - Sequence Training Protocol Upgrade**

Priority items for TASK-18:

1. Increase LSTM epochs and add a learning-rate scheduler.
2. Log training curves/checkpoints so delayed generalization can be observed.
3. Re-run the sequence suite and check whether any currently negative tasks move into `WEAK` or better.

## Outcome

The repo now has one consistent verdicting path across baseline reporting and diagnostic calibration. The refreshed publication assets, status docs, and methodology review all reflect the same rerun-backed result: classification is stronger than sequence, two classification tasks are now baseline `STRONG`, and the next serious second-paper improvement has to come from sequence-model/training upgrades rather than from more verdict bookkeeping.
