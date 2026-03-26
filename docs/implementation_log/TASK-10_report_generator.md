# TASK-10: Report Generator

> **Status:** COMPLETE
> **Builds:** SR-8
> **Validation:** V-8
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-8), Part 3 (V-8), Part 4 (TASK-10)
> **Date Started:** 2026-03-25
> **Date Completed:** 2026-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| Structured output directory writer for `results/{experiment_id}/...` | `src/reporting.py` | COMPLETE |
| `config.json` export from the experiment spec | `src/reporting.py` | COMPLETE |
| Human-readable `summary.md` | `src/reporting.py` | COMPLETE |
| Per-task `metrics.json` and `errors.json` | `src/reporting.py` | COMPLETE |
| `confusion.png` generation for classification tasks | `src/reporting.py` | COMPLETE |
| `extrap_curve.png` generation from aggregated split metrics | `src/reporting.py` | COMPLETE |
| Cross-task `comparison.md` | `src/reporting.py` | COMPLETE |
| `solvability_verdicts.json` with Section 9.4 labels | `src/reporting.py` | COMPLETE |
| Validation suite for V-8 | `tests/test_reporting.py` | COMPLETE |

---

## Files Changed

- `src/reporting.py`
- `tests/test_reporting.py`
- `docs/PROJECT_STATUS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/EXPERIMENT_CATALOG.md`
- `docs/implementation_log/TASK-10_report_generator.md`

---

## Implementation Notes

### 1. Reporting consumes SR-7/SR-6 serializers instead of re-encoding the dataclasses

`generate_report()` uses `experiment_report_to_dict()`, `aggregated_result_to_dict()`, and `single_result_to_dict()` as the source of truth for JSON payloads. This avoids drift between the runner/evaluator dataclasses and the artifact schema, and made V-8 JSON validation straightforward.

### 2. Solvability verdicts needed an explicit operationalization layer

Section 9.4 defines the labels qualitatively in terms of evidence criteria 1-9. The current pipeline directly exposes only part of that evidence. SR-8 therefore maps currently available signals into verdicts:

- Criterion 1: high IID accuracy from aggregated IID results
- Criterion 2: OOD success from non-IID aggregated results
- Criterion 3: baseline separation from the floor baselines currently available in the harness (`majority_class`, `sequence_baseline`)
- Criterion 4: seed stability from aggregated standard deviation and seed count
- Criterion 5: coherent degradation from split-wise accuracy behavior
- Criteria 6-9: only marked true when directly evidenced by the available split results; otherwise explicitly false

The resulting artifact includes both the verdict label and the per-criterion evidence flags, plus notes explaining why evidence is missing or weak.

### 3. Plot generation is report-driven

No raw dataset replay was needed:

- `confusion.png` is built by averaging stored confusion matrices across matching single-run results
- `extrap_curve.png` is built from aggregated accuracy-by-split values per model

This keeps SR-8 lightweight and deterministic.

### 4. Per-task artifact payloads were designed for downstream inspection

Each task directory now contains:

- `metrics.json`: task metadata, verdict, aggregated results, single results
- `errors.json`: grouped error taxonomies by `(model, split)` with total and mean counts
- `confusion.png`: only for classification tasks
- `extrap_curve.png`: all tasks

This should make TASK-11 and the later experiment phases much easier to inspect without re-running code.

---

## Validation Work

### V-8 test coverage added

`tests/test_reporting.py` adds 9 tests covering:

1. expected output tree creation
2. JSON parseability for all required JSON files
3. markdown summary readability and consistency with `metrics.json`
4. comparison markdown table generation
5. solvability verdict labeling for all five labels:
   - `STRONG`
   - `MODERATE`
   - `WEAK`
   - `NEGATIVE`
   - `INCONCLUSIVE`

### Validation commands run

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_reporting.py
.\.venv\Scripts\python.exe -m pytest
```

### Validation results

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| File structure | All required TASK-10 artifacts are created | `config.json`, `summary.md`, `comparison.md`, `solvability_verdicts.json`, per-task JSON/plots all created in tests | YES |
| JSON validity | All JSON files parse without error | All reporting JSON artifacts parsed successfully | YES |
| Markdown validity | Summary renders as readable markdown | Summary and comparison markdown contain human-readable tables and per-task verdict sections | YES |
| Metric consistency | Summary values match JSON payloads | Best IID/OOD metrics and verdict labels are asserted against `metrics.json` in V-8 tests | YES |
| Solvability verdict logic | Labels match Section 9.4 semantics | Dedicated tests verify `STRONG`, `MODERATE`, `WEAK`, `NEGATIVE`, `INCONCLUSIVE` cases | YES |
| Regression safety | Existing validated pipeline still passes | Full suite passes: 395 tests | YES |

---

## Deviations from Plan

### DEV-008: Report Generator operationalizes evidence criteria from available SR-7 metrics

- **What changed:** Instead of waiting for every Section 9.4 criterion to be directly measured by the pipeline, SR-8 computes verdicts from the evidence currently available in `ExperimentReport` and records per-criterion flags plus explanatory notes.
- **Why:** TASK-10 requires verdict artifacts now, while some diagnostic criteria (for example sample efficiency and counterfactual sensitivity) are only planned for later phases.
- **Impact:** Verdicts are auditable today and compatible with later refinement. Recorded in `EXPERIMENT_CATALOG.md` Part 5 and ADR-017.

---

## Completion Summary

TASK-10 completed the full reporting layer for the experiment pipeline. `src/reporting.py` now turns `ExperimentReport` objects into the full SR-8 artifact tree, including config snapshots, per-task metrics and errors, markdown summaries, plots, and solvability verdicts with explicit evidence flags. V-8 passes with 9 new tests, the full suite is green at 395 tests, and the project is now ready for TASK-11 smoke-test execution on a fully built end-to-end pipeline.
