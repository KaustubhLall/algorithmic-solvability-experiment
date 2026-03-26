# TASK-10: Report Generator

> **Status:** COMPLETE
> **Builds:** SR-8
> **Validation:** V-8 (6 tests passing)
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-8), Part 3 (V-8), Part 4 (TASK-10)
> **Date Started:** 2026-03-25
> **Date Completed:** 2026-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `generate_report_artifacts()` experiment writer | `src/reporting.py` | DONE |
| `config.json` emission with spec + execution metadata | `src/reporting.py` | DONE |
| `summary.md` experiment overview | `src/reporting.py` | DONE |
| Per-task `metrics.json` | `src/reporting.py` | DONE |
| Per-task `errors.json` | `src/reporting.py` | DONE |
| Per-task `confusion.png` for classification tasks | `src/reporting.py` | DONE |
| Per-task `extrap_curve.png` | `src/reporting.py` | DONE |
| `comparison.md` cross-task table | `src/reporting.py` | DONE |
| `solvability_verdicts.json` with Section 9.4 logic | `src/reporting.py` | DONE |
| V-8 validation suite | `tests/test_reporting.py` | DONE (6 tests) |

---

## Implementation Notes

### Artifact layout

- `generate_report_artifacts(report, output_root, registry)` writes the full SR-8 tree under `results/{experiment_id}/` (or an alternate root for tests).
- The writer clears any existing experiment directory before writing fresh artifacts. This avoids stale files from a previous run surviving into the new report tree.
- `config.json` stores the serialized experiment spec plus execution metadata (`seeds_used`, runtime, generation timestamp) so the report can be understood without loading Python objects.

### Per-task outputs

- `metrics.json` stores both the flat aggregated/single-run lists and a nested model -> split view. This keeps the file easy for scripts to scan while also making per-model drill-down convenient.
- `errors.json` aggregates error taxonomies per model and split, records per-seed counts, and computes mean error-rate shares for quick diagnostics.
- `confusion.png` is generated only for classification tasks, using the highest-accuracy available classification run (prefer IID when tied). Sequence tasks omit the confusion artifact because no confusion matrix exists for them.
- `extrap_curve.png` is generated for every task and plots aggregated accuracy by split for each model, even when only a single split is available.

### Solvability verdict policy

- The report generator implements a criteria-based policy aligned to `EXPERIMENT_DESIGN.md` Section 9.4.
- Required criteria (1-5) are checked from aggregated experiment metrics. Optional criteria (6-9) are only marked when the relevant evidence exists in the run.
- The weighted solvability score from Section 11.5 is computed over the evidence channels actually present in the report and normalized by the available weight. This prevents missing experimental axes from automatically depressing the score.
- Final labels are chosen per task from the best model-level assessment. A model can only earn `STRONG` if the required criteria hold and at least two optional criteria are positively observed.

### Edge cases handled

- Tasks with only IID evidence can still produce a `WEAK` verdict rather than being forced into `INCONCLUSIVE`.
- Tasks with no positive evidence and uniformly poor results are labeled `NEGATIVE`.
- Tasks with mixed or incomplete evidence fall back to `INCONCLUSIVE`.
- Empty or partial per-task result sets still write valid JSON/Markdown files and a placeholder extrapolation plot.

---

## Acceptance Criteria Results

V-8 validation: **6 tests, all passing** (`tests/test_reporting.py`, 1.19s)

Additional regression validation: **full test suite passes** (`.venv\Scripts\python.exe -m pytest -q`, `392 passed, 4 warnings`, 2.68s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| File structure | Expected SR-8 directory tree is created | `config.json`, `summary.md`, `comparison.md`, `solvability_verdicts.json`, and per-task artifacts are written | YES |
| JSON validity | All JSON files parse cleanly | `config.json`, `metrics.json`, `errors.json`, and `solvability_verdicts.json` parse in V-8 tests | YES |
| Markdown validity | `summary.md` renders with consistent content | V-8 checks summary/comparison markdown contains expected experiment and task rows | YES |
| Metric consistency | Markdown metrics match JSON metrics | V-8 compares summary values against `metrics.json` | YES |
| Solvability verdict logic | Labels match Section 9.4 policy | V-8 covers `WEAK`, `MODERATE`, `STRONG`, and `NEGATIVE` verdict cases | YES |

---

## Deviations from Plan

None. TASK-10 was completed without changing the SR-8 deliverable list or acceptance criteria.

---

## Completion Summary

TASK-10 delivered the full SR-8 reporting layer in `src/reporting.py` (729 lines) together with a focused V-8 suite in `tests/test_reporting.py` (6 tests). The report generator now turns any `ExperimentReport` from SR-7 into a durable artifact tree with JSON summaries, Markdown summaries/comparisons, per-task plots, and structured solvability verdicts that encode both the Section 9.4 label and the supporting evidence. The main design challenge was turning the design document's qualitative solvability criteria into deterministic report logic without over-claiming from incomplete experiments; the chosen solution keeps the labels criteria-driven and treats the weighted score as supporting evidence rather than a label shortcut. TASK-11 can now run smoke experiments and persist their artifacts directly through SR-8 without additional reporting glue.
