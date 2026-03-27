# IMPLEMENTATION LOG SUMMARY

> **PURPOSE:** A running, human-readable summary of all implementation work completed so far.
> This is the "what got built and what was learned" companion to `PROJECT_STATUS.md`.
> Detailed per-task logs live in `docs/implementation_log/TASK-XX_<name>.md`.
> Link each completed task from the table below to its detailed log.
>
> **Last Updated:** 2026-03-26 (TASK-16 complete; methodology feedback plan landed, docs reconciled to the rerun/preprint, full suite previously rerun at 460 passed / 17 warnings)
> **Format:** Append entries as tasks complete. Never delete past entries.

---

## How to Use This Document

- **At the start of a chat:** read this file to quickly understand what has been built and any surprises encountered.
- **At the end of a chat:** add a new entry to the Completed Work table and update the Lessons Learned section if anything non-obvious was discovered.
- **Full detail:** click the log link in each row for the per-task breakdown (code decisions, file paths, test results, edge cases).

---

## Completed Work

| Task | Scope | Completed | Log | Key outcome |
|---|---|---|---|---|
| TASK-01 | Input Schema (SR-2) | 2025-03-25 | [log](implementation_log/TASK-01_input_schema.md) | All 4 schema classes + 2 enums in `src/schemas.py`. 54 V-2 tests pass. |
| TASK-02 | Classification Rule DSL (SR-9) | 2025-03-25 | [log](implementation_log/TASK-02_classification_dsl.md) | Predicates, combinators, classifiers, aggregators, and rule sampling in `src/dsl/classification_dsl.py`. 58 V-9 tests pass. |
| TASK-03 | Sequence DSL (SR-10) | 2025-03-25 | [log](implementation_log/TASK-03_sequence_dsl.md) | 18+ primitives, composition, program sampling, and equivalence checking in `src/dsl/sequence_dsl.py`. 57 V-10 tests pass. |
| TASK-04 | Task Registry (SR-1) | 2025-03-25 | [log](implementation_log/TASK-04_task_registry.md) | **FOUNDATION MILESTONE.** 32 tasks across S0-S3 and C0-C3 in `src/registry.py`. 35 V-1 tests pass. |
| TASK-05 | Data Generator (SR-3) | 2025-03-25 | [log](implementation_log/TASK-05_data_generator.md) | Label re-verification, noise injection, and multi-task generation in `src/data_generator.py`. 23 V-3 tests pass. |
| TASK-06 | Split Generator (SR-4) | 2025-03-25 | [log](implementation_log/TASK-06_split_generator.md) | **DATA PIPELINE MILESTONE.** Four split strategies (IID, length, value, noise) in `src/splits.py`. 33 V-4 tests pass. |
| TASK-07 | Model Harness (SR-5) | 2025-03-25 | [log](implementation_log/TASK-07_model_harness.md) | 9 model families, unified encode -> train -> predict -> decode pipeline in `src/models/harness.py`, including a raw-sequence LSTM path. 38 V-5 tests pass. |
| TASK-08 | Evaluation Engine (SR-6) | 2025-03-25 | [log](implementation_log/TASK-08_evaluation_engine.md) | Classification + sequence metric dispatch, confusion matrix, per-class P/R/F1, error taxonomy, and metadata-conditioned breakdowns in `src/evaluation.py`. 53 V-6 tests pass. |
| TASK-09 | Experiment Runner (SR-7) | 2026-03-25 | [log](implementation_log/TASK-09_experiment_runner.md) | Multi-seed experiment orchestration, aggregation, progress logging, and JSON-ready serialization in `src/runner.py`. 36 V-7 tests pass. |
| TASK-10 | Report Generator (SR-8) | 2026-03-25 | [log](implementation_log/TASK-10_report_generator.md) | Structured experiment artifact writing, per-task metrics/errors JSON, markdown summaries, plots, and solvability verdict computation in `src/reporting.py`. 9 V-8 tests pass. |
| TASK-11 | Smoke Tests (EXP-0.x) | 2026-03-25 | [log](implementation_log/TASK-11_smoke_tests.md) | Added `src/smoke_tests.py`, a CLI entrypoint in `main.py`, a raw-sequence LSTM path in `src/models/harness.py`, 7 V-Global smoke tests, and `results/EXP-0.1` through `results/EXP-0.3` artifacts. |
| TASK-12 | Sequence Experiments | 2026-03-26 | [log](implementation_log/TASK-12_sequence_experiments.md) | Added `src/sequence_experiments.py`, `main.py sequence`, `tests/test_sequence_experiments.py`, unseen-token-safe LSTM inference, and generated `results/EXP-S1` through `results/EXP-S3` for the implemented S1-S3 sequence tiers. |
| TASK-13 | Classification Experiments | 2026-03-26 | [log](implementation_log/TASK-13_classification_experiments.md) | Added `src/classification_experiments.py`, `main.py classification`, `tests/test_classification_experiments.py`, schema-guided categorical noise for tabular robustness splits, and generated `results/EXP-C1` through `results/EXP-C3` for the implemented C1-C3 classification tiers. |
| TASK-14 | Diagnostic Experiments | 2026-03-26 | [log](implementation_log/TASK-14_diagnostic_experiments.md) | Added `src/diagnostic_experiments.py`, `main.py diagnostic`, `tests/test_diagnostic_experiments.py` (21 tests). Implemented EXP-D1 through EXP-D5 and exposed `InputEncoder.feature_names` plus `SklearnModelWrapper.estimator` for diagnostics. |
| TASK-15 | Bonus: Algorithm Discovery | 2026-03-26 | [log](implementation_log/TASK-15_bonus_algorithm_discovery.md) | Added `src/bonus_experiments.py`, `main.py bonus`, `tests/test_bonus_experiments.py` (20 tests). EXP-B1 extracts decision-tree rules; EXP-B2 searches SR-10 DSL programs using the reference algorithm as oracle. |
| TASK-16 | Methodology Feedback Planning | 2026-03-26 | [log](implementation_log/TASK-16_methodology_feedback_planning.md) | Re-read the onboarding docs, reconciled the methodology review against `PREPUBLICATION_ANALYSIS_2026-03-26.md` and the manuscript source, corrected stale empirical claims, extracted review themes from merged PRs `#19` and `#20`, and updated the status/catalog/ADR docs so the next task starts from the correct rerun-backed baseline. |

---

## Running Lessons Learned

Surprising findings, non-obvious edge cases, or things that would save time in future chats.

- **Use tuples, not lists, in frozen dataclasses.** Lists are not hashable, so frozen `@dataclass` fields must use tuples.
- **`np.random.default_rng(seed)` is the right API.** Modern NumPy generator API gives full reproducibility.
- **Batch sampling uses sequential draws from one RNG.** `sample_batch(seed, n)` is not the same as `[sample(seed+i) for i in range(n)]`.
- **Reducers should return `list[int]` not `int`.** Wrapping reducer output in a single-element list enables uniform composition.
- **Callable-based interfaces beat class hierarchies for TaskSpec.** `reference_algorithm`, `input_sampler`, and `verifier` as plain callables keep the registry simple.
- **Label re-verification at generation time catches bugs early.** Small runtime cost, large correctness benefit.
- **Noise seeds should differ from data seeds.** This avoids correlation between input sampling and perturbation.
- **Single-file modules can stay practical longer than expected.** `harness.py` has remained workable despite the original plan to split it.
- **String-based evaluation simplifies metric computation.** Returning strings from the harness keeps the evaluator uniform across tracks.
- **Track-specific error taxonomies are worth the extra branching.** Sequence and classification failures are different enough that separate categories pay for themselves.
- **Reuse one dataset per `(task, seed)` across split strategies.** This keeps split comparisons paired and reduces variance.
- **Section 9.4 needs an explicit operationalization layer.** The reporting layer should expose which criteria are actually measured rather than guessing.
- **Smoke-task data shaping is best handled outside the global registry.** Local task cloning kept smoke behavior faithful without narrowing the benchmark.
- **Sequence experiments need experiment-specific split menus.** Not every split makes sense for every task family.
- **Value extrapolation needs explicit unseen-token handling for neural sequence models.** Otherwise the LSTM crashes at inference on fresh symbols.
- **The current implemented sequence benchmark remains mostly negative.** In the rerun-backed prepublication summary, 12 of 16 sequence/control tasks are `NEGATIVE`, 2 are `WEAK` (`S1.4_count_symbol`, `S2.2_balanced_parens`), and 2 are `INCONCLUSIVE` (`S1.5_parity`, `S1.8_extrema`).
- **Classification noise needs schema-aware categorical perturbations.** Otherwise categorical-only tasks get an IID duplicate instead of a real robustness split.
- **Mid-range value windows preserve label support for threshold-like tasks.** A training band such as `[20, 80]` keeps both sides of the decision boundary represented.
- **The implemented classification benchmark is much stronger than the sequence suite at current scale.** In the rerun-backed prepublication summary, 11 of 14 classification/control tasks are `MODERATE` at baseline; `C2.1_and_rule` is `WEAK`, `C1.6_modular_class` is `INCONCLUSIVE`, the control stays `NEGATIVE`, and diagnostic calibration promotes `C1.1_numeric_threshold` to `STRONG`.
- **Diagnostic experiments can reuse baseline artifacts without re-running the full pipeline.** This keeps D1-D5 tied to the actual baseline evidence and saves a lot of compute.
- **Task-level distractor injection via `_clone_task_with_distractors` is cleaner than split-level injection.** It preserves end-to-end correctness while keeping SR-4 stable.
- **`sklearn.inspection.permutation_importance` works directly on `SklearnModelWrapper.estimator`.** Exposing the estimator was enough to unlock D4.
- **NumPy 2.0+ removed `np.trapz`; use `np.trapezoid` with a fallback.**
- **Solvability calibration benefits from a structured helper function.** Extracting `_calibrated_label()` made the verdict rules testable in isolation.
- **Decision-tree extraction works best on simple threshold rules.** Tasks like `C1.1_numeric_threshold` are near-perfectly recoverable by shallow trees.
- **Random DSL program search is surprisingly effective for primitives.** `Reverse()` and `Sort()` are easy to recover; deeper compositions are not.
- **`generate_dataset()` returns a `Dataset` object, not a list.** Use `.samples` when you need the list of examples.
- **Publication-facing claims must be tied to the latest rerun artifacts.** TASK-16 fixed a drift where the methodology review no longer matched the rerun-backed analysis and manuscript.
- **Keep benchmark claims scoped to the implemented tiers.** The defensible wording is implemented `S0-S3` and `C0-C3`, not the full aspirational S4/S5/C4/C5 roadmap.
- **Keep PRs free of workspace noise.** Review of merged PRs `#19` and `#20` reinforced excluding IDE and generated artifacts unless they are actual deliverables.

---

## Current Blockers

Issues actively blocking progress and needing resolution before the next task can start.

_None. TASK-16 is complete; the next queued work is TASK-17 methodology-feedback execution._

---

## Test / Validation Summary

Quick record of which validation procedures (V-1 through V-10 + V-Global) are passing.

| Validation | Component | Status | Notes |
|---|---|---|---|
| V-1 | Task Registry | **PASS** | 35 tests, all passing |
| V-2 | Input Schema | **PASS** | 54 tests, all passing |
| V-3 | Data Generator | **PASS** | 23 tests, all passing |
| V-4 | Split Generator | **PASS** | 33 tests, all passing |
| V-5 | Model Harness | **PASS** | 38 tests, all passing |
| V-6 | Evaluation Engine | **PASS** | 53 tests, all passing |
| V-7 | Experiment Runner | **PASS** | 36 tests, all passing |
| V-8 | Report Generator | **PASS** | 9 tests, all passing |
| V-9 | Classification Rule DSL | **PASS** | 58 tests, all passing |
| V-10 | Sequence DSL | **PASS** | 57 tests, all passing |
| V-G1 | Round-trip check | **PASS** | Covered in `tests/test_smoke_tests.py`; manual accuracy matches evaluation/report accuracy |
| V-G2 | Control task calibration | **PASS** | `EXP-0.3` verdicts are NEGATIVE for both control tasks |
| V-G3 | Trivial task ceiling | **PASS** | `EXP-0.2` decision tree/logistic regression reach 100% accuracy; verdict is MODERATE under the current operationalization |
| V-G4 | Data-model isolation | **PASS** | Runner smoke validation confirms fresh per-task dataset generation |
| V-14 | Diagnostic Experiments | **PASS** | 21 tests, all passing |
| V-15 | Bonus Experiments | **PASS** | 20 tests, all passing |

Full suite status from the fresh rerun used for the prepublication package: **460 passed, 17 warnings**. The warnings were training-stack/runtime warnings, not correctness failures.
