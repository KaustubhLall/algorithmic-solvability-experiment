# IMPLEMENTATION LOG SUMMARY

> **PURPOSE:** A running, human-readable summary of all implementation work completed so far.
> This is the "what got built and what was learned" companion to `PROJECT_STATUS.md`.
> Detailed per-task logs live in `docs/implementation_log/TASK-XX_<name>.md`.
> Link each completed task from the table below to its detailed log.
>
> **Last Updated:** 2026-03-25 (TASK-10 complete, V-8 passing, full suite green)
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
| TASK-02 | Classification Rule DSL (SR-9) | 2025-03-25 | [log](implementation_log/TASK-02_classification_dsl.md) | Predicates, combinators, classifiers, aggregators, rule sampler in `src/dsl/classification_dsl.py`. 58 V-9 tests pass. |
| TASK-03 | Sequence DSL (SR-10) | 2025-03-25 | [log](implementation_log/TASK-03_sequence_dsl.md) | 18+ primitives, composition, program sampler, equivalence checking in `src/dsl/sequence_dsl.py`. 57 V-10 tests pass. |
| TASK-04 | Task Registry (SR-1) | 2025-03-25 | [log](implementation_log/TASK-04_task_registry.md) | **FOUNDATION MILESTONE.** 28 tasks across S0-S3, C0-C3 in `src/registry.py`. 35 V-1 tests pass. |
| TASK-05 | Data Generator (SR-3) | 2025-03-25 | [log](implementation_log/TASK-05_data_generator.md) | Label re-verification, noise injection, multi-task generation in `src/data_generator.py`. 23 V-3 tests pass. |
| TASK-06 | Split Generator (SR-4) | 2025-03-25 | [log](implementation_log/TASK-06_split_generator.md) | **DATA PIPELINE MILESTONE.** 4 split strategies (IID, length, value, noise) in `src/splits.py`. 33 V-4 tests pass. |
| TASK-07 | Model Harness (SR-5) | 2025-03-25 | [log](implementation_log/TASK-07_model_harness.md) | 8 model families, unified encode->train->predict->decode pipeline in `src/models/harness.py`. 39 V-5 tests pass. |
| TASK-08 | Evaluation Engine (SR-6) | 2025-03-25 | [log](implementation_log/TASK-08_evaluation_engine.md) | Classification + sequence metric dispatch, confusion matrix, per-class P/R/F1, error taxonomy, metadata-conditioned breakdowns in `src/evaluation.py`. 53 V-6 tests pass. |
| TASK-09 | Experiment Runner (SR-7) | 2026-03-25 | [log](implementation_log/TASK-09_experiment_runner.md) | Multi-seed experiment orchestration, aggregation, progress logging, and JSON-ready serialization in `src/runner.py`. 36 V-7 tests pass. |
| TASK-10 | Report Generator (SR-8) | 2026-03-25 | [log](implementation_log/TASK-10_report_generator.md) | **FULL PIPELINE MILESTONE.** Structured experiment artifacts, per-task plots, markdown summaries, and solvability verdict logic in `src/reporting.py`. 7 V-8 tests pass. |

---

## Running Lessons Learned

Surprising findings, non-obvious edge cases, or things that would save time in future chats.

- **Use tuples, not lists, in frozen dataclasses.** Lists are not hashable, so frozen `@dataclass` fields must use tuples. This affects all callers constructing specs.
- **`np.random.default_rng(seed)` is the right API.** Modern NumPy generator API gives full reproducibility. Each `sample()` call creates its own generator from the seed.
- **Batch sampling uses sequential draws from one RNG.** `sample_batch(seed, n)` is NOT the same as `[sample(seed+i) for i in range(n)]`. The batch method is more efficient and produces better-distributed samples.
- **Reducers should return `list[int]` not `int`.** Wrapping reducer output in a single-element list enables uniform composition without type mismatch. Minor downstream cost: unwrap `[result]` when scalar is needed.
- **Callable-based interfaces beat class hierarchies for TaskSpec.** `reference_algorithm`, `input_sampler`, and `verifier` as plain callables keeps the registry simple and avoids abstraction overhead.
- **Label re-verification at generation time catches bugs early.** DataGenerator's `verify_labels=True` default has zero runtime cost on correct code and would immediately catch reference algorithm bugs.
- **Noise seed must differ from data seed.** Noise uses `seed + 2**31` to avoid correlation with input sampling. Same pattern used in SplitGenerator's noise split (`seed + 2**30`).
- **Single-file modules are fine until ~500 lines.** `harness.py` (459 lines) works well as a single file. The planned 3-file split (harness/configs/encoders) was unnecessary at this scale.
- **All 8 sklearn model families work out-of-the-box.** No special handling needed beyond the `SklearnModelWrapper`. `random_state=42` everywhere ensures reproducibility.
- **String-based evaluation simplifies metric computation.** By having `ModelHarness.run()` return string predictions and string true labels, the evaluation engine can use simple string equality for accuracy. No type coercion needed.
- **Confusion matrix rows = true class, columns = predicted class.** Standard convention. Per-class precision/recall derived directly from the matrix without re-scanning predictions.
- **Sequence token accuracy requires parsing stringified lists.** Sequence outputs are stringified by the harness, so the evaluation engine must parse them back (e.g., `"[1, 2, 3]"` -> `["1", "2", "3"]`) for token-level comparison. Non-parseable outputs are gracefully skipped.
- **Error taxonomy is track-specific.** Classification errors: correct/wrong_class/unknown_class. Sequence errors: correct/length_mismatch/content_error/off_by_one. Separate taxonomies give more actionable diagnostics.
- **Reuse one dataset per (task, seed) across split strategies.** The runner generates data once per task/seed, then derives all requested splits from that shared dataset. This keeps split comparisons paired and reduces variance from resampling.
- **Carry execution metadata forward early.** Recording `seeds_used`, per-run `split_metadata`, and JSON-ready report serialization in SR-7 makes TASK-10 simpler because reporting can focus on file layout rather than reconstructing pipeline state.
- **Normalize solvability scores over available evidence only.** TASK-10 computes the weighted solvability score using the Section 11.5 weights, but normalizes by the weights for components actually observed in the experiment. This avoids unfairly penalizing experiments that intentionally omit distractor or sample-efficiency evidence.
- **Separate verdict labels from score magnitude.** The report generator uses explicit Section 9.4 criteria for `STRONG` / `MODERATE` / `WEAK` / `NEGATIVE` / `INCONCLUSIVE`, then reports the weighted score as supporting evidence. A high score without the required criteria should not be promoted automatically.
- **Rewrite result directories per experiment run.** Clearing `results/{experiment_id}` before writing fresh SR-8 artifacts prevents stale files from previous runs from surviving into the current report tree.

---

## Current Blockers

Issues actively blocking progress and needing resolution before the next task can start.

_None. TASK-11 (Pipeline Smoke Tests) is ready to start._

---

## Test / Validation Summary

Quick record of which validation procedures (V-1 through V-10 + V-Global) are passing.

| Validation | Component | Status | Notes |
|---|---|---|---|
| V-1 | Task Registry | **PASS** | 35 tests, all passing |
| V-2 | Input Schema | **PASS** | 54 tests, all passing |
| V-3 | Data Generator | **PASS** | 23 tests, all passing (0.23s) |
| V-4 | Split Generator | **PASS** | 33 tests, all passing |
| V-5 | Model Harness | **PASS** | 39 tests, all passing |
| V-6 | Evaluation Engine | **PASS** | 53 tests, all passing |
| V-7 | Experiment Runner | **PASS** | 36 tests, all passing |
| V-8 | Report Generator | **PASS** | 7 tests, all passing |
| V-9 | Classification Rule DSL | **PASS** | 58 tests, all passing |
| V-10 | Sequence DSL | **PASS** | 57 tests, all passing |
| V-G1 | Round-trip check | NOT RUN | |
| V-G2 | Control task calibration | NOT RUN | |
| V-G3 | Trivial task ceiling | NOT RUN | |
| V-G4 | Data-model isolation | NOT RUN | |

Full-suite status: `395 passed, 4 warnings` (`.venv\Scripts\python.exe -m pytest -q`, 2.67s)
