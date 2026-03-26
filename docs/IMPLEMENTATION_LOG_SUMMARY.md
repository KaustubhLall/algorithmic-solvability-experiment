# IMPLEMENTATION LOG SUMMARY

> **PURPOSE:** A running, human-readable summary of all implementation work completed so far.
> This is the "what got built and what was learned" companion to `PROJECT_STATUS.md`.
> Detailed per-task logs live in `docs/implementation_log/TASK-XX_<name>.md`.
> Link each completed task from the table below to its detailed log.
>
> **Last Updated:** 2025-03-25 (TASK-08 complete, FOUNDATION + DATA PIPELINE milestones done, V-6 passing)
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
| TASK-01 | Input Schema (SR-2) | 2025-03-25 | [log](implementation_log/TASK-01_input_schema.md) | All 4 schema classes + 2 enums in `src/schemas.py`. 52 V-2 tests pass. |
| TASK-02 | Classification Rule DSL (SR-9) | 2025-03-25 | [log](implementation_log/TASK-02_classification_dsl.md) | Predicates, combinators, classifiers, aggregators, rule sampler in `src/dsl/classification_dsl.py`. 55 V-9 tests pass. |
| TASK-03 | Sequence DSL (SR-10) | 2025-03-25 | [log](implementation_log/TASK-03_sequence_dsl.md) | 18+ primitives, composition, program sampler, equivalence checking in `src/dsl/sequence_dsl.py`. 56 V-10 tests pass. |
| TASK-04 | Task Registry (SR-1) | 2025-03-25 | [log](implementation_log/TASK-04_task_registry.md) | **FOUNDATION MILESTONE.** 28 tasks across S0–S3, C0–C3 in `src/registry.py`. 34 V-1 tests pass. |
| TASK-05 | Data Generator (SR-3) | 2025-03-25 | [log](implementation_log/TASK-05_data_generator.md) | Label re-verification, noise injection, multi-task generation in `src/data_generator.py`. 23 V-3 tests pass. |
| TASK-06 | Split Generator (SR-4) | 2025-03-25 | [log](implementation_log/TASK-06_split_generator.md) | **DATA PIPELINE MILESTONE.** 4 split strategies (IID, length, value, noise) in `src/splits.py`. 29 V-4 tests pass. |
| TASK-07 | Model Harness (SR-5) | 2025-03-25 | [log](implementation_log/TASK-07_model_harness.md) | 8 model families, unified encode→train→predict→decode pipeline in `src/models/harness.py`. 33 V-5 tests pass. |
| TASK-08 | Evaluation Engine (SR-6) | 2025-03-25 | [log](implementation_log/TASK-08_evaluation_engine.md) | Classification + sequence metric dispatch, confusion matrix, per-class P/R/F1, error taxonomy, metadata-conditioned breakdowns in `src/evaluation.py`. 52 V-6 tests pass. |

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
- **Sequence token accuracy requires parsing stringified lists.** Sequence outputs are stringified by the harness, so the evaluation engine must parse them back (e.g., `"[1, 2, 3]"` → `["1", "2", "3"]`) for token-level comparison. Non-parseable outputs are gracefully skipped.
- **Error taxonomy is track-specific.** Classification errors: correct/wrong_class/unknown_class. Sequence errors: correct/length_mismatch/content_error/off_by_one. Separate taxonomies give more actionable diagnostics.

---

## Current Blockers

Issues actively blocking progress and needing resolution before the next task can start.

_None. TASK-01 through TASK-08 complete. TASK-09 (Experiment Runner) ready to start._

---

## Test / Validation Summary

Quick record of which validation procedures (V-1 through V-10 + V-Global) are passing.

| Validation | Component | Status | Notes |
|---|---|---|---|
| V-1 | Task Registry | **PASS** ✓ | 34 tests, all passing (0.22s) |
| V-2 | Input Schema | **PASS** ✓ | 52 tests, all passing (4.13s) |
| V-3 | Data Generator | **PASS** ✓ | 23 tests, all passing (0.23s) |
| V-4 | Split Generator | **PASS** ✓ | 29 tests, all passing (0.12s) |
| V-5 | Model Harness | **PASS** ✓ | 33 tests, all passing (1.00s) |
| V-6 | Evaluation Engine | **PASS** ✓ | 52 tests, all passing (0.84s) |
| V-7 | Experiment Runner | NOT RUN | |
| V-8 | Report Generator | NOT RUN | |
| V-9 | Classification Rule DSL | **PASS** ✓ | 55 tests, all passing (0.28s) |
| V-10 | Sequence DSL | **PASS** ✓ | 56 tests, all passing (0.16s) |
| V-G1 | Round-trip check | NOT RUN | |
| V-G2 | Control task calibration | NOT RUN | |
| V-G3 | Trivial task ceiling | NOT RUN | |
| V-G4 | Data-model isolation | NOT RUN | |
