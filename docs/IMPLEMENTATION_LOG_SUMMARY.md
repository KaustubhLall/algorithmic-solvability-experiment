# IMPLEMENTATION LOG SUMMARY

> **PURPOSE:** A running, human-readable summary of all implementation work completed so far.
> This is the "what got built and what was learned" companion to `PROJECT_STATUS.md`.
> Detailed per-task logs live in `docs/implementation_log/TASK-XX_<name>.md`.
> Link each completed task from the table below to its detailed log.
>
> **Last Updated:** 2026-03-26 (TASK-12 complete, sequence suite artifacts generated, full suite green)
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
| TASK-07 | Model Harness (SR-5) | 2025-03-25 | [log](implementation_log/TASK-07_model_harness.md) | 9 model families, unified encode->train->predict->decode pipeline in `src/models/harness.py`, now including a raw-sequence LSTM path. 38 V-5 tests pass. |
| TASK-08 | Evaluation Engine (SR-6) | 2025-03-25 | [log](implementation_log/TASK-08_evaluation_engine.md) | Classification + sequence metric dispatch, confusion matrix, per-class P/R/F1, error taxonomy, metadata-conditioned breakdowns in `src/evaluation.py`. 53 V-6 tests pass. |
| TASK-09 | Experiment Runner (SR-7) | 2026-03-25 | [log](implementation_log/TASK-09_experiment_runner.md) | Multi-seed experiment orchestration, aggregation, progress logging, and JSON-ready serialization in `src/runner.py`. 36 V-7 tests pass. |
| TASK-10 | Report Generator (SR-8) | 2026-03-25 | [log](implementation_log/TASK-10_report_generator.md) | Structured experiment artifact writing, per-task metrics/errors JSON, markdown summaries, plots, and solvability verdict computation in `src/reporting.py`. 9 V-8 tests pass. |
| TASK-11 | Smoke Tests (EXP-0.x) | 2026-03-25 | [log](implementation_log/TASK-11_smoke_tests.md) | Added `src/smoke_tests.py`, a CLI entrypoint in `main.py`, a raw-sequence LSTM path in `src/models/harness.py`, 7 V-Global smoke tests, and `results/EXP-0.1` through `results/EXP-0.3` artifacts. |
| TASK-12 | Sequence Experiments | 2026-03-26 | [log](implementation_log/TASK-12_sequence_experiments.md) | Added `src/sequence_experiments.py`, `main.py sequence`, `tests/test_sequence_experiments.py`, unseen-token-safe LSTM inference, and generated `results/EXP-S1` through `results/EXP-S3` for the implemented S1-S3 sequence tiers. |

---

## Running Lessons Learned

Surprising findings, non-obvious edge cases, or things that would save time in future chats.

- **Use tuples, not lists, in frozen dataclasses.** Lists are not hashable, so frozen `@dataclass` fields must use tuples. This affects all callers constructing specs.
- **`np.random.default_rng(seed)` is the right API.** Modern NumPy generator API gives full reproducibility. Each `sample()` call creates its own generator from the seed.
- **Batch sampling uses sequential draws from one RNG.** `sample_batch(seed, n)` is NOT the same as `[sample(seed+i) for i in range(n)]`. The batch method is more efficient and produces better-distributed samples.
- **Reducers should return `list[int]` not `int`.** Wrapping reducer output in a single-element list enables uniform composition without type mismatch. Minor downstream cost: unwrap `[result]` when scalar is needed.
- **Callable-based interfaces beat class hierarchies for TaskSpec.** `reference_algorithm`, `input_sampler`, and `verifier` as plain callables keeps the registry simple and avoids abstraction overhead.
- **Label re-verification at generation time catches bugs early.** DataGenerator's `verify_labels=True` default re-runs the reference algorithm once per sample, adding a small runtime cost in exchange for stronger correctness guarantees.
- **Noise seed must differ from data seed.** Noise uses `seed + 2**31` to avoid correlation with input sampling. Same pattern used in SplitGenerator's noise split (`seed + 2**30`).
- **Single-file modules can stay practical well past the initial plan.** `harness.py` has remained manageable as one module, so the planned split into `harness.py`/`configs.py`/`encoders.py` was unnecessary overhead at this stage.
- **Most sklearn model families work out-of-the-box.** No special handling was needed beyond `SklearnModelWrapper`; where a model exposes `random_state`, we set it for reproducibility.
- **String-based evaluation simplifies metric computation.** By having `ModelHarness.run()` return string predictions and string true labels, the evaluation engine can use simple string equality for accuracy. No type coercion needed.
- **Confusion matrix rows = true class, columns = predicted class.** Standard convention. Per-class precision/recall derived directly from the matrix without re-scanning predictions.
- **Sequence token accuracy requires parsing stringified lists.** Sequence outputs are stringified by the harness, so the evaluation engine must parse them back (e.g., `"[1, 2, 3]"` -> `["1", "2", "3"]`) for token-level comparison. Non-parseable outputs are gracefully skipped.
- **Error taxonomy is track-specific.** Classification errors: correct/wrong_class/unknown_class. Sequence errors: correct/length_mismatch/content_error/off_by_one. Separate taxonomies give more actionable diagnostics.
- **Reuse one dataset per (task, seed) across split strategies.** The runner generates data once per task/seed, then derives all requested splits from that shared dataset. This keeps split comparisons paired and reduces variance from resampling.
- **Carry execution metadata forward early.** Recording `seeds_used`, per-run `split_metadata`, and JSON-ready report serialization in SR-7 makes TASK-10 simpler because reporting can focus on file layout rather than reconstructing pipeline state.
- **Use runner/evaluation serializers as the reporting source of truth.** `experiment_report_to_dict()`, `aggregated_result_to_dict()`, and `single_result_to_dict()` keep SR-8 from drifting away from SR-7/6 field names.
- **Section 9.4 needs an explicit operationalization layer.** The design labels are qualitative; SR-8 now records per-criterion evidence flags and notes, using currently available SR-7 signals while leaving unavailable criteria clearly unmet instead of guessing.
- **Per-task plots can be generated from reports alone.** Averaging stored confusion matrices and plotting aggregated accuracy by split was enough for V-8; no raw dataset replay was necessary.
- **Smoke-task data shaping is best handled outside the global registry.** EXP-0.1 needs sort sequences bounded to length 4-8, but the benchmark task definition stays broader. Cloning that task into a smoke-local registry preserved the benchmark while matching the cataloged smoke regime.
- **Current solvability calibration needs richer smoke evidence for classification.** Under ADR-017, IID-only single-seed trivial tasks can at best look WEAK. TASK-11 therefore expanded EXP-0.2 with a baseline, a noise split, and 5 seeds so V-G3 can validate the currently measurable minimum evidence and land at MODERATE.
- **Sequence experiments need experiment-specific split menus.** Binary/stateful tasks such as `S2.1_cumulative_xor` and `S2.2_balanced_parens` do not have a meaningful value-range extrapolation regime, so TASK-12 keeps EXP-S2 on IID + length extrapolation only.
- **Value extrapolation needs explicit unseen-token handling for neural sequence models.** The raw-sequence LSTM now maps unseen test-time tokens into an input-only unknown bucket so EXP-S1/EXP-S3 value-range runs complete instead of crashing.
- **The current sequence baseline suite is mostly negative at today's supported scale.** Across `results/EXP-S1` through `results/EXP-S3`, only `S1.4_count_symbol` and `S2.2_balanced_parens` reached WEAK evidence; most other sequence tasks remained NEGATIVE under the present models/sample budgets.

---

## Current Blockers

Issues actively blocking progress and needing resolution before the next task can start.

_None. TASK-13 (Classification Experiments) is ready to start._

---

## Test / Validation Summary

Quick record of which validation procedures (V-1 through V-10 + V-Global) are passing.

| Validation | Component | Status | Notes |
|---|---|---|---|
| V-1 | Task Registry | **PASS** | 35 tests, all passing |
| V-2 | Input Schema | **PASS** | 54 tests, all passing |
| V-3 | Data Generator | **PASS** | 23 tests, all passing (0.23s) |
| V-4 | Split Generator | **PASS** | 33 tests, all passing |
| V-5 | Model Harness | **PASS** | 38 tests, all passing |
| V-6 | Evaluation Engine | **PASS** | 53 tests, all passing |
| V-7 | Experiment Runner | **PASS** | 36 tests, all passing (0.95s) |
| V-8 | Report Generator | **PASS** | 9 tests, all passing |
| V-9 | Classification Rule DSL | **PASS** | 58 tests, all passing |
| V-10 | Sequence DSL | **PASS** | 57 tests, all passing |
| V-G1 | Round-trip check | **PASS** | Covered in `tests/test_smoke_tests.py`; manual accuracy matches evaluation/report accuracy |
| V-G2 | Control task calibration | **PASS** | `EXP-0.3` verdicts are NEGATIVE for both control tasks |
| V-G3 | Trivial task ceiling | **PASS** | `EXP-0.2` decision tree/logistic regression reach 100% accuracy; verdict is MODERATE under current operationalization |
| V-G4 | Data-model isolation | **PASS** | Runner smoke validation confirms fresh per-task dataset generation |

Full suite status: **414 tests passing**.
