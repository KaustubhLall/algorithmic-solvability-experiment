# IMPLEMENTATION LOG SUMMARY

> **PURPOSE:** A running, human-readable summary of all implementation work completed so far.
> This is the "what got built and what was learned" companion to `PROJECT_STATUS.md`.
> Detailed per-task logs live in `docs/implementation_log/TASK-XX_<name>.md`.
> Link each completed task from the table below to its detailed log.
>
> **Last Updated:** 2025-03-25 (TASK-01 complete)
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

---

## Running Lessons Learned

Surprising findings, non-obvious edge cases, or things that would save time in future chats.

- **Use tuples, not lists, in frozen dataclasses.** Lists are not hashable, so frozen `@dataclass` fields must use tuples. This affects all callers constructing specs.
- **`np.random.default_rng(seed)` is the right API.** Modern NumPy generator API gives full reproducibility. Each `sample()` call creates its own generator from the seed.
- **Batch sampling uses sequential draws from one RNG.** `sample_batch(seed, n)` is NOT the same as `[sample(seed+i) for i in range(n)]`. The batch method is more efficient and produces better-distributed samples.

---

## Current Blockers

Issues actively blocking progress and needing resolution before the next task can start.

_None. TASK-01 complete, TASK-02 ready to start._

---

## Test / Validation Summary

Quick record of which validation procedures (V-1 through V-10 + V-Global) are passing.

| Validation | Component | Status | Notes |
|---|---|---|---|
| V-1 | Task Registry | NOT RUN | |
| V-2 | Input Schema | **PASS** ✓ | 52 tests, all passing (4.13s) |
| V-3 | Data Generator | NOT RUN | |
| V-4 | Split Generator | NOT RUN | |
| V-5 | Model Harness | NOT RUN | |
| V-6 | Evaluation Engine | NOT RUN | |
| V-7 | Experiment Runner | NOT RUN | |
| V-8 | Report Generator | NOT RUN | |
| V-9 | Classification Rule DSL | NOT RUN | |
| V-10 | Sequence DSL | NOT RUN | |
| V-G1 | Round-trip check | NOT RUN | |
| V-G2 | Control task calibration | NOT RUN | |
| V-G3 | Trivial task ceiling | NOT RUN | |
| V-G4 | Data-model isolation | NOT RUN | |
