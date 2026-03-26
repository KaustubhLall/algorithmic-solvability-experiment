# PROJECT STATUS: Algorithmic Solvability Experiment

> **PURPOSE:** This is the single entry-point document for any new implementation chat.
> Read this first, then follow the links to detailed documents.
> Update this document at the end of every chat session.
>
> **Last Updated:** 2025-03-25
> **Current Phase:** Pre-implementation (documentation complete, no code written yet)

---

## Quick Context

**Goal:** Determine whether ML models can detect that a task is governed by a compact deterministic algorithm, by training on synthetic input/output pairs and testing for systematic generalization beyond the training distribution.

**Two task tracks:**
- **Sequence track (S-tiers):** Variable-length discrete token sequences → transformed sequences
- **Classification track (C-tiers):** Mixed numerical + categorical tabular inputs → categorical label

**Two authoritative design documents:**
- `docs/EXPERIMENT_DESIGN.md` — *what* to build and *why* (task tiers, models, metrics, evidence criteria)
- `docs/EXPERIMENT_CATALOG.md` — *how* to build it (shared resources SR-1–10, experiments, validation V-1–10, execution plan TASK-01–15, deviation log)

---

## Current Task

| Field | Value |
|---|---|
| **Next task to implement** | TASK-01: Input Schema System |
| **Status** | NOT STARTED |
| **Blocked by** | Nothing — this is the first task |
| **Relevant spec** | `EXPERIMENT_CATALOG.md` Part 1 (SR-2), Part 4 (TASK-01), Part 3 (V-2) |

---

## Implementation Progress

| Task | Scope | Status | Notes |
|---|---|---|---|
| TASK-01 | Input Schema (SR-2) | NOT STARTED | |
| TASK-02 | Classification Rule DSL (SR-9) | NOT STARTED | Depends on TASK-01 |
| TASK-03 | Sequence DSL (SR-10) | NOT STARTED | Depends on TASK-01 |
| TASK-04 | Task Registry (SR-1) | NOT STARTED | Depends on TASK-01–03 |
| TASK-05 | Data Generator (SR-3) | NOT STARTED | Depends on TASK-04 |
| TASK-06 | Split Generator (SR-4) | NOT STARTED | Depends on TASK-05 |
| TASK-07 | Model Harness (SR-5) | NOT STARTED | Depends on TASK-01, 05 |
| TASK-08 | Evaluation Engine (SR-6) | NOT STARTED | Depends on TASK-07 |
| TASK-09 | Experiment Runner (SR-7) | NOT STARTED | Depends on TASK-04–08 |
| TASK-10 | Report Generator (SR-8) | NOT STARTED | Depends on TASK-09 |
| TASK-11 | Smoke Tests (EXP-0.x) | NOT STARTED | **GATE** — depends on TASK-10 |
| TASK-12 | Sequence Experiments | NOT STARTED | Depends on TASK-11 |
| TASK-13 | Classification Experiments | NOT STARTED | Depends on TASK-11 |
| TASK-14 | Diagnostic Experiments | NOT STARTED | Depends on TASK-12–13 |
| TASK-15 | Bonus: Algorithm Discovery | NOT STARTED | Depends on TASK-14 |

**Milestone Gates:**
- `[ ]` FOUNDATION complete (TASK-01–04 done + V-1, V-2, V-9, V-10 passing)
- `[ ]` DATA PIPELINE complete (TASK-05–06 done + V-3, V-4 passing)
- `[ ]` FULL PIPELINE complete (TASK-07–10 done + V-5 through V-8 passing)
- `[ ]` SMOKE TEST GATE cleared (TASK-11 done + V-G1–G4 passing)

---

## Actual File Structure (update as built)

```
DataScience/
├── docs/
│   ├── EXPERIMENT_DESIGN.md          # Design rationale and task spec
│   ├── EXPERIMENT_CATALOG.md         # Execution plan, SR specs, validation
│   ├── PROJECT_STATUS.md             # THIS FILE — entry point for new chats
│   ├── IMPLEMENTATION_LOG_SUMMARY.md # Running summary of all implementation work
│   ├── ARCHITECTURE_DECISIONS.md     # All architectural/design decisions made during impl
│   └── implementation_log/           # Per-task detailed logs
│       ├── TASK-01_input_schema.md
│       ├── TASK-02_classification_dsl.md
│       ├── TASK-03_sequence_dsl.md
│       ├── TASK-04_task_registry.md
│       ├── TASK-05_data_generator.md
│       ├── TASK-06_split_generator.md
│       ├── TASK-07_model_harness.md
│       ├── TASK-08_evaluation_engine.md
│       ├── TASK-09_experiment_runner.md
│       ├── TASK-10_report_generator.md
│       └── TASK-11_smoke_tests.md
├── src/                              # Source code (to be created during implementation)
│   ├── schemas.py                    # SR-2: Input Schema
│   ├── registry.py                   # SR-1: Task Registry
│   ├── data_generator.py             # SR-3: Data Generator
│   ├── splits.py                     # SR-4: Split Generator
│   ├── evaluation.py                 # SR-6: Evaluation Engine
│   ├── runner.py                     # SR-7: Experiment Runner
│   ├── reporting.py                  # SR-8: Report Generator
│   ├── dsl/
│   │   ├── classification_dsl.py     # SR-9: Classification Rule DSL
│   │   └── sequence_dsl.py           # SR-10: Sequence DSL
│   └── models/
│       ├── harness.py                # SR-5: Model Harness
│       ├── configs.py
│       └── encoders.py
├── results/                          # Experiment outputs (auto-generated)
├── tests/                            # Validation test suite (mirrors V-1 through V-10)
└── main.py
```

> Update the `src/` and `tests/` sections above with actual file paths as they are created.

---

## Known Deviations from Plan

_None yet. See `EXPERIMENT_CATALOG.md` Part 5 (Deviation Log) for structured entries._

---

## How to Continue in a New Chat

Paste the following as your first message in a new implementation chat:

```
Continue implementing the algorithmic solvability experiment from where we left off.

Read these documents in order:
1. docs/PROJECT_STATUS.md        — current task, progress table, file structure
2. docs/ARCHITECTURE_DECISIONS.md — all architectural decisions made so far
3. docs/IMPLEMENTATION_LOG_SUMMARY.md — running summary of completed work
4. docs/EXPERIMENT_CATALOG.md    — full spec: SR interfaces, TASK details, validation
5. docs/EXPERIMENT_DESIGN.md     — design rationale (reference as needed)

Check the Deviation Log in EXPERIMENT_CATALOG.md Part 5 before starting.
Then implement the next incomplete task shown in PROJECT_STATUS.md.
After completing the task, update PROJECT_STATUS.md, IMPLEMENTATION_LOG_SUMMARY.md,
and write a detailed log in docs/implementation_log/TASK-XX_<name>.md.
```
