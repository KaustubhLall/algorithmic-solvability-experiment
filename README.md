# Algorithmic Solvability Experiment

> **Goal:** Determine whether machine learning models can detect that a task is governed by a compact deterministic algorithm, by training on synthetic input-output pairs and testing for systematic generalization beyond the training distribution.

## Quick Start

**To continue implementation in a new chat:**

1. Clone the repository
2. Open `START_HERE.MD` and follow the instructions
3. Or read `docs/PROJECT_STATUS.md` first for the current state

## What This Is

A comprehensive experimental framework for testing algorithmic solvability detection:

- **Two task tracks:** Sequence (variable-length tokens) and Classification (mixed numerical/categorical tabular inputs)
- **Synthetic ground truth:** Every task has a known reference algorithm that generates all labels
- **Systematic generalization tests:** Length extrapolation, value-range shift, unseen feature combinations, adversarial splits
- **Evidence criteria:** STRONG/MODERATE/WEAK/NEGATIVE/INCONCLUSIVE solvability verdicts
- **Full validation pipeline:** Every component is validated before proceeding to the next

## Document Structure

```
docs/
├── PROJECT_STATUS.md             ← START HERE: current task, progress, file structure
├── EXPERIMENT_DESIGN.md          ← Design rationale, task tiers, models, metrics
├── EXPERIMENT_CATALOG.md         ← Full spec, execution plan, validation procedures
├── ARCHITECTURE_DECISIONS.md     ← Append-only ADR log
├── IMPLEMENTATION_LOG_SUMMARY.md ← Running summary and validation status
└── implementation_log/           ← Per-task detailed logs
```

## Execution Plan

15 implementation tasks (TASK-01 → TASK-15) with clear dependencies:

- **TASK-01:** Input Schema System (start here)
- **TASK-02–03:** Classification and Sequence DSLs (parallel)
- **TASK-04:** Task Registry [MILESTONE: FOUNDATION]
- **TASK-05–06:** Data Generator + Split Generator [MILESTONE: DATA PIPELINE]
- **TASK-07–10:** Model Harness → Report Generator [MILESTONE: FULL PIPELINE]
- **TASK-11:** Smoke Tests [GATE: must pass before experiments]
- **TASK-12–13:** Core experiments (parallel)
- **TASK-14:** Diagnostics
- **TASK-15:** Bonus algorithm discovery

## Current Status

- ✅ Complete design and execution infrastructure
- ✅ All documentation and scaffolding
- 🔄 Ready to begin implementation (TASK-01: Input Schema System)
- ❌ No code written yet

## Why This Matters

Many ML systems appear to learn patterns, but it's unclear whether they're learning the underlying algorithm or just surface correlations. This framework provides a controlled way to test algorithmic understanding with known ground truth and systematic out-of-distribution tests.

## Repository

**Public URL:** https://github.com/KaustubhLall/algorithmic-solvability-experiment

## License

MIT License — see LICENSE file for details.
