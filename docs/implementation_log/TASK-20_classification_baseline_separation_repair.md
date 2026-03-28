# TASK-20: Classification Baseline Separation Repair

**Date:** 2026-03-27
**Task ID:** TASK-20
**Status:** COMPLETE

---

## Objective

Repair the remaining classification credibility gap identified in the methodology review: `C2.1_and_rule` was behaving like an easy, learnable task, but its sampled class prior made the majority baseline too competitive and left the task labeled `WEAK`.

The acceptance target for TASK-20 was:

1. verify the imbalance diagnosis,
2. repair the task without weakening the global SR-8 verdict logic,
3. rerun the affected experiment stack,
4. refresh the publication-facing docs and manuscript inputs so the new result is reflected consistently.

---

## Inputs Read Before Implementation

Per the standard continuation flow, this task was grounded in:

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/EXPERIMENT_CATALOG.md`
- `docs/EXPERIMENT_DESIGN.md`
- `docs/METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md`
- `docs/PREPUBLICATION_ANALYSIS_2026-03-26.md`

I also checked the most recent merged-PR review context before changing code. The actionable carry-forward items were:

- remove the local workspace path from publication-facing markdown,
- resolve the stale 32-task versus 30-scored-task ambiguity explicitly,
- make sure old review comments were not being silently reintroduced.

---

## Diagnosis

`C2.1_and_rule` is defined as:

- positive iff `x1 > 50` and `cat1 == "A"`
- negative otherwise

Under the original uniform sampler:

- `P(x1 > 50) ~= 0.5`
- `P(cat1 == "A") ~= 1/3`
- therefore `P(positive) ~= 1/6`

That means the task prior itself pushed the majority-class baseline close to `0.86` IID accuracy. In practice, the pre-repair artifact showed:

- majority-class IID accuracy: `0.8600`
- best IID accuracy: `0.9978`
- baseline gap: `0.1378`

Since criterion 3 in SR-8 requires a gap of at least `0.15`, the task was failing baseline separation for prior-skew reasons, not because the task was genuinely hard.

---

## Code Changes

### 1. Task-local sampler repair in `src/registry.py`

Added `_cls_balanced_and_rule_sampler(schema)` and wired it into `C2.1_and_rule`.

Design choice:

- keep the rule unchanged,
- keep the global SR-8 baseline-separation threshold unchanged,
- change only the task-local sampling prior,
- explicitly cover the positive region and all three negative regions.

The sampler now:

- samples positives at roughly 50% frequency via `x1 in (50, 100]` and `cat1 = "A"`,
- samples negatives by covering:
  - `x1 <= 50, cat1 = "A"`
  - `x1 > 50, cat1 in {"B", "C"}`
  - `x1 <= 50, cat1 in {"B", "C"}`
- leaves unused features such as `x2` sampled normally from the schema.

This was logged as:

- `DEV-022` in `docs/EXPERIMENT_CATALOG.md`
- `ADR-033` in `docs/ARCHITECTURE_DECISIONS.md`

### 2. Regression tests

Added one new registry-level and one new data-generator-level guardrail:

- `tests/test_registry.py`
  - `test_c2_1_and_rule_sampler_balances_labels`
- `tests/test_data_generator.py`
  - `test_c2_1_and_rule_dataset_is_approximately_balanced`

These protect both the sampler itself and the generated dataset distribution.

---

## Validation

### Targeted tests

- `python -m pytest tests/test_registry.py -q`
  - `36 passed`
- `python -m pytest tests/test_data_generator.py -q`
  - `24 passed`

### Distribution spot-check

Using the repaired registry and data generator:

- sampler-level positive fraction over 1000 seeds: `0.514`
- generated-dataset class balance over 900 samples:
  - `YES = 0.5133`
  - `NO = 0.4867`

### Full validation suite

- `python -m pytest`
  - `465 passed, 17 warnings`

Warnings remained optimization/runtime related only.

---

## Experiment Reruns

TASK-20 reran the affected experiment stack:

- `python main.py classification --output-root results`
- `python main.py diagnostic --output-root results`
- `python main.py bonus --output-root results`
- `python scripts/generate_publication_assets.py`

These were sufficient because:

- the task definition changed for a classification task,
- diagnostics consume baseline classification artifacts,
- bonus rule extraction includes `C2.1_and_rule`,
- sequence and smoke code paths were unchanged.

---

## Empirical Outcome

### `C2.1_and_rule`

Post-repair artifact (`results/EXP-C2/per_task/C2.1_and_rule/metrics.json`):

- best IID model: `random_forest`
- best IID accuracy: `1.0000`
- majority baseline IID accuracy: `0.5015`
- IID gap: `0.4985`
- best noise accuracy: `0.9452`
- best value-extrapolation accuracy: `1.0000`
- verdict: `STRONG`
- score: `0.7248`

This is the expected methodological outcome:

- the task was already learnable,
- now the artifact bundle reflects that cleanly.

### Classification benchmark shift

From refreshed `output/publication_assets/data/baseline_track_summary.csv`:

- classification now stands at:
  - `3 STRONG`
  - `9 MODERATE`
  - `1 INCONCLUSIVE`
  - `1 NEGATIVE`
- there are now **0 weak classification tasks**

### Bonus consistency

`EXP-B1` still passes 9/12 tasks overall, and `C2.1_and_rule` remains cleanly recoverable:

- best depth: `2`
- best rule-extraction accuracy: `0.9993`
- extracted structure uses only the relevant features `cat1` and `x1`

That strengthens the interpretation that the old `WEAK` label was a methodology artifact, not a task-design problem.

---

## Documentation Updates

Updated:

- `docs/PROJECT_STATUS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/EXPERIMENT_CATALOG.md`
- `docs/METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md`
- `docs/PREPUBLICATION_ANALYSIS_2026-03-26.md`

Key doc changes:

- promoted the classification story from `2 STRONG / 1 WEAK` to `3 STRONG / 0 WEAK`,
- removed the local workspace path from the publication-facing markdown analysis,
- documented the difference between:
  - 32 tasks in the registry,
  - 30 scored tasks in the baseline solvability tables,
- queued `TASK-21` as the next methodology execution step.

---

## Remaining Follow-Up

TASK-20 is complete, but it leaves one immediate methodology upgrade still open:

- `TASK-21`: add a baseline-visible distractor split for classification tasks

That is now the best next step because criterion 7 still relies mainly on diagnostic evidence instead of appearing directly in the baseline suite.

---

## Files Changed

- `src/registry.py`
- `tests/test_registry.py`
- `tests/test_data_generator.py`
- `docs/PROJECT_STATUS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/EXPERIMENT_CATALOG.md`
- `docs/METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md`
- `docs/PREPUBLICATION_ANALYSIS_2026-03-26.md`
- `docs/implementation_log/TASK-20_classification_baseline_separation_repair.md`

Generated artifacts refreshed:

- `results/EXP-C1`
- `results/EXP-C2`
- `results/EXP-C3`
- `results/EXP-D1`
- `results/EXP-D2`
- `results/EXP-D3`
- `results/EXP-D4`
- `results/EXP-D5`
- `results/EXP-B1`
- `results/EXP-B2`
- `output/publication_assets/`
