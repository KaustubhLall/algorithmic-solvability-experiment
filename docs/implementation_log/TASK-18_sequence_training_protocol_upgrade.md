# TASK-18 Implementation Log: Sequence Training Protocol Upgrade

- **Date:** 2026-03-26
- **Task:** TASK-18
- **Status:** COMPLETE
- **Branch:** `codex/task-18-sequence-training-protocol-upgrade`

## Goal

Execute Action 1.2 from the methodology review by making the sequence-training stack serious enough for publication-backed interpretation:

- increase the LSTM training budget,
- add learning-rate scheduling and regularization,
- log per-epoch curves,
- rerun the affected suites,
- regenerate the publication asset bundle with enough evidence to write a stronger paper.

## Inputs Reviewed

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/METHODOLOGY_SYSTEMATIC_REVIEW_2026-03-26.md`
- `src/models/harness.py`
- `src/runner.py`
- `src/sequence_experiments.py`
- `scripts/generate_publication_assets.py`
- current `results/EXP-S1` through `results/EXP-S3`

## Problem Statement

TASK-16's methodology review identified two linked weaknesses:

1. the sequence LSTM was under-trained relative to the algorithmic-learning literature,
2. the publication package had no training-dynamics evidence and incomplete runtime coverage.

That left the paper vulnerable to the criticism that sequence failures might mostly reflect a weak protocol rather than task hardness.

## Implementation Summary

### 1. Upgraded the LSTM training protocol

Updated `src/models/harness.py` so the sequence LSTM now:

- trains for up to 200 epochs,
- uses weight decay,
- uses `ReduceLROnPlateau`,
- splits off an internal validation slice for checkpoint selection,
- restores the best validation-loss checkpoint,
- logs per-epoch train loss, validation loss, held-out exact match, held-out token accuracy, and learning rate.

### 2. Exposed training-curve artifacts through SR-7

Updated `src/runner.py` so `SingleRunResult` can store `training_curve`.

This let the upgraded sequence runs preserve per-seed, per-epoch evidence instead of collapsing everything to final metrics.

### 3. Strengthened the executed sequence protocol

Updated `src/sequence_experiments.py` so:

- `EXP-S1` and `EXP-S2` run with 5 seeds instead of 3,
- `EXP-S1` and `EXP-S2` use 1000 samples instead of the smaller earlier budget,
- the default LSTM config reflects the upgraded training parameters.

### 4. Closed the runtime-coverage gap in the publication bundle

Updated:

- `src/diagnostic_experiments.py`
- `src/bonus_experiments.py`
- `scripts/generate_publication_assets.py`

so that:

- diagnostic and bonus summaries now log runtime,
- runtime fallback parsing works from JSON payloads when needed,
- the publication bundle reaches full 16/16 runtime coverage,
- a new sequence-training-dynamics dataset and figure are generated:
  - `output/publication_assets/data/sequence_training_dynamics.csv`
  - `output/publication_assets/figures/sequence_training_dynamics.png`

### 5. Expanded regression coverage

Updated tests in:

- `tests/test_model_harness.py`
- `tests/test_runner.py`
- `tests/test_sequence_experiments.py`
- `tests/test_diagnostic_experiments.py`
- `tests/test_bonus_experiments.py`

to cover the richer sequence artifacts and runtime metadata.

## Files Changed

- `src/models/harness.py`
- `src/runner.py`
- `src/sequence_experiments.py`
- `src/diagnostic_experiments.py`
- `src/bonus_experiments.py`
- `scripts/generate_publication_assets.py`
- `tests/test_model_harness.py`
- `tests/test_runner.py`
- `tests/test_sequence_experiments.py`
- `tests/test_diagnostic_experiments.py`
- `tests/test_bonus_experiments.py`
- `docs/PROJECT_STATUS.md`
- `docs/IMPLEMENTATION_LOG_SUMMARY.md`
- `docs/ARCHITECTURE_DECISIONS.md`
- `docs/EXPERIMENT_CATALOG.md`

## Validation Performed

### Targeted tests

- `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_diagnostic_experiments.py`
  - Result: `22 passed`
- `.\\.venv\\Scripts\\python.exe -m pytest tests\\test_bonus_experiments.py`
  - Result: `20 passed`

### Full regression suite

- `.\\.venv\\Scripts\\python.exe -m pytest`
  - Result: `463 passed, 17 warnings`

### Experiment reruns

- `python main.py sequence --output-root results`
- `python main.py diagnostic --output-root results`
- `python main.py bonus --output-root results`
- `python main.py smoke --output-root results`
- `python main.py classification --output-root results`
- `python scripts/generate_publication_assets.py`

Key shell-measured runtimes from the fresh TASK-18 pass:

- `sequence`: about `4499.9s`
- `diagnostic`: about `2455.4s`
- `bonus`: about `52.5s`
- `smoke`: about `77.5s`
- `classification`: about `196.2s`

## Empirical Outcome

### Sequence baseline distribution improved

After the TASK-18 rerun, the implemented sequence/control benchmark moved to:

- `11 NEGATIVE`
- `1 MODERATE`
- `2 WEAK`
- `2 INCONCLUSIVE`

Key promotions:

- `S1.4_count_symbol` -> `MODERATE`
- `S1.5_parity` -> `WEAK` under the real LSTM path
- `S2.3_running_min` -> `INCONCLUSIVE`

### Training-dynamics evidence became publication-usable

The new figure showed a more nuanced story than the earlier final-metric-only summaries:

- some tasks saturate early (`S1.4_count_symbol`, `S2.2_balanced_parens`),
- harder tasks continue improving beyond the validation-selected checkpoint (`S1.2_sort`, `S2.3_running_min`),
- reported scores remain conservative because checkpoint selection uses internal validation loss rather than best held-out epoch.

### Runtime completeness was fixed

`output/publication_assets/data/publication_summary.json` now reports runtime coverage for **16/16** required experiments.

## Decisions Captured

This task produced:

- `ADR-031` in `docs/ARCHITECTURE_DECISIONS.md`
- `DEV-020` in `docs/EXPERIMENT_CATALOG.md`

## Remaining Gaps / Next Task

TASK-18 completed Action 1.2 from the methodology plan. The next queued work after the rerun-backed interpretation pass is:

- **TASK-19 - Methodology Synthesis and Second-Paper Draft**

Scientific gaps still open after TASK-18:

1. classification baseline separation still needs repair on `C2.1_and_rule`,
2. baseline distractor coverage is still not native to classification runs,
3. the sequence story is still architecture-limited because no Transformer or trace-supported model exists yet.

## Outcome

TASK-18 converted the sequence benchmark from a clearly underpowered baseline into a credible methodology testbed. It did not solve the sequence track, but it removed a serious training-protocol objection, produced the evidence needed for a more defensible paper, and completed the reproducibility story for the publication asset bundle.
