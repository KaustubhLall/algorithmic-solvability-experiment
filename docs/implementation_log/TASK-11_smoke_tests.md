# TASK-11: Smoke Tests (EXP-0.x)

> **Status:** COMPLETE
> **Builds:** EXP-0.1/0.2/0.3
> **Validation:** V-G1 through V-G4
> **Spec:** `EXPERIMENT_CATALOG.md` Part 2 (EXP-0.1/0.2/0.3), Part 3 (V-G1 through V-G4), Part 4 (TASK-11)
> **Date Started:** 2026-03-25
> **Date Completed:** 2026-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| Smoke experiment specifications and runners | `src/smoke_tests.py` | COMPLETE |
| Reproducible CLI entrypoint | `main.py` | COMPLETE |
| Raw-sequence LSTM support for sequence smoke | `src/models/harness.py` | COMPLETE |
| Automated V-Global coverage | `tests/test_smoke_tests.py` | COMPLETE |
| Smoke artifacts | `results/EXP-0.1/`, `results/EXP-0.2/`, `results/EXP-0.3/` | COMPLETE |

---

## Implementation Notes

- Added `ModelFamily.LSTM` and a PyTorch-backed `LSTMSequenceModel` to `src/models/harness.py`. Classical sklearn families still use the fixed-feature encoder; the LSTM family now trains directly on raw integer sequences.
- Kept the EXP-0.1 sort task bounded to the cataloged length-4-8 regime by cloning `S1.2_sort` into a smoke-local registry instead of changing the benchmark-wide task definition.
- Added `src/smoke_tests.py` to hold the TASK-11 experiment specs, smoke-local registry construction, and artifact-writing helpers.
- Replaced the placeholder `main.py` with a minimal CLI so `python main.py smoke --output-root results` reproduces the smoke suite.
- Added `tests/test_smoke_tests.py` to encode both the experiment expectations and the global checks:
  - EXP-0.1 artifact creation and LSTM threshold,
  - EXP-0.2 trivial-task ceiling,
  - EXP-0.3 control-task calibration,
  - V-G1 round-trip accuracy consistency,
  - V-G4 data-model isolation.

---

## Acceptance Criteria Results

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| EXP-0.1 | Pipeline runs; LSTM >90% exact match on sort | `results/EXP-0.1` generated. LSTM exact match = `0.9050`, token accuracy = `0.9773`; MLP exact match = `0.0150`. | YES |
| EXP-0.2 | Decision tree 100% on C1.1 | `results/EXP-0.2` generated. Decision tree IID/noise accuracy = `1.0000` across 5 seeds; logistic regression also = `1.0000`; report verdict = `MODERATE`. | YES |
| EXP-0.3 | Random tasks produce near-chance accuracy | `C0.1_random_class` accuracy = `0.3100`; `S0.1_random_labels` exact match = `0.0000`; both verdicts = `NEGATIVE`. | YES |
| V-G1 | Round-trip manual accuracy matches evaluation/report output | `tests/test_smoke_tests.py::test_vg1_round_trip_accuracy_matches_report` passes. | YES |
| V-G2 | Control tasks produce WEAK/NEGATIVE verdicts | Both control tasks in `EXP-0.3` are `NEGATIVE`. | YES |
| V-G3 | Trivial-task ceiling validated | `EXP-0.2` reaches perfect tree/logistic accuracy and a `MODERATE` verdict under the current TASK-10 operationalization. | YES |
| V-G4 | No cross-task data contamination | `tests/test_smoke_tests.py::test_vg4_runner_generates_fresh_data_per_task` passes. | YES |
| Full validation | No regressions in prior tasks | `.venv\Scripts\python.exe -m pytest -q` -> `403 passed, 9 warnings`. | YES |

---

## Deviations from Plan

- **DEV-009 logged in `docs/EXPERIMENT_CATALOG.md`.** The original EXP-0.2 smoke spec was single-seed IID only, but under ADR-017 that setup cannot demonstrate baseline separation, extrapolation success, or seed stability.
- TASK-11 therefore expanded EXP-0.2 to include:
  - `majority_class` as a floor baseline,
  - a `NOISE` split,
  - 5 seeds.
- Under the same operationalization, V-G3 is satisfied by `MODERATE` or better rather than strictly `STRONG`, because criteria 6-9 are intentionally unmeasured at this stage.

---

## Completion Summary

TASK-11 converted the previously validated modules into a reproducible smoke gate. The key new capability was a real raw-sequence LSTM path so the sequence smoke test could clear the sort threshold without rewriting the whole harness API. With that in place, `src/smoke_tests.py` and `main.py` now run the three smoke experiments end-to-end and emit artifacts into `results/EXP-0.1` through `results/EXP-0.3`. The final artifacts show bounded sort is learnable by the new LSTM (`90.5%` exact match), trivial tabular rules saturate (`100%` for decision tree and logistic regression), and the control tasks remain negative, which clears the smoke-test gate for TASK-12.
