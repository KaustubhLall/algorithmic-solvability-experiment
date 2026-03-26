# Algorithmic Solvability Experiment

A Python research app for testing whether machine learning models can detect that a task is governed by a compact deterministic algorithm. It generates synthetic input/output data from known reference algorithms, runs controlled experiments, and reports solvability evidence from generalization behavior.

## Who It's For

- ML researchers and experimentation engineers studying systematic generalization on controlled synthetic tasks

## What It Does

- Defines both sequence and classification task tracks with typed schemas and DSLs.
- Registers benchmark tasks with deterministic reference algorithms and verifiers.
- Generates labeled datasets with reproducible seeds and optional input noise.
- Creates train/test splits including IID, length, value, and noise shifts.
- Trains multiple model families through a unified model harness.
- Evaluates runs with task-aware metrics, error taxonomies, and metadata breakdowns.
- Writes experiment artifacts, plots, markdown summaries, and solvability verdicts.

## How It Works

| Component | Repo evidence |
|---|---|
| CLI | `main.py` runs the TASK-11 smoke workflow by default and writes artifacts to `results/`. |
| Task layer | `src/schemas.py`, `src/registry.py`, and `src/dsl/*` define schemas, tasks, and program/rule DSLs. |
| Data flow | `src/data_generator.py` samples inputs, applies reference algorithms, and verifies labels before splits. |
| Execution | `src/splits.py` -> `src/models/harness.py` -> `src/evaluation.py` -> `src/runner.py` orchestrate split, train, predict, score, and aggregate. |
| Reporting | `src/reporting.py` serializes metrics, summaries, plots, and solvability verdict outputs. |

## Project Status

The implementation is past the original scaffold stage. The smoke suite, sequence baseline suite, and classification baseline suite are all landed, and the next planned step is TASK-14 diagnostics. For the current state, see `docs/PROJECT_STATUS.md`.

## How To Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the default smoke suite from the repo root:

```bash
python main.py
```

4. Optional commands:

```bash
python main.py smoke --output-root results
python main.py sequence
python main.py classification
python -m pytest -q
```

If you are using the checked-in virtualenv on Windows, the local interpreter path is:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Repo Notes

- Results are written under `results/<experiment-id>/`.
- The default registry currently contains 32 benchmark tasks across the implemented S0-S3 and C0-C3 tiers.
- The unified harness currently exposes 9 model families.

## Not Found In Repo

Deployed service, web UI, database, API server, and production hosting details are not present in this repository.
