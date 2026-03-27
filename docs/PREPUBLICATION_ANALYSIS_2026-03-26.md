# Prepublication Analysis

**Date:** 2026-03-26  
**Workspace:** `C:\Users\kaust\PycharmProjects\DataScience`

## Scope

This document is the publication-facing analysis layer for the current benchmark state. It is based on:

- a fresh successful rerun of `pytest`, `main.py smoke`, `main.py sequence`, `main.py classification`, `main.py diagnostic`, and `main.py bonus`,
- the derived publication asset bundle in `output/publication_assets/`,
- direct inspection of the generated figures, checklist, and result summaries.

The goal here is not to restate experiment logs. It is to decide what the data actually supports, what the paper should claim, what must be disclosed as a limitation, and how the manuscript should be structured for peer review.

## Artifact Completeness

The derived publication asset bundle is complete for the implemented benchmark:

- result prerequisites are present for all required baseline, diagnostic, and bonus experiments,
- all derived data tables were regenerated into `output/publication_assets/data/`,
- all publication figures were regenerated into `output/publication_assets/figures/`,
- the checklist in `output/publication_assets/data/publication_checklist.md` confirms complete figure and table generation.

One reporting limitation remains:

- runtime values are recoverable from 9 of 16 experiment summaries,
- `EXP-D1` through `EXP-D5` and `EXP-B1` through `EXP-B2` do not currently log runtime in their summary artifacts,
- operational runtime claims in the paper should therefore be described as shell-measured wall-clock observations from the fresh rerun, not as values parsed from all result summaries.

## Fresh Execution Pass

The current codebase passed the full test suite on the fresh rerun:

- `460 passed`
- `17 warnings`

Warnings remain concentrated in the training stack rather than in correctness:

- `MLPClassifier` convergence warnings,
- `LogisticRegression` convergence warnings,
- scikit-learn warnings on high-cardinality sequence outputs.

Fresh shell-measured wall-clock runtimes from this pass were:

| Workflow | Wall-clock time |
|---|---:|
| `pytest -q` | about `77.4s` |
| `main.py smoke` | about `69.5s` |
| `main.py sequence` | about `175.7s` |
| `main.py classification` | about `157.1s` |
| `main.py diagnostic` | about `490.2s` |
| `main.py bonus` | about `59.9s` |

That places the full experiment rerun at about `15.9` minutes, or about `17.2` minutes including the validation test pass.

## Benchmark Inventory

The implemented benchmark currently covers 30 tasks across 8 implemented tiers:

| Track | Tiers | Task count |
|---|---|---:|
| Sequence | `S0` `S1` `S2` `S3` | 16 |
| Classification | `C0` `C1` `C2` `C3` | 14 |

Tier-level inventory:

| Track | Tier | Tasks |
|---|---|---:|
| Sequence | `S0` | 1 |
| Sequence | `S1` | 8 |
| Sequence | `S2` | 4 |
| Sequence | `S3` | 3 |
| Classification | `C0` | 1 |
| Classification | `C1` | 5 |
| Classification | `C2` | 5 |
| Classification | `C3` | 3 |

This matters for the paper framing: the manuscript should present the benchmark as an implemented `S0-S3` and `C0-C3` study, not as evidence over the full aspirational tier design in the planning docs.

## Baseline Findings

Baseline track summaries from `output/publication_assets/data/baseline_track_summary.csv`:

| Track | Tasks | Mean score | Mean best IID | Mean best OOD | Negative | Weak | Inconclusive | Moderate | Strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequence | 16 | 0.3359 | 0.3195 | 0.2135 | 12 | 2 | 2 | 0 | 0 |
| Classification | 14 | 0.6753 | 0.9355 | 0.9702 | 1 | 1 | 1 | 11 | 0 |

### Interpretation

The main empirical split in the benchmark is now very clear:

- the classification track already provides broad evidence of learnability and extrapolation on deterministic tabular rule tasks,
- the sequence track does not yet provide broad evidence of learned algorithmic solvability under the current model family.

### Strongest baseline classification evidence

The strongest baseline classification tasks are clustered in `C1-C3`:

| Task | Baseline label | Baseline score |
|---|---|---:|
| `C2.3_nested_if_else` | `MODERATE` | 0.7535 |
| `C1.2_range_binning` | `MODERATE` | 0.7480 |
| `C3.3_rank_based` | `MODERATE` | 0.7466 |
| `C1.1_numeric_threshold` | `MODERATE` | 0.7291 |
| `C2.6_categorical_gate` | `MODERATE` | 0.7271 |

The baseline classification story is not "all tasks are solved." It is "most implemented tabular rule tasks are solved well enough to meet the paper's moderate-evidence criterion."

### Weakest classification cases

The paper should explicitly surface the unresolved or fragile classification cases:

- `C0.1_random_class` remains `NEGATIVE`, which is a healthy control result.
- `C1.6_modular_class` remains `INCONCLUSIVE` with materially weaker generalization than the rest of the classification track.
- `C2.1_and_rule` is only `WEAK`, despite near-perfect raw accuracy, because it does not satisfy the full evidence bundle.

### Best sequence cases

Only two sequence tasks reach `WEAK`:

- `S2.2_balanced_parens`
- `S1.4_count_symbol`

The best sequence tasks show that the pipeline can extract real signal from some symbolic tasks, but the benchmark-wide learned-sequence story remains weak. That distinction is important for publication: a few promising sequence tasks do not justify a general claim of algorithmic sequence learning.

## Diagnostic Findings

Diagnostic overview from `output/publication_assets/data/diagnostic_overview.csv`:

| Diagnostic | Unit | Evaluated | Positive outcomes | Rate |
|---|---|---:|---:|---:|
| `EXP-D1` sample efficiency | tasks | 8 | 6 | 0.75 |
| `EXP-D2` distractor robustness | tasks | 5 | 4 | 0.80 |
| `EXP-D3` noise robustness | tasks | 4 | 4 | 1.00 |
| `EXP-D4` feature alignment | task-model pairs | 9 | 9 | 1.00 |
| `EXP-D5` label changes | tasks | 30 | 1 | 0.0333 |

### What diagnostics strengthen

Diagnostics clearly strengthen the classification result section:

- sample efficiency separates the intended algorithmic tasks from controls,
- distractor robustness is strong on most tested classification tasks,
- noise robustness is uniformly strong on the tested subset,
- feature-importance alignment is perfect on the tested task-model subset.

The diagnostics also reveal something useful about the sequence track:

- some sequence tasks show strong sample-efficiency signatures even when they do not clear the full solvability threshold.

That is scientifically useful because it argues against the simplistic interpretation "sequence tasks fail because the benchmark is bad." The more plausible interpretation is "sequence tasks are algorithmically coherent, but the current learned models do not generalize broadly enough."

### What diagnostics do not justify

The manuscript should not imply that diagnostics transformed the overall conclusion. They did not.

Calibration overview from `output/publication_assets/data/calibration_overview.csv`:

| Scope | Tasks | Upgrades | Downgrades | Unchanged |
|---|---:|---:|---:|---:|
| All tasks | 30 | 1 | 0 | 29 |
| Sequence | 16 | 0 | 0 | 16 |
| Classification | 14 | 1 | 0 | 13 |

Only one task changed label after calibration:

- `C1.1_numeric_threshold`: `MODERATE -> STRONG`

So the correct publication reading is:

- diagnostics increase confidence in several task-level mechanisms,
- diagnostics do not materially alter the benchmark-wide conclusion,
- the paper should treat calibration as confirmatory, not revolutionary.

### Most important diagnostic failure

The clearest targeted weakness remains:

- `C3.1_xor` fails distractor robustness.

This is worth highlighting in the paper because it shows the benchmark is not simply rewarding high raw accuracy. A task can look strong at baseline and still reveal brittle feature selection under intervention.

## Bonus Recovery Findings

Bonus summary from `output/publication_assets/data/bonus_summary.csv`:

| Experiment | Tasks evaluated | Tasks passing | Pass rate |
|---|---:|---:|---:|
| `EXP-B1` rule extraction | 12 | 9 | 0.7500 |
| `EXP-B2` program search | 9 | 7 | 0.7778 |

This is one of the most important sections for the final paper because it helps interpret the sequence-learning weakness.

The key reading is:

- symbolic recovery is substantially stronger than learned sequence generalization,
- many sequence tasks appear algorithmically recoverable even when the current learned sequence models do not generalize well,
- the benchmark therefore looks more like a model-capacity or representation challenge than a benchmark-design failure.

That is a stronger and more defensible claim than saying the system has broadly solved algorithmic sequence learning.

## What the Paper Can Honestly Claim

The data supports the following publication-grade claims:

1. On the implemented `C0-C3` tabular benchmark, standard tabular models provide strong empirical evidence of algorithmic solvability on most deterministic classification tasks.
2. On the implemented `S0-S3` symbolic-sequence benchmark, learned sequence models do not yet show broad algorithmic solvability.
3. Diagnostic interventions mostly reinforce the classification results and expose at least one meaningful fragility (`C3.1_xor` under distractors).
4. Bonus symbolic recovery succeeds often enough to suggest that benchmark task design is stronger than current sequence-model generalization.

The data does **not** support these stronger claims:

- that machine learning broadly detects algorithmic solvability across modalities,
- that the sequence track is solved,
- that diagnostics upgrade the benchmark-wide conclusion,
- that all classification rule families are equally easy.

## What Still Needs Disclosure

To make the manuscript peer-review robust, the paper should disclose the following limitations directly:

- training warnings remain in the stack and reflect optimization debt even though correctness is intact,
- diagnostics cover targeted subsets, not every task in the benchmark,
- runtime logging is incomplete in several result summaries,
- the current benchmark evidence only covers implemented tiers `S0-S3` and `C0-C3`,
- the sequence conclusions are model-family dependent and should not be overgeneralized to all neural sequence architectures.

## Recommended Manuscript Structure

This is the structure the paper should follow.

### 1. Introduction

- Define the question as empirical detection of algorithmically solvable mappings from examples.
- Distinguish function approximation, extrapolative generalization, and symbolic recovery.
- Preview the main asymmetry: strong classification evidence, weak learned sequence evidence, stronger symbolic recovery than sequence learning.

### 2. Benchmark Design

- Describe the two-track setup.
- Report implemented benchmark inventory with the `S0-S3` and `C0-C3` scope table.
- Summarize the solvability labeling rule and the role of IID, OOD, and diagnostics.

### 3. Experimental Setup

- Describe the model families actually used in the repo.
- Describe split families and the benchmark evidence criteria.
- Include the fresh rerun and validation note.

### 4. Baseline Results

- Present the verdict distribution by track.
- Present the task-level IID vs OOD figure.
- Emphasize that classification is broadly successful and sequence is not.

### 5. Diagnostics

- Present D1-D5 as targeted stress tests rather than as a second benchmark.
- Use the diagnostic matrix figure.
- Highlight the single promotion to `STRONG` and the XOR distractor failure.

### 6. Bonus Symbolic Recovery

- Present `EXP-B1` and `EXP-B2`.
- Use this section to argue that weak sequence learning should not be read as weak benchmark construction.

### 7. Limitations and Threats to Validity

- Training warnings,
- subset coverage in diagnostics,
- incomplete runtime logging,
- implemented-tier scope,
- model-family dependence of sequence conclusions.

### 8. Reproducibility

- Point to the exact output bundle in `output/publication_assets/`,
- note that figures and summary tables were regenerated from result artifacts,
- include the checklist and fresh rerun command set.

## Recommended Figure Set

Recommended main-text figures:

1. `output/publication_assets/figures/baseline_verdict_distribution.png`
2. `output/publication_assets/figures/iid_vs_ood_accuracy.png`
3. `output/publication_assets/figures/diagnostic_matrix.png`
4. `output/publication_assets/figures/bonus_success_rates.png`

Recommended appendix or secondary figures:

1. `output/publication_assets/figures/accuracy_by_track.png`
2. `output/publication_assets/figures/calibrated_label_shift.png`
3. `output/publication_assets/figures/top_calibrated_scores.png`

## Bottom Line

The project is now in a state where a real paper can be written from verified assets rather than from an execution narrative.

The central submission-ready conclusion should be:

- the benchmark already yields convincing evidence for many deterministic tabular classification tasks,
- the current learned sequence stack remains the scientific bottleneck,
- symbolic recovery results suggest the benchmark is ahead of the present sequence learners rather than the other way around.
