# Second-Paper Analysis

**Date:** 2026-03-27

## Scope

This document is the publication-facing analysis layer for the current benchmark state. It is based on:

- the latest successful `pytest` pass (`465 passed`, `17 warnings`),
- fresh TASK-20 reruns of `classification`, `diagnostic`, and `bonus`,
- the regenerated asset bundle in `output/publication_assets/`,
- the unchanged TASK-18 sequence and TASK-19 manuscript-support artifacts where no code changed,
- direct interpretation of the current figure and table outputs.

Its job is to answer four questions:

1. what the current evidence supports,
2. what the paper should say,
3. what the paper should avoid claiming,
4. which figures and tables are worth carrying into the manuscript.

## Artifact Completeness

The publication asset bundle is complete for the implemented benchmark:

- all required baseline, diagnostic, and bonus experiment artifacts are present,
- all required derived tables and figures were regenerated successfully,
- runtime coverage is available for 16 of 16 required experiments,
- the checklist in `output/publication_assets/data/publication_checklist.md` is fully checked.

This means the paper can now rely on one coherent evidence bundle instead of a mixture of hand-written summaries.

## Fresh TASK-20 Execution Summary

The current validation and affected-rerun state is:

| Workflow | Status | Runtime |
|---|---|---:|
| `pytest` | pass | `72.2s` |
| `main.py classification --output-root results` | pass | `181.9s` |
| `main.py diagnostic --output-root results` | pass | `2711.8s` |
| `main.py bonus --output-root results` | pass | `59.9s` |
| `python scripts/generate_publication_assets.py` | pass | `1.4s` |

The artifact-parsed runtime total across the 16 tracked experiments is now approximately `7517.2s` (`125.3` minutes).

## Implemented Benchmark Inventory

There are two different counts that need to be stated clearly:

- the registry currently contains **32 tasks** across implemented tiers `S0-S3` and `C0-C3`,
- the baseline solvability tables cover **30 scored tasks** because one constant control per track (`S0.2_lookup_table`, `C0.2_majority_class`) sits outside the baseline reporting set.

The manuscript should use the scored-task counts when discussing baseline verdict distributions, while making the full registry count explicit in the reproducibility/setup description.

## Baseline Findings

From `output/publication_assets/data/baseline_track_summary.csv`:

| Track | Tasks | Mean score | Mean best IID | Mean best OOD | Negative | Weak | Inconclusive | Moderate | Strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequence | 16 | 0.3551 | 0.3807 | 0.2591 | 11 | 2 | 2 | 1 | 0 |
| Classification | 14 | 0.6792 | 0.9356 | 0.9702 | 1 | 0 | 1 | 9 | 3 |

### Reading the split

The classification/sequence asymmetry is now cleaner than in the earlier draft:

- classification is broadly positive within the implemented benchmark,
- sequence remains the limiting regime even after the TASK-18 training upgrade,
- the last weak classification edge case has been removed without loosening the reporting criteria.

### Strongest classification evidence

The clearest positive cases are now:

- `C1.1_numeric_threshold` (`STRONG`, decision tree, IID `1.000`, OOD `1.000`)
- `C2.6_categorical_gate` (`STRONG`, decision tree, IID `0.999`, OOD `1.000`)
- `C2.1_and_rule` (`STRONG`, random forest, IID `1.000`, OOD `1.000`)
- `C2.3_nested_if_else` (`MODERATE`, decision tree, score `0.7535`)
- `C3.3_rank_based` (`MODERATE`, logistic regression, score `0.7466`)

`C2.1_and_rule` is the key TASK-20 result. Under the repaired sampler:

- the sampler label balance is about `0.514 YES / 0.486 NO`,
- the majority baseline drops to `0.5015` IID accuracy,
- the best IID model reaches `1.0000`,
- the IID baseline-separation gap is now `0.4985`,
- the verdict moves from `WEAK` to `STRONG` under the unchanged SR-8 logic.

The paper should now frame classification as strong evidence of algorithmic solvability on deterministic tabular rule tasks within the executed scope. It should still avoid claiming universal success: `C1.6_modular_class` remains `INCONCLUSIVE`, and the control remains `NEGATIVE`.

### Strongest sequence evidence

The sequence story is unchanged from TASK-18 in substance:

- `S1.4_count_symbol` remains `MODERATE` with IID `0.991` and OOD `0.908`,
- `S1.5_parity` remains `WEAK` with IID `0.977` and OOD `0.947`,
- `S2.2_balanced_parens` remains `WEAK` with IID `0.997` and OOD `0.990`,
- `S2.3_running_min` remains `INCONCLUSIVE` with IID `0.644` and OOD `0.000`.

This still supports the same careful conclusion:

- the current sequence stack can learn some algorithmic structure,
- but it still does not justify a broad positive claim about learned algorithmic sequence generalization.

## Training-Dynamics Findings

TASK-18 remains central to the paper because the training-dynamics assets show how the upgraded LSTM behaves, not just where it ends.

The key figure is `output/publication_assets/figures/sequence_training_dynamics.png`.

What it still shows:

- `S1.4_count_symbol` and `S2.2_balanced_parens` saturate early,
- `S1.2_sort` keeps improving on held-out exact match after the validation-selected checkpoint,
- `S2.3_running_min` shows the clearest delayed improvement, with best held-out epochs well after the checkpointed epoch.

That gives the paper a precise sequence claim:

- the short-horizon protocol was underestimating sequence potential,
- longer training helps,
- but the remaining gap is still too large to attribute sequence weakness only to training budget.

## Diagnostic Findings

From `output/publication_assets/data/diagnostic_overview.csv`:

| Diagnostic | Evaluated | Positive outcomes | Rate |
|---|---:|---:|---:|
| `EXP-D1` sample efficiency | 8 | 6 | 0.75 |
| `EXP-D2` distractor robustness | 5 | 4 | 0.80 |
| `EXP-D3` noise robustness | 4 | 4 | 1.00 |
| `EXP-D4` feature alignment | 9 | 9 | 1.00 |
| `EXP-D5` label changes | 30 | 0 | 0.00 |

Diagnostics should be described as:

- mechanistic support for the classification story,
- a consistency check on the reporting logic,
- partial evidence that the negative sequence results are not merely benchmark pathology.

The best example of ongoing fragility remains `C3.1_xor`, which still fails distractor robustness. That is worth keeping in the paper because it shows the benchmark can expose brittle feature use, not just headline wins.

## Bonus Recovery Findings

From `output/publication_assets/data/bonus_summary.csv`:

| Experiment | Tasks evaluated | Tasks passing | Pass rate |
|---|---:|---:|---:|
| `EXP-B1` rule extraction | 12 | 9 | 0.7500 |
| `EXP-B2` program search | 9 | 7 | 0.7778 |

The bonus section remains important because it supports the benchmark-validity argument:

- benchmark tasks are often symbolically recoverable,
- learned sequence models still underperform that symbolic recoverability,
- the benchmark therefore looks stronger than the current sequence learners.

TASK-20 adds one more concrete classification example here: `EXP-B1` recovers `C2.1_and_rule` with best depth `2`, best accuracy `0.9993`, and a structurally faithful rule using only `cat1` and `x1`.

## What the Second Paper Can Claim

The current data supports the following claims:

1. On the implemented `C0-C3` tabular benchmark, standard tabular models now provide strong empirical evidence of algorithmic solvability on most deterministic classification tasks.
2. On the implemented `S0-S3` symbolic benchmark, learned sequence models still do not provide broad evidence of algorithmic solvability.
3. TASK-18's longer LSTM training protocol materially improved several sequence tasks and changed the interpretation of earlier negative results, but did not erase the broader sequence gap.
4. TASK-20 removed a genuine methodology confound in `C2.1_and_rule` without weakening the verdict logic, which strengthens the credibility of the classification result section.
5. Symbolic recovery succeeds often enough to suggest that the main bottleneck is model capability or inductive bias, not benchmark incoherence.

## What the Second Paper Should Not Claim

The manuscript should avoid these stronger claims:

- that machine learning broadly detects algorithmic solvability across modalities,
- that the sequence track is solved,
- that the current repo has resolved architecture sensitivity on sequence tasks,
- that diagnostic success is exhaustive across all tasks,
- that the executed benchmark covers the full long-term roadmap.

## Required Disclosures

To be peer-review robust, the manuscript should state the following limitations directly:

- the executed evidence covers only implemented tiers `S0-S3` and `C0-C3`,
- the baseline verdict tables summarize 30 scored tasks even though the registry contains 32 tasks total,
- convergence warnings remain in the training stack,
- diagnostics still cover targeted subsets rather than every task,
- baseline distractor evidence is still partly diagnostic-backed rather than baseline-native,
- no Transformer, scratchpad, or trace-supervised sequence model has been tested yet.

## Recommended Manuscript Structure

The second paper should now be organized as follows:

### 1. Introduction

- define the question as empirical evidence for algorithmic solvability from examples,
- distinguish learned generalization from symbolic recovery,
- preview the classification/sequence asymmetry.

### 2. Benchmark Scope and Methodology Repairs

- describe the implemented `S0-S3` and `C0-C3` scope,
- explain the 32-registered versus 30-scored-task distinction,
- summarize what TASK-17, TASK-18, and TASK-20 changed methodologically.

### 3. Baseline Results

- include the verdict-distribution figure,
- include the IID-vs-OOD figure,
- emphasize that classification is broadly positive while sequence is not.

### 4. Sequence Training Dynamics

- include the training-dynamics figure,
- explain early saturation versus delayed held-out gains,
- argue that longer training helps but does not remove the architecture gap.

### 5. Diagnostics and Symbolic Recovery

- summarize D1-D5,
- highlight `C3.1_xor` distractor fragility,
- explain why bonus recovery strengthens the benchmark interpretation.

### 6. Limitations and Next Steps

- disclose scope limits and model-family dependence,
- point to baseline distractor-split support and sequence architecture expansion as the next steps.

## Figures Worth Carrying into the Paper

These remain the highest-value figures for the manuscript:

- `output/publication_assets/figures/baseline_verdict_distribution.png`
- `output/publication_assets/figures/iid_vs_ood_accuracy.png`
- `output/publication_assets/figures/sequence_training_dynamics.png`
- `output/publication_assets/figures/diagnostic_matrix.png`
- `output/publication_assets/figures/bonus_success_rates.png`

If the paper needs to be shortened, the first cut should still be `bonus_success_rates.png`, not the training-dynamics figure.

## Deliverable Target

This analysis should drive the second-paper manuscript in:

- `output/pdf/algorithmic_solvability_second_paper_2026-03-26.tex`
- `output/pdf/algorithmic_solvability_second_paper_2026-03-26.pdf`

The manuscript should only be considered ready after:

- successful compilation,
- rendered-page visual inspection,
- confirmation that figure labels and tables match the refreshed asset bundle.
