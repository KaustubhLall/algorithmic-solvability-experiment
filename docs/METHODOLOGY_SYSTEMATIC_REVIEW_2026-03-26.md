# Methodology Review: Post-Implementation Assessment and Upgrade Plan

> **Status:** Cross-checked against the 2026-03-27 TASK-20 reruns, refreshed `output/publication_assets/`, and the second-paper manuscript source
> **Date:** 2026-03-27
> **Scope:** Assess the implemented `S0-S3` and `C0-C3` benchmark after TASK-20, identify which methodology-review concerns are now resolved, and prioritize the next execution work toward a stronger second paper.

---

## 1. Executive Summary

The repo is now in a materially better methodology state than the first prepublication pass and better than the post-TASK-19 state.

- **Action 1.1 is complete.** Baseline SR-8 reporting and EXP-D5 use the same D1/D2/D4 evidence resolver.
- **Action 1.2 is complete.** The sequence LSTM now trains for 200 epochs with weight decay and `ReduceLROnPlateau`, uses 5 seeds on `EXP-S1` and `EXP-S2`, and logs per-epoch training curves.
- **Action 1.3 is complete.** `C2.1_and_rule` no longer suffers from trivial-baseline dominance caused by an imbalanced task prior.
- **Runtime coverage is complete.** The publication asset bundle has parsed runtime values for all 16 required experiments.
- **The paper narrative is now evidence-backed.** Publication-facing analysis and manuscript updates can be sourced directly from the refreshed asset bundle.

The main scientific conclusion remains asymmetric:

- **Classification is strong within the implemented scope.** On the executed `C0-C3` scored benchmark, the current baseline distribution is 3 `STRONG`, 9 `MODERATE`, 1 `INCONCLUSIVE`, and 1 `NEGATIVE`.
- **Sequence learning improved but is still the main bottleneck.** On the executed `S0-S3` benchmark, the current distribution is 11 `NEGATIVE`, 1 `MODERATE`, 2 `WEAK`, and 2 `INCONCLUSIVE`.
- **Symbolic recovery remains stronger than learned sequence generalization.** EXP-B1 passes 9/12 classification tasks and EXP-B2 recovers exact programs for 7/9 searched sequence tasks.

The next high-leverage execution task is now **TASK-21: baseline distractor split support**.

---

## 2. Current Rerun-Backed State

### 2.1 Validation and Execution Status

The current codebase passes the full validation suite:

- `465 passed`
- `17 warnings`

The warnings remain optimization-oriented rather than correctness-oriented:

- `LogisticRegression` convergence warnings
- `MLPClassifier` convergence warnings
- high-cardinality scikit-learn warnings on sequence outputs

TASK-20 reran the affected experiment stack:

- `python main.py classification --output-root results`
- `python main.py diagnostic --output-root results`
- `python main.py bonus --output-root results`
- `python scripts/generate_publication_assets.py`

From `output/publication_assets/data/experiment_runtimes.csv`, the total artifact-parsed experiment runtime is now approximately **7517.2s** (`125.3` minutes).

### 2.2 Implemented Benchmark Scope

The implemented benchmark remains intentionally narrower than the aspirational design:

| Scope type | Sequence | Classification | Total |
|---|---:|---:|---:|
| Registry tasks | 17 | 15 | 32 |
| Baseline scored tasks | 16 | 14 | 30 |

The 32-vs-30 distinction matters and should now be stated explicitly:

- the registry includes two constant controls (`S0.2_lookup_table`, `C0.2_majority_class`),
- the baseline solvability tables summarize the 30 scored tasks used in the reporting bundle.

### 2.3 Baseline Track Summary

From `output/publication_assets/data/baseline_track_summary.csv`:

| Track | Tasks | Mean score | Mean best IID | Mean best OOD | Negative | Weak | Inconclusive | Moderate | Strong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequence | 16 | 0.3551 | 0.3807 | 0.2591 | 11 | 2 | 2 | 1 | 0 |
| Classification | 14 | 0.6792 | 0.9356 | 0.9702 | 1 | 0 | 1 | 9 | 3 |

The repo now supports a cleaner split claim:

- classification tasks are broadly learnable under the current model families and evaluation criteria,
- sequence tasks are not broadly solved even after the training-protocol upgrade,
- the classification result section no longer has an obvious baseline-separation confound.

---

## 3. What Changed After TASK-20

### 3.1 `C2.1_and_rule` Is No Longer Methodologically Undersold

TASK-20 addressed the clearest remaining classification credibility gap.

Under the old sampler:

- `C2.1_and_rule` was approximately `83/17` negative/positive,
- the majority baseline reached about `0.86` IID accuracy,
- criterion 3 failed even though the task was otherwise clearly learnable.

Under the repaired sampler:

- label balance is approximately `0.514 YES / 0.486 NO`,
- the majority baseline falls to `0.5015` IID accuracy,
- the best IID model reaches `1.0000`,
- the IID baseline-separation gap rises to `0.4985`,
- the verdict moves from `WEAK` to `STRONG`.

This is the right kind of methodology repair:

- the rule is unchanged,
- the reporting thresholds are unchanged,
- the benchmark now reflects learnability rather than class-prior skew.

### 3.2 Classification Is Now Cleaner to Defend

The most important classification-task states are now:

| Task | Best model | Best IID | Best OOD | Label | Interpretation |
|---|---|---:|---:|---|---|
| `C1.1_numeric_threshold` | `decision_tree` | 1.000 | 1.000 | `STRONG` | Strong single-threshold baseline result |
| `C2.1_and_rule` | `random_forest` | 1.000 | 1.000 | `STRONG` | TASK-20 repair removed prior-skew artifact |
| `C2.6_categorical_gate` | `decision_tree` | 0.999 | 1.000 | `STRONG` | Strong mixed categorical-numeric rule learning |
| `C2.3_nested_if_else` | `decision_tree` | 0.997 | 1.000 | `MODERATE` | Strong task with no extra diagnostic promotion |
| `C1.6_modular_class` | `random_forest` | 0.829 | 0.773 | `INCONCLUSIVE` | Still the main non-control classification ambiguity |

That gives the paper a cleaner classification message:

- strong positive evidence on deterministic tabular rule tasks,
- one remaining inconclusive task,
- one control that stays negative,
- no remaining weak classification verdict.

### 3.3 Diagnostics and Bonus Recovery Remain Consistent

From `output/publication_assets/data/diagnostic_overview.csv`:

| Diagnostic | Evaluated | Positive outcomes | Rate |
|---|---:|---:|---:|
| `EXP-D1` sample efficiency | 8 | 6 | 0.75 |
| `EXP-D2` distractor robustness | 5 | 4 | 0.80 |
| `EXP-D3` noise robustness | 4 | 4 | 1.00 |
| `EXP-D4` feature alignment | 9 | 9 | 1.00 |
| `EXP-D5` label changes | 30 | 0 | 0.00 |

From `output/publication_assets/data/bonus_summary.csv`:

| Experiment | Tasks evaluated | Tasks passing | Pass rate |
|---|---:|---:|---:|
| `EXP-B1` rule extraction | 12 | 9 | 0.7500 |
| `EXP-B2` program search | 9 | 7 | 0.7778 |

Interpretation:

- diagnostics continue to strengthen the classification story more than the sequence story,
- EXP-D5's zero label changes remain healthy because the baseline already includes the same diagnostic evidence,
- symbolic recovery still outperforms learned sequence generalization,
- `C2.1_and_rule` now joins the clean symbolic-recovery classification cases rather than sitting as an awkward weak verdict.

---

## 4. Methodology Review: What Is Now Resolved vs. Still Open

### 4.1 Resolved Since TASK-16

The following methodology concerns are now substantially resolved:

| Concern | Previous state | Current state |
|---|---|---|
| Verdict wiring drift | Baseline SR-8 and EXP-D5 could disagree | Unified resolver in SR-8 and D5 |
| Short sequence training horizon | 20-epoch LSTM with no scheduler | 200-epoch LSTM, weight decay, scheduler, 5 seeds |
| Missing training-dynamics evidence | No per-epoch monitoring | Per-epoch `training_curve` artifacts plus figure |
| Incomplete runtime reporting | Missing runtime metadata in some experiments | 16/16 runtime coverage |
| Narrative drift in paper docs | Prepublication text lagged behind reruns | Asset-driven manuscript workflow |
| `C2.1_and_rule` baseline separation | Majority prior masked learnability | Balanced sampler; task now `STRONG` |

### 4.2 Still Open

The following issues remain real and should be stated plainly:

| Issue | Why it still matters | Best next action |
|---|---|---|
| No baseline distractor split | Criterion 7 is still visible mainly through diagnostics rather than baseline runs | TASK-21 baseline distractor split support |
| No Transformer or other stronger sequence architecture | Current sequence conclusions are still model-family dependent | Phase 2 architecture expansion |
| No process/trace supervision | Stateful S2/S3 tasks still lack the most natural supervision signal | Phase 3 trace support |
| Diagnostic coverage remains partial | Optional criteria still do not cover all tasks evenly | Expand D1/D4 subsets and baseline distractor coverage |

---

## 5. Updated Action Plan

### 5.1 Immediate Execution Queue

| Priority | Action | Status | Why it matters now |
|---|---|---|---|
| 1 | **Action 1.4: Add baseline distractor split coverage** | `NEXT` | Moves criterion 7 from diagnostic-only to baseline-visible evidence |
| 2 | **Action 2.1: Add Transformer sequence baseline** | `PENDING` | Needed to separate task hardness from LSTM mismatch |
| 3 | **Action 2.2/2.3: PE ablation and scaled LSTM** | `PENDING` | Clarifies whether length/position bias is the dominant sequence failure mode |
| 4 | **Action 3.1: Trace/process supervision support** | `PENDING` | Needed for stateful S2/S3 tasks |

### 5.2 Action Status Table

| Action | Description | Status | Evidence |
|---|---|---|---|
| 1.1 | Wire D1/D2/D4 into baseline verdicting | `COMPLETE` | TASK-17, baseline and D5 now agree |
| 1.2 | Upgrade LSTM training budget and log curves | `COMPLETE` | TASK-18, training-dynamics assets, improved sequence labels |
| 1.3 | Repair `C2.1_and_rule` baseline separation | `COMPLETE` | TASK-20, task now `STRONG` with unchanged verdict logic |
| 1.4 | Add `DISTRACTOR` split to baseline classification runs | `READY` | Still needed for direct criterion-7 visibility |
| 2.1 | Add Transformer sequence model | `PENDING` | No architecture comparison yet |
| 2.2 | Positional-encoding ablation | `PENDING` | No PE-conditioned length-generalization evidence yet |
| 2.3 | Scaled LSTM variant | `PENDING` | Needed to isolate capacity vs. architecture mismatch |
| 3.1 | Execution-trace support | `PENDING` | Still absent for S2/S3 stateful tasks |
| 3.2 | Architecture-conditioned verdict matrix | `PENDING` | Best-model collapse still hides architecture sensitivity |
| 3.3 | Adversarial boundary split | `PENDING` | Classification OOD remains easier than it could be |
| 3.4 | Deferred tier implementation (`S4/S5/C4/C5`) | `PENDING` | Benchmark scope remains narrower than the design roadmap |

### 5.3 Recommended Next Task

**TASK-21: Baseline Distractor Split Support**

Recommended scope:

1. implement a baseline-visible distractor split for classification tasks,
2. thread it into the TASK-13 experiment specs and SR-7/SR-8 artifact pipeline,
3. rerun classification, diagnostics if needed, and publication assets,
4. update the paper so criterion 7 is evidenced directly in the baseline suite instead of primarily through diagnostics.

This is now the highest-value next step because it strengthens the remaining classification-methodology gap without reopening the solved baseline-separation issue.

---

## 6. Checklist

### Completed in the current methodology pass

- [x] Action 1.1 completed and rerun-backed
- [x] Action 1.2 completed and rerun-backed
- [x] Action 1.3 completed and rerun-backed
- [x] `pytest`
- [x] `python main.py classification --output-root results`
- [x] `python main.py diagnostic --output-root results`
- [x] `python main.py bonus --output-root results`
- [x] `python scripts/generate_publication_assets.py`
- [x] Runtime coverage verified at 16/16 experiments
- [x] Publication-facing markdown analysis updated from the refreshed asset bundle

### Still pending

- [ ] Action 1.4 baseline distractor split support
- [ ] Phase 2 sequence architecture expansion
- [ ] Phase 3 trace supervision and adversarial split work

---

## 7. Bottom Line

The methodology situation is now healthier in the ways that matter most for a defensible paper:

- the reporting logic is consistent,
- the sequence stack has a more serious baseline protocol,
- the publication asset bundle is complete,
- the classification story no longer has an obvious baseline-separation artifact.

The science is also clearer:

- classification is genuinely strong within the implemented benchmark,
- sequence learning improved but remains the main unsolved component,
- symbolic recovery continues to suggest that the benchmark is ahead of the current learned sequence models.

That makes the next move straightforward. The repo does **not** need another round of narrative cleanup first. It needs targeted execution work, starting with **TASK-21**.

---

## 8. Selected References

1. [Lake and Baroni (2018), "Generalization without Systematicity"](https://arxiv.org/abs/1711.00350)
2. [Kim and Linzen (2020), "COGS"](https://aclanthology.org/2020.emnlp-main.731/)
3. [Velickovic and Blundell (2021), "Neural Algorithmic Reasoning"](https://arxiv.org/abs/2105.02761)
4. [Velickovic et al. (2022), "CLRS Algorithmic Reasoning Benchmark"](https://arxiv.org/abs/2205.15659)
5. [Power et al. (2022), "Grokking: Generalization Beyond Overfitting"](https://arxiv.org/abs/2201.02177)
6. [Nanda et al. (2023), "Progress Measures for Grokking via Mechanistic Interpretability"](https://arxiv.org/abs/2301.05217)
7. [Grinsztajn et al. (2022), "Why do tree-based models still outperform deep learning on tabular data?"](https://arxiv.org/abs/2207.08815)
8. [Balog et al. (2017), "DeepCoder"](https://openreview.net/forum?id=ByldLrqlx)
