# EXPERIMENT DESIGN: Can ML Detect Algorithmic Solvability?

> **STATUS: EXECUTION-READY DESIGN DOCUMENT**
> This is the authoritative design document for the algorithmic-solvability experiment.
> All implementation work should be traceable back to sections in this file.
> Read alongside `EXPERIMENT_CATALOG.md` which provides the experiment catalog,
> execution plan, and validation procedures.
>
> **Last Updated:** 2025-03-25

---

# Table of Contents

1. [Purpose and Goal](#1-purpose-and-goal)
2. [Core Questions](#2-core-questions)
3. [Important Distinctions](#3-important-distinctions)
4. [Guiding Principles](#4-guiding-principles)
5. [Input Representation Strategy](#5-input-representation-strategy)
6. [Task Tier System](#6-task-tier-system)
7. [Data Generation Strategy](#7-data-generation-strategy)
8. [Model Families](#8-model-families)
9. [Evidence Criteria for Algorithmic Solvability](#9-evidence-criteria-for-algorithmic-solvability)
10. [Dataset Split Strategy](#10-dataset-split-strategy)
11. [Metrics](#11-metrics)
12. [Major Caveats](#12-major-caveats)
13. [Bonus Objective: Algorithm Discovery](#13-bonus-objective-algorithm-discovery)
14. [Related Research](#14-related-research)

---

# 1. Purpose and Goal

## What We Are Testing

Given a dataset of input-output pairs produced by a known algorithm, can we:

1. Train ML models to predict the output from the input.
2. Test whether they generalize systematically beyond the training distribution.
3. Use that generalization behavior to infer that the underlying task is **algorithmically solvable** — meaning a compact, deterministic procedure governs the mapping.

## What We Are NOT Doing (Primary Goal)

- We are **not** trying to recover the source code of the hidden algorithm. That is a bonus.
- We are **not** building a single model that "discovers" algorithms in general.
- We are **not** restricting ourselves to one data modality. The experiment spans both sequence-symbolic tasks and tabular mixed-type classification tasks.

## Why Classification Tasks Matter

Many real-world algorithmic problems produce categorical outputs from mixed-type inputs. Examples include:

- rule-based classifiers,
- decision trees,
- threshold-based labeling,
- state-machine-driven categorization,
- feature-interaction-driven classification.

If the framework only covers sequence-to-sequence symbolic tasks, it misses an entire family of algorithmically solvable problems that look like standard tabular ML. Adding classification tiers ensures the framework generalizes to settings where inputs are heterogeneous and outputs are discrete categories.

---

# 2. Core Questions

1. **Learnability**: Which algorithmic tasks can ML models learn from examples alone?
2. **Generalization depth**: Do models generalize beyond the training distribution (size, value range, feature combinations)?
3. **Architecture sensitivity**: Which model families succeed on which task families, and why?
4. **Solvability signal**: Can the pattern of model success/failure across evaluation regimes serve as a reliable indicator that the task is governed by a compact algorithm?
5. **Classification-specific**: Can models learn rule-based classification algorithms from mixed-type tabular data, and can we distinguish algorithmic classification from statistical correlation?

---

# 3. Important Distinctions

Three claims that must not be conflated:

### 3.1 Function Approximation
A model predicts outputs accurately on held-out data drawn from the same distribution. This is necessary but insufficient.

### 3.2 Algorithmic Generalization
A model continues to succeed when input size, composition depth, feature combinations, or value ranges change beyond training support. **This is the primary target.**

### 3.3 Algorithm Discovery
A model or downstream procedure identifies a compact symbolic or executable program that generates the function. **This is a bonus.**

---

# 4. Guiding Principles

1. Use synthetic tasks where the true generating process is known exactly.
2. Control task complexity explicitly via metadata.
3. Generate unlimited data automatically from the reference algorithm.
4. Evaluate on harder test regimes than training: longer inputs, wider value ranges, unseen feature combinations, unseen compositions.
5. Compare weak baselines to strong models to calibrate difficulty.
6. Record failure modes and error types, not just aggregate accuracy.
7. Prefer exact correctness metrics where possible.
8. Support both sequence-symbolic and tabular-classification task families.
9. Keep data generation, model training, and evaluation as independent modules.
10. Design every task so that a correct reference implementation can verify any prediction.

---

# 5. Input Representation Strategy

The experiment uses two distinct input modalities.

## 5.1 Sequence-Symbolic Inputs (for Tiers S0–S5)

These are variable-length discrete token sequences.

- integer sequences,
- binary strings,
- character strings over small alphabets,
- token lists,
- small serialized structures.

**Encoding for models:**

- Sequence models (LSTM, Transformer): token embedding + positional encoding.
- Tabular baselines on sequence tasks: handcrafted summary features (length, mean, min, max, histogram bins, n-grams).

## 5.2 Tabular Mixed-Type Inputs (for Tiers C0–C5)

These are fixed-width feature vectors with a mix of numerical and categorical columns.

### Numerical features
Continuous or discrete numbers. Examples: age, temperature, count, measurement, score.

**Encoding:**

- Raw values for tree models.
- Standardized (z-score) or min-max normalized for neural models.
- Optionally: binned into ordinal categories for ablation.

### Categorical features
Discrete unordered labels from a finite set. Examples: color, region, type, flag.

**Encoding:**

- Integer label encoding for tree models.
- One-hot encoding for neural models.
- Optionally: learned embeddings for high-cardinality categoricals in neural models.

### Mixed-type feature vectors
Each sample is a row with `n_num` numerical columns and `n_cat` categorical columns.

**Key design rule:** The number and types of features are fixed per task but vary across tasks in the benchmark.

### Irrelevant / distractor features
For robustness testing, append features that are not used by the reference algorithm. The model must learn to ignore them.

---

# 6. Task Tier System

The benchmark is organized into two parallel tracks: **Sequence (S-tiers)** and **Classification (C-tiers)**. Each track has a control tier and escalating complexity.

## Track A: Sequence-Symbolic Tasks

### Tier S0: Controls (non-algorithmic)

Purpose: calibrate false positives and validate the evaluation pipeline.

- **S0.1 Random labels**: inputs mapped to uniformly random outputs.
- **S0.2 Lookup table**: small fixed table, no extrapolatable rule.
- **S0.3 Shallow statistical**: output correlates with simple statistics (e.g., mean) but has no exact rule.
- **S0.4 Spurious shortcut**: output depends on a surface feature that happens to correlate in training but not in test.

### Tier S1: Simple one-step transforms

- **S1.1 Reverse**: reverse a sequence.
- **S1.2 Sort**: sort a sequence ascending.
- **S1.3 Rotate**: rotate sequence by `k` positions.
- **S1.4 Count symbol**: count occurrences of a target value.
- **S1.5 Parity**: return 0 or 1 based on parity of the bit string.
- **S1.6 Prefix sum**: cumulative sum of the input.
- **S1.7 Deduplicate**: remove duplicates preserving first occurrence.
- **S1.8 Extrema**: return max, min, or argmax.

### Tier S2: Stateful / iterative algorithms

- **S2.1 Cumulative XOR**: running XOR over a bit string.
- **S2.2 Balanced parentheses**: classify whether a bracket string is balanced.
- **S2.3 Running min/max**: output the running minimum or maximum sequence.
- **S2.4 Finite-state matcher**: accept/reject based on a small finite automaton.
- **S2.5 Checksum**: compute a modular checksum.
- **S2.6 Binary addition**: add two bit strings with carry.

### Tier S3: Multi-step compositional

- **S3.1 Dedup-sort-count**: deduplicate, sort, count.
- **S3.2 Map-filter-reduce**: e.g., filter even → map (+1) → sum.
- **S3.3 Postfix eval**: evaluate arithmetic expressions in postfix.
- **S3.4 Run-length encode/decode**: RLE compression or decompression.
- **S3.5 Canonicalize**: normalize a structured token sequence to canonical form.

### Tier S4: Structural / graph algorithms

- **S4.1 Shortest path length**: on small serialized graphs.
- **S4.2 Connectivity check**: is the graph connected?
- **S4.3 Merge intervals**: merge overlapping intervals.
- **S4.4 Edit distance**: on short strings.
- **S4.5 Topological sort validity**: is a given ordering a valid topological sort?

### Tier S5: DSL program family

- Programs composed from a typed DSL of list primitives.
- Systematically vary depth, primitive set, and composition order.
- This tier generates many tasks from one implementation.

## Track B: Classification Tasks (Mixed Numerical + Categorical Inputs → Categorical Output)

These tiers specifically address classification problems where inputs have mixed types and the output is a discrete categorical label determined by a known algorithm.

### Tier C0: Controls (non-algorithmic classification)

Purpose: same as S0 but for tabular classification.

- **C0.1 Random class**: categorical output assigned uniformly at random regardless of input.
- **C0.2 Majority class**: output is always the most common class. Tests whether models just predict the mode.
- **C0.3 Correlated noise**: output loosely correlates with one feature but has irreducible noise — no exact rule exists.
- **C0.4 Spurious categorical**: output correlates with a categorical feature in training but the correlation is reversed in test.

### Tier C1: Single-rule threshold / boundary classification

These are the simplest algorithmic classifiers. Each uses one or two features and a deterministic rule.

- **C1.1 Numeric threshold**: `class = A if x1 > t else B`. Single feature, single cut.
- **C1.2 Range binning**: `class = bin(x1)` where bins are contiguous intervals. Multiple classes.
- **C1.3 Categorical match**: `class = A if cat1 == "red" else B`. Pure categorical rule.
- **C1.4 Majority vote**: `class = mode(cat1, cat2, cat3)`. Output is the most frequent categorical value among several features.
- **C1.5 Numeric comparison**: `class = A if x1 > x2 else B`. Compares two numeric features.
- **C1.6 Modular class**: `class = x1 mod k`. Maps a numeric feature to a class via modular arithmetic.

Why these matter:
- They are the atomic building blocks of more complex rule systems.
- They establish whether simple baselines can perfectly solve trivial rules.
- They test whether models can extrapolate a threshold beyond the training value range.

### Tier C2: Multi-feature conjunctive / disjunctive rules

These require combining evidence from multiple features, potentially of mixed types.

- **C2.1 Conjunctive (AND)**: `class = A if x1 > t1 AND cat1 == "red" else B`.
- **C2.2 Disjunctive (OR)**: `class = A if x1 > t1 OR cat1 == "red" else B`.
- **C2.3 Nested if-else**: a small hand-written decision tree with 3-5 splits over mixed features.
- **C2.4 Linear boundary**: `class = A if w1*x1 + w2*x2 > b else B`. Linear decision boundary on numeric features.
- **C2.5 k-of-n rule**: `class = A if at least k of n boolean predicates are true`.
- **C2.6 Categorical + numeric gate**: `class = A if cat1 == "type1" AND x1 > threshold_for_type1 else ...`. The threshold depends on the categorical value.

Why these matter:
- They test whether models can learn feature combinations, not just marginals.
- Conjunctive and disjunctive rules are the building blocks of rule lists and decision trees.
- The categorical gate (C2.6) is specifically important: it tests whether models can learn conditional numeric thresholds — a very common pattern in real algorithmic classifiers.

### Tier C3: Feature interaction and nonlinear classification

These tasks require learning interactions that no single feature can reveal.

- **C3.1 XOR**: `class = A if (x1 > t1) XOR (x2 > t2) else B`. Neither feature alone predicts the class.
- **C3.2 Parity over categoricals**: `class = A if an odd number of categorical features equal their target value`.
- **C3.3 Rank-based rule**: `class = A if x1 is the largest among {x1, x2, x3}`. Depends on relative rank, not absolute value.
- **C3.4 Distance-based region**: `class = A if (x1, x2) is within radius r of center c`. Circular or elliptical decision boundary.
- **C3.5 Interaction polynomial**: `class = A if x1 * x2 > t`. Multiplicative interaction.
- **C3.6 Conditional distribution shift**: the class rule changes depending on a categorical feature value. Different sub-populations have different decision boundaries.

Why these matter:
- XOR and parity are classic tests that linear models and naive baselines fail completely.
- Rank-based and distance-based rules test invariance properties.
- Conditional shift (C3.6) is critical: it tests whether models learn population-specific algorithms.

### Tier C4: Stateful / aggregation-dependent classification

These tasks require computing aggregate statistics or maintaining state before classifying.

- **C4.1 Group statistics**: input is a set of rows sharing a group key. `class = A if mean(x1) within group > threshold`. The classification depends on an aggregate over the group, not a single row.
- **C4.2 Outlier flag**: `class = outlier if x1 deviates from group mean by more than k standard deviations`. Requires computing group statistics first.
- **C4.3 Sequence pattern**: input is a short sequence of categorical events. `class = A if the pattern [cat1=X, cat2=Y] appears at any position`. Requires scanning.
- **C4.4 Cumulative threshold**: input is a sequence of numeric values. `class = A if the cumulative sum exceeds a threshold at any point`. Requires running state.
- **C4.5 Voting ensemble**: multiple sub-rules each produce a class, and the final class is the majority vote. Tests whether models can learn implicit ensembling.

Why these matter:
- They bridge tabular classification and sequence reasoning.
- They test whether models can implicitly compute aggregates.
- Group-dependent classification is common in real algorithmic systems.

### Tier C5: Multi-step compositional classification

These require chaining several operations before classification.

- **C5.1 Feature engineering pipeline**: `bin(x1) → combine with cat1 → apply rule tree → class`. The classification depends on a derived feature.
- **C5.2 Hierarchical classification**: first classify into a coarse group, then use group-specific rules for fine class.
- **C5.3 Multi-stage filter**: discard rows not matching a categorical predicate, then classify remaining by numeric rule.
- **C5.4 DSL classification programs**: generate classification rules from a typed DSL of predicates and combinators (AND, OR, NOT, threshold, match, compare, aggregate).
- **C5.5 Lookup-then-classify**: use one categorical feature to select a sub-table, then classify based on numeric features in that sub-table.

Why these matter:
- They are the closest to real-world algorithmic classification pipelines.
- They test compositional generalization in the classification setting.
- DSL-generated classification rules (C5.4) enable systematic complexity scaling.

---

# 7. Data Generation Strategy

## 7.1 Sequence Tasks: DSL Generator

Use a typed DSL over integer-list primitives.

Key primitives: `map`, `filter`, `sort`, `reverse`, `unique`, `take`, `drop`, `sum`, `count`, `max`, `min`, `parity`, `prefix_sum`, `zip`, `concat`.

Programs are sampled with controlled depth, type consistency, and deduplication.

## 7.2 Classification Tasks: Rule Generator

Build a **classification rule generator** that produces tabular datasets from algorithmic rules.

### Architecture

```
RuleGenerator
├── FeatureSchema      → defines n_num, n_cat, value ranges, cardinalities
├── RuleTree           → the algorithmic classifier (deterministic)
├── InputSampler       → generates feature vectors according to a distribution
├── Labeler            → applies the RuleTree to produce the class label
├── MetadataRecorder   → logs rule complexity, feature usage, class balance
└── Verifier           → re-applies the rule to verify every label
```

### Feature schema specification

Each task defines:

- `n_num`: number of numeric features (1–20).
- `n_cat`: number of categorical features (0–10).
- `num_ranges`: value range per numeric feature.
- `cat_cardinalities`: number of unique values per categorical feature.
- `n_irrelevant`: number of distractor features appended (0–10).
- `n_classes`: number of output classes (2–10).

### Rule specification

Rules are deterministic functions: `f(features) → class`.

Rules can be:

- threshold comparisons,
- categorical matches,
- boolean combinations (AND, OR, NOT, XOR, k-of-n),
- nested if-else trees,
- linear boundaries,
- rank-based comparisons,
- aggregation-then-threshold,
- compositions of the above.

### Class balance control

The class distribution depends on both the rule and the input distribution. Control this by:

- adjusting input sampling to target approximate balance,
- recording actual class ratios as metadata,
- testing models under both balanced and imbalanced regimes,
- never hiding class imbalance from the evaluation.

### Noise injection (for robustness testing only)

The reference algorithm is deterministic. However, to test robustness:

- add Gaussian noise to numeric features after label assignment,
- randomly flip a small fraction of categorical values,
- add purely random distractor columns.

**Important:** noise is added to inputs, never to labels. Labels remain exactly correct with respect to the clean inputs that generated them. Noisy inputs therefore have "stale" labels — this is intentional and tests whether models can cope with input perturbation.

## 7.3 Metadata Logged Per Sample

Every generated sample must record:

- task ID and tier,
- rule or program specification,
- input dimensions and types,
- output class,
- complexity score (rule depth, number of features used, number of operations),
- class balance ratio in the batch,
- whether distractor features are present,
- whether noise was applied and at what level.

---

# 8. Model Families

## 8.1 Group A: Simple Baselines

**Purpose:** establish lower bound, detect trivially solvable tasks, detect shortcuts.

| Model | Applicable to | Notes |
|---|---|---|
| Logistic regression | classification tiers | linear decision boundary only |
| Decision tree (single, shallow) | classification tiers | can learn axis-aligned splits |
| Linear model | sequence tiers (on summary features) | baseline for sequence tasks |
| k-NN | both | tests local similarity |
| Majority class / constant predictor | both | floor baseline |

### When to use
Always. Run these first on every task.

## 8.2 Group B: Standard Tabular Models

**Purpose:** test whether standard ML models designed for tabular data can learn algorithmic classification rules.

| Model | Applicable to | Notes |
|---|---|---|
| Random Forest | classification tiers | ensemble of axis-aligned trees |
| Gradient Boosted Trees (XGBoost / LightGBM) | classification tiers | strong tabular baseline |
| MLP on tabular features | classification tiers | tests whether neural nets match trees |
| CatBoost | classification tiers | native categorical feature handling |

### When to use
On all C-tier tasks. These are expected to be strong on C1–C2 and potentially C3.

### What they tell you
- Whether tree-based models can learn rule-based classifiers perfectly.
- Whether neural MLPs match or lag behind trees on algorithmic rules.
- Where interaction complexity exceeds tree capacity.

## 8.3 Group C: Sequence Models

**Purpose:** test standard neural sequence learners on S-tier tasks.

| Model | Applicable to | Notes |
|---|---|---|
| LSTM / GRU encoder-decoder | sequence tiers | recurrent baseline |
| Bidirectional LSTM | sequence tiers (non-causal) | sees full input |
| Transformer encoder | sequence tiers | attention-based |
| Transformer encoder-decoder | sequence tiers | for seq-to-seq |

### When to use
On all S-tier tasks. Optionally on C4–C5 tasks where input has sequential structure.

### What they tell you
- Whether standard sequence learners can imitate algorithmic mappings.
- How much length extrapolation is achievable.
- Whether attention helps over recurrence.

## 8.4 Group D: Structure-Aware / Algorithmic-Bias Models (Optional)

**Purpose:** test whether explicit computational inductive biases improve genuine generalization.

Candidates:

- memory-augmented networks (NTM/DNC style),
- graph neural networks for graph-tier tasks,
- pointer networks for selection tasks,
- models with intermediate supervision on execution traces.

### When to use
Only after Groups A–C have been evaluated. Use on tasks where Group C models show partial but imperfect generalization.

## 8.5 Group E: Symbolic / Neuro-Symbolic Methods (Bonus)

**Purpose:** attempt to recover the underlying algorithm or rule.

Candidates:

- decision tree extraction from trained neural model features,
- symbolic regression for numeric rules,
- DSL program search guided by neural predictions,
- rule list induction on classification tasks.

### When to use
Only on tasks where a Group B or C model has demonstrated strong algorithmic generalization. This is the bonus objective.

---

# 9. Evidence Criteria for Algorithmic Solvability

A task should be considered **empirically algorithmically solvable from examples** only if multiple criteria are satisfied.

## 9.1 Minimum Evidence (all required)

1. **High IID accuracy**: a nontrivial model achieves near-perfect accuracy on standard held-out data.
2. **Extrapolation success**: the model retains strong performance on out-of-distribution test data (longer sequences, wider value ranges, unseen feature combinations).
3. **Baseline separation**: the model significantly outperforms simple baselines on extrapolation and adversarial splits.
4. **Seed stability**: results are consistent across at least 5 random seeds.
5. **Coherent degradation**: errors increase smoothly with task complexity, not randomly.

## 9.2 Stronger Evidence (desirable)

6. **Counterfactual sensitivity**: perturbing algorithm-relevant features changes the output predictably; perturbing irrelevant features does not.
7. **Distractor robustness**: adding irrelevant features does not degrade performance.
8. **Sample efficiency**: the model learns faster than on non-algorithmic controls of similar input size.
9. **Transfer**: a model trained on one composition of primitives partially succeeds on novel compositions.

## 9.3 Strongest Evidence (bonus objective)

10. **Symbolic recovery**: a compact rule or program can be extracted that matches the reference algorithm on a hard test set.

## 9.4 Evidence Strength Labels

Use these labels in experiment reports:

| Label | Meaning |
|---|---|
| **STRONG** | Criteria 1–5 met, plus at least two of 6–9 |
| **MODERATE** | Criteria 1–5 met |
| **WEAK** | Only criterion 1 met (IID accuracy only) |
| **NEGATIVE** | All tested models fail even after representation and training controls checked |
| **INCONCLUSIVE** | Mixed results, possible confound or insufficient evaluation |

---

# 10. Dataset Split Strategy

## 10.1 Splits for Sequence Tasks

| Split | Description | Purpose |
|---|---|---|
| **IID** | Random holdout from training distribution | Sanity check |
| **Length extrapolation** | Train on lengths 4–16, test on 25–64, hard test on 65–128 | Core generalization test |
| **Value-range shift** | Train on integers [-10, 10], test on [-100, 100] | Tests value generalization |
| **Composition** | Train on depth ≤ d, test on depth d+1 | Tests compositional generalization |
| **Template holdout** | Hold out specific operation orderings | Tests structural generalization |
| **Adversarial** | Constructed to break shallow heuristics | Tests robustness |

## 10.2 Splits for Classification Tasks

| Split | Description | Purpose |
|---|---|---|
| **IID** | Random holdout from training feature distribution | Sanity check |
| **Value-range extrapolation** | Train on x ∈ [0, 50], test on x ∈ [50, 100] or [-50, 0] | Tests threshold extrapolation |
| **Unseen category combinations** | Hold out specific combinations of categorical values | Tests combinatorial generalization |
| **Feature subset shift** | Change the marginal distribution of one or more features while keeping the rule the same | Tests distributional robustness |
| **Distractor injection** | Add irrelevant features at test time (or remove them) | Tests feature selection |
| **Class imbalance shift** | Train balanced, test heavily imbalanced (or vice versa) | Tests calibration robustness |
| **Rule complexity increase** | Train on shallow rules, test on deeper compositions | Tests compositional generalization for classifiers |
| **Adversarial boundary** | Sample test points near the decision boundary | Tests precision of learned boundary |

---

# 11. Metrics

## 11.1 Core Classification Metrics

- **Accuracy** (exact match for categorical output).
- **Per-class precision, recall, F1**.
- **Macro-averaged F1** (unweighted across classes).
- **Weighted F1** (weighted by class frequency).
- **Confusion matrix** (full, not just accuracy).
- **ROC-AUC** (for binary tasks or one-vs-rest for multiclass).
- **Calibration error** (expected calibration error if probabilities are available).

## 11.2 Core Sequence Metrics

- **Exact match accuracy** (entire output sequence must be correct).
- **Token-level accuracy** (per-position correctness).
- **Mean absolute error** (for numeric scalar outputs).

## 11.3 Extrapolation Metrics

Report all core metrics broken down by:

- input length or feature count,
- value range regime,
- rule depth / composition complexity,
- presence of distractors,
- class imbalance ratio.

## 11.4 Diagnostic Metrics

- **Error type taxonomy**: off-by-one, wrong class, boundary confusion, hallucinated class, truncation, etc.
- **Failure breakpoint**: the input size or complexity at which accuracy drops below a threshold.
- **Confidence under shift**: how model confidence changes when input distribution shifts.
- **Feature importance alignment**: does the model's feature importance match the features actually used by the reference algorithm?
- **Learning curve**: accuracy as a function of training set size (sample efficiency).

## 11.5 Solvability Score

A composite per-task score for ranking. Components:

| Component | Weight | Description |
|---|---|---|
| IID accuracy | 0.15 | Basic capability |
| Extrapolation accuracy | 0.25 | Core generalization signal |
| Baseline gap | 0.15 | Advantage over simple baselines |
| Seed stability | 0.10 | Consistency |
| Distractor robustness | 0.10 | Feature selection quality |
| Sample efficiency | 0.10 | Learning speed vs. controls |
| Degradation coherence | 0.15 | Smooth complexity scaling |

This score ranks tasks by **empirical algorithmic learnability**. It does not replace raw metrics.

---

# 12. Major Caveats

## 12.1 Success may reflect shortcuts
If training and test distributions overlap too strongly, apparent algorithmic understanding may be surface-level pattern matching. Mitigate with adversarial and extrapolation splits.

## 12.2 Failure does not mean no algorithm exists
Failure means the tested model class, representation, and training regime were insufficient. It does **not** prove the task is not algorithmically solvable.

## 12.3 Some algorithms are representation-sensitive
A poor encoding (e.g., treating ordered categories as unordered, or serializing a graph poorly) can cause failure even when the task is simple.

## 12.4 Class imbalance can mask or inflate accuracy
If one class dominates, accuracy can be misleadingly high. Always report per-class metrics.

## 12.5 Noise injection must be carefully controlled
Adding noise to inputs after labeling creates "label noise" from the model's perspective. This tests robustness but can confound solvability assessment if overdone.

## 12.6 Distractor features can inflate model complexity requirements
A task may be trivially solvable given the right features but appear hard if many distractors are present. Report results with and without distractors.

## 12.7 Multi-step tasks can hide distribution collapse
As composition depth increases, many sampled programs may produce semantically equivalent outputs. Track output diversity.

## 12.8 Feature interaction tasks are hard to scale
XOR and parity over many features become exponentially hard. Keep feature counts bounded and complexity intentional.

## 12.9 Extrapolation is not always meaningful
For some classification tasks (e.g., categorical-only rules), value-range extrapolation does not apply. Use the appropriate extrapolation axis for each task type.

---

# 13. Bonus Objective: Algorithm Discovery

This is a secondary phase, attempted only after the solvability assessment is complete.

## Approach

1. Identify tasks where a model demonstrates **STRONG** or **MODERATE** algorithmic solvability.
2. For classification tasks: attempt decision tree extraction, rule list induction, or symbolic rule search.
3. For sequence tasks: attempt DSL program search or symbolic regression.
4. Validate any recovered rule/program on the hardest held-out test set.

## Why This Sequencing Matters

By first identifying which tasks show algorithmic generalization, you narrow the search space for symbolic recovery dramatically. Attempting discovery on all tasks simultaneously is wasteful.

---

# 14. Related Research

Relevant areas to consult during implementation:

- **Program induction / inductive programming**: recovering programs from examples.
- **Program synthesis from I/O examples**: DSL-based search for executable programs.
- **Neural algorithmic reasoning**: neural networks imitating classical algorithms (CLRS benchmark).
- **Systematic generalization**: compositional generalization beyond training distribution (SCAN, COGS).
- **Length extrapolation**: generalizing to longer sequences than seen in training.
- **Neuro-symbolic methods**: combining neural learning with symbolic rule extraction.
- **Tabular ML benchmarks**: comparing trees vs. neural nets on structured tabular data.
- **Rule learning / rule induction**: classic ML approaches to learning interpretable rules from data.

---

# Appendix: Glossary

| Term | Definition |
|---|---|
| **Algorithmic solvability** | The property that a task's input-output mapping is governed by a compact, deterministic algorithm |
| **IID** | Independent and identically distributed (same distribution as training) |
| **Extrapolation** | Testing on inputs outside the training distribution (larger, wider, novel combinations) |
| **DSL** | Domain-specific language — a small programming language for generating tasks |
| **Tier** | A complexity level in the task benchmark |
| **Solvability score** | A composite metric summarizing evidence for algorithmic learnability |
| **Reference algorithm** | The known ground-truth algorithm that generates the data for a task |
| **Distractor feature** | An input feature not used by the reference algorithm |
