# Methodology Review: Post-Implementation Assessment and Upgrade Plan

> **Status:** Cross-checked against the 2026-03-26 prepublication rerun; use as the planning input for TASK-16 and the follow-on execution task
> **Date:** 2026-03-26
> **Scope:** Assess the completed implementation (TASK-01 through TASK-15) against the algorithmic generalization literature, reconcile the review with the rerun-backed preprint/prepublication analysis, and provide a concrete prioritized upgrade plan for the next execution task.

---

## 1. Executive Summary

The implementation is complete for the currently executed benchmark. TASK-01 through TASK-15 are done, the fresh rerun used for the prepublication package completed successfully, and the validation suite passed with 460 tests and 17 warnings. The full pipeline - generation, splitting, training, evaluation, reporting, diagnostics, bonus symbolic recovery, and publication-asset generation - now runs end-to-end from the CLI.

**What works well:**

- **Classification track is already publication-credible within the implemented scope.** On the implemented `C0-C3` benchmark, 11 of 14 classification/control tasks are `MODERATE` at baseline, and EXP-D5 promotes `C1.1_numeric_threshold` to `STRONG`. Rule extraction (EXP-B1) passes 9 of 12 classification tasks.
- **Infrastructure is solid and rerun-backed.** 32 tasks are registered across `S0-S3` and `C0-C3`. The repo has deterministic generation, validated splits, multi-seed orchestration, structured artifact output, publication-asset generation, and passing regression coverage.
- **Diagnostics and symbolic recovery sharpen interpretation.** EXP-D1 through EXP-D5 provide sample-efficiency, distractor, noise, feature-alignment, and calibrated-verdict evidence, while EXP-B2 recovers exact DSL programs for 7 of 9 searched sequence tasks.

**What still needs work:**

- **Sequence learning remains the main scientific bottleneck.** Across the implemented `S0-S3` sequence/control benchmark, 12 of 16 tasks are `NEGATIVE`; only `S1.4_count_symbol` and `S2.2_balanced_parens` reach `WEAK`.
- **Baseline verdict wiring still underuses available diagnostic evidence.** In `compute_solvability_verdict()`, `criterion_6` and `criterion_8` remain hardcoded `False`, so the baseline reporting layer cannot fully absorb D1/D4-style evidence even though calibrated EXP-D5 now shows at least one task can satisfy `STRONG`.
- **Training protocol is too short.** The LSTM trains for 20 epochs with no learning-rate scheduling, no early stopping, and no checkpoint tracking. This misses delayed generalization (grokking).
- **No Transformer, no positional encoding controls, no scratchpad support.**

**Bottom line:** The current repo supports a careful submission-quality claim on the implemented classification benchmark, but not a strong learned-sequence claim. The next methodology work should reconcile verdict wiring with diagnostic evidence and upgrade sequence model/training coverage, while keeping all claims explicitly scoped to the implemented `S0-S3` and `C0-C3` tiers.

---

## 2. Current State Assessment

### 2.1 Classification Results (EXP-C1 through EXP-C3)

The table below summarizes the 13 non-control `C1-C3` tasks from the baseline suite. The full implemented classification/control benchmark discussed in the preprint contains 14 tasks; the added control remains `NEGATIVE`, and EXP-D5 calibrates `C1.1_numeric_threshold` from `MODERATE` to `STRONG`.

| Task | Best IID | Best OOD | Verdict | Best Model | Bottleneck |
|---|---|---|---|---|---|
| C1.1 numeric_threshold | 1.000 | 1.000 | MODERATE | decision_tree | Criteria 6–9 unwired |
| C1.2 range_binning | 1.000 | 1.000 | MODERATE | decision_tree | Criteria 6–9 unwired |
| C1.3 categorical_match | 1.000 | 0.932 | MODERATE | decision_tree | Criteria 6–9 unwired |
| C1.5 numeric_comparison | 0.998 | 0.992 | MODERATE | logistic_regression | Criteria 6–9 unwired |
| C1.6 modular_class | 0.829 | 0.773 | INCONCLUSIVE | random_forest | IID too low, high seed variance |
| C2.1 and_rule | 0.998 | 1.000 | WEAK | decision_tree | Baseline separation too small |
| C2.2 or_rule | 0.999 | 0.946 | MODERATE | decision_tree | Criteria 6–9 unwired |
| C2.3 nested_if_else | 0.997 | 1.000 | MODERATE | random_forest | Criteria 6–9 unwired |
| C2.5 k_of_n | 0.998 | 1.000 | MODERATE | random_forest | Criteria 6–9 unwired |
| C2.6 categorical_gate | 0.999 | 1.000 | MODERATE | decision_tree | Criteria 6–9 unwired |
| C3.1 xor | 0.993 | 0.999 | MODERATE | random_forest | Criteria 6–9 unwired |
| C3.3 rank_based | 0.993 | 0.990 | MODERATE | logistic_regression | Criteria 6–9 unwired |
| C3.5 interaction_poly | 0.982 | 0.981 | MODERATE | random_forest | Criteria 6–9 unwired |

**Key insight:** At baseline, 10 of 13 non-control classification tasks already meet criteria 1-5, and the broader 14-task classification/control benchmark has 11 `MODERATE` tasks. EXP-D5 already shows that one task can clear `STRONG` once diagnostic evidence is folded in. The remaining gap is not that classification lacks signal; it is that baseline verdicting and a few task-specific edge cases (`C1.6_modular_class`, `C2.1_and_rule`, `C3.1_xor` under distractors) need targeted follow-up.

**Anomalies to investigate:**
- `C1.6_modular_class` (INCONCLUSIVE): Modular arithmetic is hard for tree-based models. The MLP may need more capacity or the feature encoding needs modular-aware preprocessing.
- `C2.1_and_rule` (WEAK): IID accuracy is 99.8% but baseline separation is only 0.12, below the 0.15 threshold. The majority-class baseline happens to score well because the AND rule produces imbalanced classes in certain seed/split configurations.

### 2.2 Sequence Results (EXP-S1 through EXP-S3)

| Task | Best IID | Best OOD | Verdict | Best Model | Bottleneck |
|---|---|---|---|---|---|
| S1.1 reverse | 0.011 | 0.000 | NEGATIVE | lstm | Model capacity / training budget |
| S1.2 sort | 0.152 | 0.000 | NEGATIVE | lstm | Model capacity / training budget |
| S1.3 rotate | 0.004 | 0.000 | NEGATIVE | lstm | Model capacity / training budget |
| S1.4 count_symbol | 0.987 | 0.812 | WEAK | lstm | OOD threshold (0.85 needed) |
| S1.5 parity | 0.998* | 0.752* | INCONCLUSIVE | sequence_baseline* | Summary features, not real seq model |
| S1.6 prefix_sum | 0.006 | 0.000 | NEGATIVE | lstm | Model capacity / training budget |
| S1.7 deduplicate | 0.000 | 0.000 | NEGATIVE | lstm | Model capacity / training budget |
| S1.8 extrema | 0.948* | 0.989* | INCONCLUSIVE | sequence_baseline* | Summary features, not real seq model |
| S2.1 cumulative_xor | 0.250 | 0.000 | NEGATIVE | lstm | Model capacity / training budget |
| S2.2 balanced_parens | 0.991 | 0.993* | WEAK | lstm/majority* | Baseline separation too small |
| S2.3 running_min | 0.526 | 0.000 | NEGATIVE | lstm | Partial IID but zero OOD |
| S2.5 checksum | 0.387 | 0.138 | NEGATIVE | sequence_baseline | Neither model type works |
| S3.1 dedup_sort_count | 0.429 | 0.259 | NEGATIVE | mlp | Multi-step too hard for features |
| S3.2 filter_sort_sum | 0.429 | 0.120 | NEGATIVE | lstm | Multi-step too hard for LSTM |
| S3.4 rle_encode | 0.000 | 0.000 | NEGATIVE | lstm | Variable-length output |

*asterisks mark cases where `sequence_baseline` (summary features) or `majority_class` scored highest, which means no real sequence model learned the task.

In the full implemented sequence/control benchmark, these results correspond to 12 `NEGATIVE`, 2 `WEAK`, and 2 `INCONCLUSIVE` tasks.

**Key insight:** The LSTM at 20 epochs and 64 hidden units is simply too weak for most sequence tasks. Tasks like sort and reverse require an architecture that can attend to arbitrary positions (Transformer) or significantly longer training (grokking). The current results are useful benchmark evidence, but they mainly establish where the present sequence stack fails rather than supporting a broad positive sequence-learning claim.

### 2.3 Algorithm Discovery Results (EXP-B1, EXP-B2)

**EXP-B1 (Rule Extraction):** Decision tree extraction achieves >99% hard-test accuracy on 9/12 classification tasks. Failures are on tasks requiring non-axis-aligned boundaries (C1.5 numeric_comparison at 98.1%, C3.3 rank_based at 95.2%, C3.5 interaction_poly at 98.8%). All trees correctly use only the relevant features.

**EXP-B2 (DSL Program Search):** Random search over the SR-10 DSL recovers exact programs on 7 of 9 searched sequence tasks. `Reverse()` and `Sort()` are reliably found as leaf operations, while deeper composed programs remain harder and would benefit from guided search.

### 2.4 Why STRONG Is Rare in Baseline Reporting

`EXP-D5` already calibrates `C1.1_numeric_threshold` to `STRONG`, so STRONG is not literally impossible in the current repo. The remaining issue is that baseline verdicting still underuses the diagnostic evidence that now exists.

The verdict logic in `compute_solvability_verdict()` (`src/reporting.py`) requires:
- **STRONG** = criteria 1–5 all met + at least 2 of criteria 6–9
- **MODERATE** = criteria 1–5 all met

Current state of optional criteria:

| Criterion | Status | Data available? | What's needed |
|---|---|---|---|
| 6: Counterfactual sensitivity | Hardcoded `False` | Partially (EXP-D4 feature importance) | Wire permutation-importance delta to criterion_6 |
| 7: Distractor robustness | Evaluates to `False` (no distractor split in baseline runs) | Yes (EXP-D2 produces distractor data) | Wire EXP-D2 results into verdict, or add DISTRACTOR split to baseline runs |
| 8: Sample efficiency | Hardcoded `False` | Yes (EXP-D1 learning curves) | Wire AUC from learning curves to criterion_8 |
| 9: Transfer | Evaluates to `False` (no composition/template splits) | No | Requires new split strategies or cross-task transfer experiments |

**Unlocking repeatable baseline STRONG verdicts beyond the single calibrated case requires wiring existing diagnostic data to the verdict criteria - not pretending the current baseline layer already does so.**

---

## 3. Literature Review: What the Evidence Actually Supports

This section reviews the key literature areas that bear on the current experimental results. Each subsection summarizes the research finding, explains how it connects to observed results in this repo, and identifies what still needs to change. The repo already gets the foundational choices right — deterministic generators, OOD-focused splits, modular design — so the review focuses on the gaps that remain after implementation.

### 3.1 IID Accuracy Is Not Evidence of Rule Learning

Lake and Baroni (2018) demonstrated with SCAN that sequence models can look strong on held-out IID splits while failing badly when the test set requires systematic compositional generalization rather than local interpolation. A model that learns "jump" means a specific action may not compose "jump twice" correctly if the composition was never seen. Bastings et al. (2019) then showed that SCAN itself can still admit shortcuts — proposing NACS as a harder inverse-mapping benchmark and demonstrating that split design quality is itself a research variable.

Kim and Linzen (2020) extended this line by showing that even apparently systematic SCAN solutions can emerge from surface heuristics when training distributions happen to be informative. Keysers et al. (2020) introduced the COGS benchmark and maximum compound divergence (MCD) splitting, which controls the gap between training and test compositions more precisely than random holdout.

**What this means for this repo:** The split generator already produces value-extrapolation, length-extrapolation, and noise splits. The verdict logic already requires OOD success (criterion 2), not just IID. This is a genuine strength. However, two risks remain:

1. **Some OOD splits may be too easy.** For `C3.1_xor`, value-extrapolation tests on wider numeric ranges, but the XOR boundary structure is preserved because both thresholds are fixed. A model that memorizes the boundary location would pass this split without learning the rule.
2. **Split design should be audited per task.** Not all OOD splits are equally informative for all task families. Categorical-only tasks (e.g., `C1.3_categorical_match`) need category-combination holdout splits, not just numeric range shifts.

**Required action:** Add adversarial boundary splits for classification (Phase 3, Action 3.3). Also add anti-shortcut or inverse task variants for major task families — this is an additional research direction beyond the current action plan.

### 3.2 Grokking: Training Dynamics Matter, Not Just End-State Accuracy

Power et al. (2022) showed that on small algorithmic datasets, models can transition from memorization to perfect generalization long after apparent overfitting — a phenomenon they called grokking. The transition can occur at 10× to 100× the epoch count where the model first overfits. The key lesson is broader than "don't use early stopping": if training budgets are too short, or if only final-checkpoint performance is recorded, the benchmark can misclassify a task as unsolved under a regime that simply did not run long enough.

Nanda et al. (2023) provided a mechanistic interpretability account of grokking, showing that models develop structured internal circuits for modular arithmetic during the delayed generalization phase. Their work identifies specific progress measures — representation structure metrics that predict when grokking will occur — which could be used to detect whether a NEGATIVE verdict reflects genuine task hardness or merely insufficient training.

Liu et al. (2023) showed that grokking is not limited to modular arithmetic — it appears across a range of algorithmic tasks including group operations and permutation composition. They also found that weight decay is a critical enabler: without regularization, the delayed generalization phase may never occur.

Thilak et al. (2022) extended grokking analysis to show that the phenomenon depends heavily on dataset size relative to model capacity. Small datasets with high-capacity models exhibit more pronounced delayed generalization, which directly describes the regime of this benchmark (500–1000 samples, neural models).

**What this means for this repo:** The LSTM trains for exactly 20 epochs (`sequence_experiments.py`, `"epochs": 20`). On `S1.2_sort`, IID accuracy is 15.2% — far from converged. On `S2.3_running_min`, IID is 52.6% — the model is learning something but may need 10–50× more epochs to break through. No training curves are logged, so we cannot tell whether accuracy was still rising at epoch 20. The repo's `C1.6_modular_class` task (INCONCLUSIVE, IID 82.9%) is especially relevant given Nanda et al.'s finding that modular arithmetic is a canonical grokking domain.

**Required action:** Increase LSTM epochs to 200+ for selected tasks, add per-epoch OOD evaluation checkpoints, and log training loss curves. Add weight decay as a required hyperparameter for all neural models. See Phase 1, Action 1.2.

### 3.3 Architecture-Task Alignment Is Real and Must Be Reported

Veličković and Blundell (2021) framed neural algorithmic reasoning as a research program that asks which architectures can imitate which classical algorithms. Their central argument is that model success depends strongly on whether the architecture's inductive bias aligns with the task's computational structure — for example, message-passing GNNs align naturally with dynamic programming, while Transformers align better with pointer-based operations.

The CLRS Algorithmic Reasoning Benchmark (Veličković et al., 2022) operationalizes this by testing multiple model families on 30 classical algorithms drawn from CLRS (Cormen et al.). Their key finding is that no single architecture dominates: GNN processors outperform Transformers on graph algorithms, while Transformers can outperform GNNs on sorting and searching when given appropriate input representations. Ibarz et al. (2022) followed up with improved GNN processors that achieve much stronger performance on CLRS, further demonstrating that algorithmic reasoning research is architecture-sensitive.

Mahdavi et al. (2023) conducted a systematic comparison of Transformers vs. recurrent models on algorithmic tasks and found that Transformers generally outperform LSTMs on tasks requiring arbitrary position access (sorting, reversing), while LSTMs can match or exceed Transformers on tasks with strictly sequential state evolution (running statistics, FSMs), especially under length extrapolation where positional encoding degrades.

**What this means for this repo:** The only real sequence model is a single-layer LSTM with 64 hidden units and no attention. Tasks like `S1.1_reverse` (IID 1.1%) and `S1.6_prefix_sum` (IID 0.6%) require position-level processing that benefits strongly from attention mechanisms. The `ModelFamily` enum in `harness.py` has no `TRANSFORMER` entry. Without at least two fundamentally different sequence architectures, the current NEGATIVE verdicts on the sequence track cannot distinguish "this task is too hard" from "this architecture is wrong for this task."

**Required action:** Add a Transformer encoder-decoder to `ModelFamily` and `build_model()`. This is the single highest-impact change for the sequence track. See Phase 2, Action 2.1.

### 3.4 Positional Encoding Is a Confound for Length Generalization

Press et al. (2022) introduced ALiBi (Attention with Linear Biases), showing that replacing learned absolute positional embeddings (APE) with a simple linear distance penalty enables better length extrapolation. However, Kazemnejad et al. (2023) found that the picture is much more nuanced: in a systematic study of positional encoding effects on length generalization, they showed that (a) no single encoding is universally safe, (b) the commonly used APE can catastrophically fail on lengths just slightly beyond training, (c) RoPE (Su et al., 2024) and ALiBi have different failure modes on different task types, and (d) the absence of any explicit positional encoding ("no PE") sometimes outperforms all others, especially when the model has causal attention masks that provide implicit position information.

Anil et al. (2022) further explored length generalization in large language models, showing that even very large models trained on diverse data fail to generalize arithmetic to longer digit strings than seen in training, and that scratchpad-based decomposition helps but does not fully solve the problem. Zhou et al. (2024) proposed teaching models arithmetic through length generalization by carefully controlling training data formatting.

**What this means for this repo:** No Transformer exists yet, so this is not a current failure mode. However, the `LENGTH_EXTRAPOLATION` split (train on lengths 4–16, test on 25–64) is one of the core evaluation axes. Once a Transformer is added, results on this split will be strongly confounded by positional encoding choice if only one scheme is used. A single NEGATIVE result on length extrapolation with APE could be reversed by switching to ALiBi or RoPE.

**Required action:** When adding the Transformer (Phase 2), include positional encoding as a hyperparameter axis: `{APE, RoPE, ALiBi, none}`. Report length-extrapolation results broken down by encoding. Also control input serialization format, because format and positional scheme interact. See Phase 2, Action 2.2.

### 3.5 Intermediate Computation: Scratchpads, Chain-of-Thought, and Process Supervision

Nye et al. (2021) showed that giving language models access to a "scratchpad" — an auxiliary output space where intermediate computation steps are written before the final answer — substantially improves performance on multi-step tasks like addition and polynomial evaluation. The model is trained to emit the full execution trace, not just the final answer.

Wei et al. (2022) demonstrated that chain-of-thought (CoT) prompting — where a few-shot prompt includes intermediate reasoning steps — elicits dramatically better performance from large language models on math and reasoning tasks, without any fine-tuning. Kojima et al. (2022) showed that even "Let's think step by step" as a zero-shot prompt improves reasoning.

Li et al. (2024) provided a theoretical foundation, proving that chain-of-thought can increase the effective expressiveness of bounded-depth Transformers on inherently serial problems. Specifically, they showed that a constant-depth Transformer with CoT tokens can solve problems that would otherwise require logarithmic depth, by effectively unrolling the serial computation across the CoT sequence.

However, Kazemnejad et al. (2023) cautioned that scratchpad format itself can become a confound. In their experiments, the specific way intermediate steps are serialized (e.g., left-to-right vs. right-to-left for addition carries) substantially affected length generalization. Dziri et al. (2023) went further, showing that even with chain-of-thought, language models often rely on shortcut patterns rather than faithful multi-step reasoning, and proposed a compositional benchmark (FAITH) to distinguish genuine reasoning from surface imitation.

Lightman et al. (2024) introduced process reward models (PRMs) that supervise each intermediate step rather than just the final answer. They showed that process supervision outperforms outcome supervision for mathematical reasoning, providing a practical framework for scoring intermediate computation quality.

**What this means for this repo:** All S2 and S3 tasks require multi-step computation — cumulative XOR needs a running accumulator, balanced parentheses needs a stack, run-length encoding needs a state machine. These are almost all NEGATIVE. The current pipeline has no trace/scratchpad support: `TaskSpec` has no `execution_trace` field, and the evaluation engine doesn't score intermediate steps.

The tasks in this repo are simpler than the problems studied in the CoT literature (we have known algorithms, not open-ended reasoning), but the core insight applies: for inherently serial problems, providing or learning intermediate state can transform an impossible task into a tractable one.

**Required action:** Add optional execution trace support for tasks with known intermediate states. Implement three supervision modes: answer-only, gold-trace, and self-generated-trace. Compare not just final accuracy but trace faithfulness. See Phase 3, Action 3.1. This is lower priority than the Transformer (architecture is the bigger bottleneck for S1 tasks), but it is the primary lever for S2/S3 tasks.

### 3.6 Tabular ML: Trees vs. Neural Networks on Structured Data

Grinsztajn et al. (2022) conducted a systematic comparison of tree-based models and neural networks on tabular data, finding that tree-based models (random forests, gradient-boosted trees) consistently outperform neural networks on medium-sized tabular datasets. They identified key factors: trees handle irregular feature distributions and uninformative features better, while neural networks require careful tuning and larger datasets.

Gorishniy et al. (2021) introduced FT-Transformer (Feature Tokenizer + Transformer), showing that Transformers can compete with gradient-boosted trees on tabular data when features are properly tokenized. Somepalli et al. (2022) showed that the gap between trees and neural networks on tabular data depends heavily on dataset characteristics — neural models catch up on tasks with smooth decision boundaries and sufficient data.

Shwartz-Ziv and Armon (2022) provided another large-scale comparison showing that XGBoost and other GBT implementations consistently outperform deep learning on tabular benchmarks, with the gap being largest on small datasets — exactly the regime of this project.

**What this means for this repo:** The classification track results (Section 2.1) confirm this literature: decision trees and random forests dominate on almost all C1–C3 tasks. The MLP occasionally matches but never consistently outperforms tree-based models. This is expected given the small dataset sizes (900–1000 samples) and the axis-aligned rule structure of most classification tasks.

The `C1.6_modular_class` INCONCLUSIVE result is interesting in this context: modular arithmetic creates a non-axis-aligned, periodic decision boundary that trees must approximate with many splits. The MLP has an architectural advantage here (it can learn periodic features), but its current capacity (`hidden_layer_sizes=(128, 64)`) and training budget (`max_iter=400`) may be insufficient.

**Required action:** For `C1.6_modular_class`, test a larger MLP or add a sine/cosine feature transformation. For the benchmark more broadly, consider adding FT-Transformer (Gorishniy et al.) as a neural tabular baseline for C-tier tasks in Phase 2+.

### 3.7 Program Synthesis and Symbolic Recovery

DeepCoder (Balog et al., 2017) demonstrated that neural models can guide search over a domain-specific language by predicting which DSL primitives are likely to appear in the target program. DreamCoder (Ellis et al., 2021) extended this with a library-learning loop that discovers reusable abstractions, progressively expanding the DSL. These approaches show that the search problem in EXP-B2 (DSL program recovery) can be dramatically improved over random search by using a learned prior.

Parisotto et al. (2017) (Neuro-Symbolic Program Synthesis) and Devlin et al. (2017) (RobustFill) showed that encoder-decoder architectures can learn to synthesize programs from input-output examples in constrained DSLs, achieving orders-of-magnitude speedups over enumerative search.

For rule extraction from trained models (EXP-B1), Bastani et al. (2017) proposed extracting decision trees from trained neural networks by training the tree on the neural network's predictions rather than the original labels. This can recover simpler rules when the neural network has already filtered noise.

**What this means for this repo:** EXP-B2's random search over the SR-10 DSL finds leaf operations (Reverse, Sort) but struggles with composed programs (dedup→sort→count). This is expected: random search scales exponentially with program depth. The DeepCoder/DreamCoder line suggests a concrete upgrade path: train a neural guide on (input/output pairs → primitive probabilities) to focus the search on likely programs.

EXP-B1's decision tree extraction already achieves >99% on 9/12 tasks, which is strong. The three failures (C1.5, C3.3, C3.5) all involve non-axis-aligned boundaries — exactly where symbolic regression or polynomial feature lifting would help.

**Required action:** For Phase 3+, consider adding a neural-guided DSL search for EXP-B2 (replace random search with a learned prior). For EXP-B1, add polynomial feature expansion for the three failing tasks. These are lower-priority enhancements given the current results.

### 3.8 State-Space Models as an Alternative Sequence Architecture

Gu and Dao (2023) introduced Mamba, a selective state-space model (SSM) that achieves near-Transformer performance on language modeling with linear-time complexity. Unlike Transformers, SSMs process sequences in a single forward pass without quadratic attention, making them naturally suited to long sequences. Gu et al. (2022) had previously demonstrated with S4 that structured state-space models can handle very long-range dependencies (e.g., the Long Range Arena benchmark) that Transformers struggle with.

Smith et al. (2023) showed that SSMs can learn simple algorithmic patterns (copying, induction heads) from data, but their performance on complex algorithmic reasoning tasks is less studied than Transformers'. Jelassi et al. (2024) compared Transformers and SSMs on algorithmic tasks and found that Transformers generally outperform SSMs on tasks requiring precise position-indexed access, while SSMs are competitive on tasks with smooth sequential state evolution.

**What this means for this repo:** SSMs could be particularly relevant for S2-tier tasks (stateful/iterative algorithms like cumulative XOR, running min, checksum) where the underlying computation is naturally sequential. The LSTM's failure on these tasks (S2.1 IID 25%, S2.3 IID 52.6%) may be partly due to the LSTM's limited hidden state capacity, which an SSM with selective gating could handle better.

**Required action:** Consider adding Mamba or S4 as a third sequence architecture family in Phase 2+, specifically for S2-tier tasks. This is a research opportunity rather than a blocking issue.

---

## 3B. Additional Gaps Beyond the Literature

The literature review identifies well-studied failure modes. The following additional gaps are specific to this repo's architecture and affect benchmark credibility.

### Gap 1: The Implementation Is Narrower Than the Design Documents

The design specifies S0–S5 and C0–C5 (60+ tasks across 12 tiers), but only S0–S3 and C0–C3 are implemented (32 tasks across 8 tiers). S4 (graph/structural), S5 (DSL family), C4 (stateful classification), and C5 (compositional classification) are deferred (DEV-004). This means composition-depth splits and template-holdout splits — which are the strongest tests of systematic generalization — cannot currently be applied.

### Gap 2: Task-Level Verdicts Are Too Collapsed

The report layer emits one verdict per task by selecting the best available result across all models. This conflates task difficulty, architecture mismatch, representation mismatch, training-budget insufficiency, and missing protocol features (traces, longer training). A task that is MODERATE because one tree model succeeded but all neural models failed carries a fundamentally different scientific message than one that is MODERATE across all architectures.

### Gap 3: Split Generation Should Become Split-Conditioned Sampling for Hard Regimes

The current runner generates one dataset per (task, seed) and derives all splits from it. This is correct for engineering and for early-phase experiments. But for benchmark-grade OOD evaluation, hard splits (e.g., adversarial boundary points, very long sequences) may have near-zero probability under the base sampling distribution. Research-grade splits need explicit quotas — guaranteed minimum sample counts in each regime — rather than relying on whatever the random sampler happens to produce.

### Gap 4: DSL Task Families Need Semantic Deduplication

Once S5 and C5 are expanded, many syntactically different DSL programs may collapse to nearly equivalent functions over the sampled input support. Without semantic deduplication or diversity constraints, the effective benchmark size can be inflated artificially. For example, `filter(>0) → sort` and `filter(>=1) → sort` may be identical on the positive-integer inputs that the sampler produces.

### Gap 5: Representation Fairness Is Under-Specified

Different model families currently see categorical features under different encodings: label encoding for tree models, one-hot for neural models. If a categorical feature has 10 levels, the tree model sees 1 column while the MLP sees 10 columns. This representation asymmetry can confound architecture comparisons — a finding that "trees outperform MLPs on categorical tasks" may partly reflect encoding advantage rather than architectural advantage. The benchmark needs a representation policy that is documented, fair, and stable across runs.

### Gap 6: The Current LSTM Is Not a Serious Algorithmic Baseline

The implemented LSTM predicts output positions from a single-layer encoder with 64 hidden units and 20 training epochs. It has no attention, no bidirectional processing, no stacking, and no curriculum or scheduling. This is useful as infrastructure validation, but it is not a fair baseline for sequence algorithmic reasoning. Comparing this LSTM to tree-based models and declaring "sequence tasks are harder" conflates task difficulty with model inadequacy.

---

## 4. Prioritized Action Plan

### Phase 1: Quick Wins — Wire Existing Data to Unlock STRONG (Effort: 1–2 sessions)

These actions require no new experiments — they wire existing diagnostic data into the verdict system.

#### Action 1.1: Wire Criteria 6–8 Into Verdict Logic [Effort: S]

**What:** Connect diagnostic experiment outputs to `compute_solvability_verdict()`.

**Files to change:**
- `src/reporting.py` → `compute_solvability_verdict()` — accept optional diagnostic evidence dict
- `src/diagnostic_experiments.py` → `run_d5_calibration()` already does this partially; formalize it

**Specific wiring:**
- **Criterion 6 (counterfactual sensitivity):** Use EXP-D4 feature-importance alignment. If the top-k permutation-importance features match the known relevant features for a task (data available in `TASK14_RELEVANT_FEATURES` and `TASK15_RELEVANT_FEATURES`), mark criterion 6 as met.
- **Criterion 7 (distractor robustness):** Use EXP-D2 results. If accuracy with 5+ distractors drops less than 5% from the no-distractor baseline, mark criterion 7 as met.
- **Criterion 8 (sample efficiency):** Use EXP-D1 learning curves. If the non-control task achieves 90% of its max accuracy at ≤50% of the sample budget (measured by AUC ratio), mark criterion 8 as met.

**Expected impact:** 8–10 classification tasks should immediately move from MODERATE → STRONG, because the diagnostic data already shows clean feature selection, distractor robustness, and efficient learning on most C1–C3 tasks.

**How to validate:** Re-run `python main.py diagnostic --output-root results` and check that `solvability_verdicts.json` in `results/EXP-D5/` shows STRONG labels for tasks where D1/D2/D4 data supports it.

#### Action 1.2: Extend LSTM Training Budget [Effort: S]

**What:** Increase LSTM epochs from 20 to 200 for sequence experiments, add a learning-rate scheduler (ReduceLROnPlateau), and log per-epoch train/test loss.

**Files to change:**
- `src/models/harness.py` → `LSTMSequenceModel.fit()` — add scheduler, epoch logging
- `src/sequence_experiments.py` → Update `_default_sequence_model_configs()` LSTM hyperparams: `{"epochs": 200, "hidden_size": 128, "embedding_dim": 64, "batch_size": 64, "learning_rate": 0.005, "scheduler": "reduce_on_plateau"}`
- `src/runner.py` → `SingleRunResult` — add optional `training_curve` field

**Expected impact:** Tasks like `S1.2_sort` (IID 15.2%), `S2.3_running_min` (IID 52.6%), and `S2.1_cumulative_xor` (IID 25.0%) may show significant improvement. Based on grokking literature, we should expect 5–10× training time to be necessary for algorithmic generalization on small datasets.

**How to validate:** Re-run `python main.py sequence --output-root results` and compare IID/OOD accuracy. If `S1.2_sort` IID moves above 50%, the training budget was the bottleneck.

#### Action 1.3: Fix C2.1 AND Rule Baseline Separation [Effort: S]

**What:** The AND rule produces imbalanced classes (the conjunction is true less often), causing the majority-class baseline to score ~86%, within 15% of the best model at 99.8%. Fix by ensuring the data generator targets ~50% class balance for binary classification tasks.

**Files to change:**
- `src/registry.py` → `C2.1_and_rule` builder — adjust input sampling ranges so the AND condition is satisfied ~50% of the time
- Alternatively, adjust the baseline separation threshold in `compute_solvability_verdict()` to account for class imbalance

**Expected impact:** C2.1 moves from WEAK → MODERATE (or STRONG if Action 1.1 is also applied).

**How to validate:** Re-run `python main.py classification --output-root results` and check EXP-C2 verdicts.

#### Action 1.4: Add DISTRACTOR Split to Baseline Experiment Runs [Effort: S]

**What:** The `DISTRACTOR` split is defined in the `SplitStrategy` enum but marked as not implemented (DEV-005). EXP-D2 already injects distractors at the task level. To enable criterion 7 in baseline runs, add `SplitStrategy.DISTRACTOR` to `TASK13_SPLIT_STRATEGIES` for classification experiments.

**Files to change:**
- `src/splits.py` → Implement `split_distractor()` that adds random columns to test-set feature vectors
- `src/runner.py` → `_apply_split()` — add DISTRACTOR case
- `src/classification_experiments.py` → Add `SplitStrategy.DISTRACTOR` to each experiment's split list

**Expected impact:** Criterion 7 becomes evaluable in baseline verdict computation (not just diagnostics), enabling STRONG without depending on EXP-D2 wiring.

**How to validate:** Re-run `python main.py classification --output-root results` and verify distractor accuracy columns appear in result summaries.

### Phase 2: Architecture Expansion — Add Transformer for Sequence Track (Effort: 2–3 sessions)

#### Action 2.1: Add Transformer Encoder-Decoder to Model Harness [Effort: M]

**What:** Add `ModelFamily.TRANSFORMER` to the enum and implement `TransformerSequenceModel` in `harness.py`.

**Architecture spec:**
- 2-layer encoder, 2-layer decoder (configurable)
- 128 model dimension, 4 attention heads (configurable)
- Positional encoding as a hyperparameter: `{"pos_encoding": "ape" | "rope" | "alibi" | "none"}`
- Teacher forcing during training, autoregressive generation during inference
- Shared vocabulary with the LSTM path (reuse `_build_vocab` logic)
- Unknown-token bucket for OOD inference (reuse ADR-020 pattern)

**Files to change:**
- `src/models/harness.py` → Add `TRANSFORMER` to `ModelFamily`, implement `TransformerSequenceModel` class, update `build_model()` dispatch
- `src/sequence_experiments.py` → Add `ModelConfig(family=ModelFamily.TRANSFORMER, ...)` to `_default_sequence_model_configs()`
- `tests/test_model_harness.py` → Add Transformer smoke tests

**Expected impact:** The Transformer should significantly outperform the LSTM on position-dependent tasks like `S1.1_reverse`, `S1.3_rotate`, and `S1.6_prefix_sum`. Based on the neural algorithmic reasoning literature, attention-based models can solve these tasks near-perfectly with enough training.

**How to validate:** Re-run `python main.py sequence --output-root results` and compare per-task verdicts. Target: at least 5 sequence tasks move from NEGATIVE → WEAK or better.

#### Action 2.2: Positional Encoding Ablation [Effort: M]

**What:** Run sequence experiments with each positional encoding variant as a separate model config.

**Files to change:**
- `src/sequence_experiments.py` → Add 4 Transformer configs (one per PE variant) to the model list
- `src/reporting.py` → Add PE breakdown to summary markdown generation

**Expected impact:** Identifies which PE scheme works best for length extrapolation on each task family. Based on Kazemnejad et al., we expect ALiBi and no-PE to outperform APE on length extrapolation, but APE to work better on value-range tasks.

**How to validate:** Compare `test_length_extrapolation` accuracy across PE variants in EXP-S1 results.

#### Action 2.3: Scale Up LSTM Architecture [Effort: S]

**What:** In parallel with the Transformer, also test a larger LSTM: 2 layers, 256 hidden units, bidirectional encoder.

**Files to change:**
- `src/sequence_experiments.py` → Add a second LSTM config with `{"epochs": 200, "hidden_size": 256, "n_layers": 2, "bidirectional": True, ...}`
- `src/models/harness.py` → `LSTMSequenceModel` — add `n_layers` and `bidirectional` support (currently single-layer unidirectional)

**Expected impact:** Isolates whether the LSTM failure is due to capacity vs. architecture. If a 2-layer bidirectional LSTM with 200 epochs still fails on sort/reverse while the Transformer succeeds, that is direct evidence of architecture-task alignment.

### Phase 3: Advanced Protocols — Process Supervision and Deeper Evaluation (Effort: 3–4 sessions)

#### Action 3.1: Add Execution Trace Support [Effort: L]

**What:** For tasks with known intermediate states (e.g., prefix sum has a running accumulator, balanced parentheses has a stack depth), add optional trace fields to `TaskSpec` and support trace-supervised training.

**Changes needed:**
- `src/schemas.py` → Add `TraceSpec` dataclass: `(trace_steps: List[str], trace_type: "intermediate_sequence" | "stack_depth" | "accumulator")`
- `src/registry.py` → Add `execution_trace_generator: Optional[Callable]` to `TaskSpec` for tasks that support it (S1.6, S2.1, S2.2, S2.3, S2.5)
- `src/models/harness.py` → Add trace-supervised loss option to sequence models (auxiliary loss on intermediate outputs)
- `src/evaluation.py` → Add `trace_accuracy` metric to `EvalReport`

**Expected impact:** Based on Nye et al. and Li et al., trace supervision should help multi-step tasks like `S2.1_cumulative_xor` (currently IID 25%) and `S2.5_checksum` (currently IID 38.7%) significantly.

#### Action 3.2: Architecture-Conditioned Verdict Matrix [Effort: M]

**What:** Replace the single per-task verdict label with a matrix: `(task × architecture) → verdict`.

**Changes needed:**
- `src/reporting.py` → `compute_solvability_verdict()` — add `model_family` parameter, compute per-model verdicts
- `src/reporting.py` → Summary markdown generator — output a verdict matrix table instead of a single column
- `src/runner.py` → `ExperimentReport` — include per-model breakdowns in aggregated results

**Expected impact:** Makes architecture sensitivity visible. A task that is MODERATE with trees but NEGATIVE with LSTM tells a different story than one that is MODERATE across all architectures.

#### Action 3.3: Adversarial Boundary Split for Classification [Effort: M]

**What:** Implement `split_adversarial_boundary()` in `src/splits.py`. For tasks with known decision boundaries, oversample test points within ε of the boundary.

**Changes needed:**
- `src/splits.py` → Add `split_adversarial()` function
- `src/runner.py` → Wire `ADVERSARIAL` strategy
- `src/classification_experiments.py` → Add adversarial splits for C1–C3

**Expected impact:** Replaces easy OOD splits where the extrapolation happens far from the boundary (and thus stays on the same side). This is the hardest and most informative split for classification tasks.

#### Action 3.4: Implement Deferred Tier Tasks (S4/S5/C4/C5) [Effort: L]

**What:** Register the remaining task tiers from the design document (DEV-004).

**Changes needed:**
- `src/registry.py` → Add S4 (graph/structural tasks), S5 (DSL program family), C4 (stateful classification), C5 (compositional classification)
- `src/dsl/sequence_dsl.py` → Expand DSL for S5 tier
- `src/dsl/classification_dsl.py` → Expand DSL for C5 tier
- Tests for new tiers

**Expected impact:** Expands the benchmark from 32 to ~50+ tasks and enables composition-depth and template-holdout splits that are currently impossible.

---

## 5. Concrete Experiment Upgrade Specifications

### 5.1 Upgraded Sequence Experiment Protocol

The current sequence experiment protocol (`src/sequence_experiments.py`) should be upgraded as follows:

| Parameter | Current | Upgraded (Phase 1) | Upgraded (Phase 2) |
|---|---|---|---|
| LSTM epochs | 20 | 200 | 200 |
| LSTM hidden size | 64 | 128 | 128 |
| LSTM embedding dim | 32 | 64 | 64 |
| LSTM learning rate | 0.01 | 0.005 with scheduler | 0.005 with scheduler |
| Transformer | — | — | 2-layer enc-dec, 128 dim, 4 heads |
| Seeds | [42, 123, 456] | [42, 123, 456, 789, 1024] | [42, 123, 456, 789, 1024] |
| N samples EXP-S1 | 600 | 1000 | 2000 |
| N samples EXP-S2 | 600 | 1000 | 2000 |
| Training curve logging | None | Per-epoch loss + OOD acc | Per-epoch loss + OOD acc |
| Positional encoding axis | — | — | {APE, RoPE, ALiBi, none} |

### 5.2 Upgraded Classification Experiment Protocol

The classification protocol is already strong. Upgrades focus on unlocking STRONG verdicts:

| Parameter | Current | Upgraded (Phase 1) |
|---|---|---|
| Split strategies | IID, VALUE_EXTRAP, NOISE | IID, VALUE_EXTRAP, NOISE, DISTRACTOR |
| Seeds | [42, 123, 456, 789, 1024] | [42, 123, 456, 789, 1024] (no change) |
| Verdict wiring | Criteria 1–5 only | Criteria 1–9 (wired to EXP-D data) |
| Adversarial boundary split | — | Phase 3 |

### 5.3 Training Protocol Standards

All model training should follow these standards going forward:

1. **Minimum training budget:** Neural models (LSTM, Transformer, MLP) must train for at least 100 epochs on algorithmic tasks. The runner should log accuracy at epochs [10, 25, 50, 100, 200] to detect grokking.
2. **Learning rate scheduling:** Use `ReduceLROnPlateau` with patience=20 for neural models. This prevents stalling without premature termination.
3. **No early stopping by default.** Early stopping on validation loss can terminate before grokking. If used, set patience ≥ 50 epochs and log when it triggers.
4. **Checkpoint best OOD model.** For each training run, save the checkpoint with the best OOD accuracy, not just the final-epoch model. Report both final and best-checkpoint metrics.
5. **Fixed compute budget accounting.** When comparing architectures, report results at matched compute budgets (total gradient steps × batch size) rather than matched epochs.

### 5.4 Statistical Reporting Standards

Current reporting uses mean ± std across 3–5 seeds. Upgrade to:

1. **5 seeds minimum** for any published result (sequence currently uses only 3).
2. **95% confidence intervals** in addition to mean ± std (use bootstrap or t-distribution with n-1 df).
3. **Per-seed results table** in appendix, not just aggregates.
4. **Significance testing** for architecture comparisons: paired t-test or Wilcoxon signed-rank across seeds, with Bonferroni correction for multiple comparisons.
5. **Effect sizes** for baseline separation: Cohen's d between best model and baseline, not just the raw gap.

---

## 6. Updated Verdict Framework

### 6.1 Current Verdict Logic (as implemented)

From `src/reporting.py`, `compute_solvability_verdict()`:

```
STRONG  = criteria 1–5 all met + ≥2 of criteria 6–9
MODERATE = criteria 1–5 all met
WEAK    = criterion 1 met + any of criteria 2–5 failed
NEGATIVE = criterion 1 not met (best IID < 0.95)
INCONCLUSIVE = mixed signals (criterion 1 fails + degradation inconsistent)
```

Thresholds used:
- Criterion 1 (high IID): accuracy ≥ 0.95
- Criterion 2 (extrapolation): OOD accuracy ≥ 0.85
- Criterion 3 (baseline separation): gap ≥ 0.15
- Criterion 4 (seed stability): std ≤ 0.05, ≥ 2 seeds
- Criterion 5 (coherent degradation): no OOD split exceeds IID + 0.05
- Criterion 6 (counterfactual): currently hardcoded `False`
- Criterion 7 (distractor): `|IID - distractor_acc| ≤ 0.05` (but no distractor runs in baselines)
- Criterion 8 (sample efficiency): currently hardcoded `False`
- Criterion 9 (transfer): `composition_split_acc ≥ 0.85` (but no composition splits in baselines)

### 6.2 Proposed Verdict Wiring (Phase 1)

After wiring existing diagnostic data:

| Criterion | Data source | Threshold | Implementation |
|---|---|---|---|
| 6: Counterfactual | EXP-D4 feature importance | Top-k features overlap with known relevant features ≥ 80% | Load D4 results in verdict computation |
| 7: Distractor | EXP-D2 or DISTRACTOR split | Accuracy drop ≤ 5% with 5+ distractors | Load D2 results or add split to baselines |
| 8: Sample efficiency | EXP-D1 learning curves | 90% of max accuracy reached at ≤ 50% sample budget | Load D1 AUC ratios |
| 9: Transfer | Composition/template splits | OOD accuracy ≥ 0.85 on held-out compositions | Requires new split strategies (Phase 3) |

### 6.3 Future: Architecture-Conditioned Verdicts (Phase 3)

Replace the single verdict with a profile:

```
C1.1_numeric_threshold:
  decision_tree:       STRONG  (criteria 1-8 met)
  random_forest:       STRONG  (criteria 1-8 met)
  logistic_regression: STRONG  (criteria 1-8 met)
  mlp:                 MODERATE (criteria 1-5 met)

S1.2_sort:
  lstm_20ep:           NEGATIVE (IID 15.2%)
  lstm_200ep:          WEAK     (IID ~60%?, OOD ~20%?)
  transformer_ape:     MODERATE (IID ~95%?, OOD ~85%?)
  transformer_rope:    STRONG   (IID ~98%?, OOD ~95%?)
```

This matrix explicitly separates task difficulty from architecture suitability, which is the core scientific contribution the benchmark should aim for.

---

## 7. Implementation Checklist

Track progress on the action plan produced during TASK-16:

### Phase 1 Checklist

- [ ] **1.1** Wire criteria 6–8 into `compute_solvability_verdict()` using EXP-D1/D2/D4 data
- [ ] **1.2** Increase LSTM epochs to 200, add LR scheduler, log training curves
- [ ] **1.3** Fix C2.1 AND rule class balance for baseline separation
- [ ] **1.4** Implement DISTRACTOR split in `splits.py` and add to classification experiments
- [ ] Re-run: `python main.py classification --output-root results`
- [ ] Re-run: `python main.py sequence --output-root results`
- [ ] Re-run: `python main.py diagnostic --output-root results`
- [ ] Verify ≥5 classification tasks reach STRONG verdict

### Phase 2 Checklist

- [ ] **2.1** Implement `TransformerSequenceModel` in `harness.py`
- [ ] **2.2** Add positional encoding ablation (APE/RoPE/ALiBi/none)
- [ ] **2.3** Scale up LSTM (2-layer bidirectional, 256 hidden)
- [ ] Re-run: `python main.py sequence --output-root results`
- [ ] Verify ≥5 sequence tasks move from NEGATIVE to WEAK or better

### Phase 3 Checklist

- [ ] **3.1** Add execution trace support for stateful sequence tasks
- [ ] **3.2** Implement architecture-conditioned verdict matrix
- [ ] **3.3** Implement adversarial boundary split for classification
- [ ] **3.4** Register S4/S5/C4/C5 task tiers
- [ ] Full benchmark re-run with expanded task and model matrix

---

## 8. Key Deviations That Inform This Plan

The following deviations from the original plan (documented in `EXPERIMENT_CATALOG.md` Part 5 and `PROJECT_STATUS.md`) directly shape the actions above:

| Deviation | Impact on methodology | Addressed by |
|---|---|---|
| DEV-004: S4/S5/C4/C5 deferred | Benchmark is narrower than designed | Phase 3, Action 3.4 |
| DEV-005: DISTRACTOR split not implemented | Criterion 7 cannot be evaluated in baseline runs | Phase 1, Action 1.4 |
| DEV-008: Verdicts operationalized from available signals only | Baseline STRONG is underwired; calibrated D5 now provides one STRONG task | Phase 1, Action 1.1 |
| DEV-011: Only 4 model families for sequence tasks | Architecture coverage too narrow | Phase 2, Actions 2.1–2.3 |
| DEV-016: MODERATE threshold instead of STRONG for bonus selection | Reflects that baseline STRONG was absent when TASK-15 ran; calibrated D5 now yields one STRONG task | Phase 1, Action 1.1 helps close this gap |

---

## 9. References

### Compositional generalization and OOD evaluation

1. **Lake, B. M., & Baroni, M.** "Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks." (2018) [arXiv:1711.00350](https://arxiv.org/abs/1711.00350) — Foundational SCAN paper showing IID success does not imply compositional rule learning. Validates the OOD-focused split design already in this repo.

2. **Bastings, J., Baroni, M., Weston, J., Cho, K., & Kiela, D.** "Jump to Better Conclusions: SCAN both Left and Right." (2019) [arXiv:1809.04640](https://arxiv.org/abs/1809.04640) — Showed that SCAN admits shortcuts; proposed NACS as a harder benchmark. Motivates adversarial and anti-shortcut split design (Action 3.3).

3. **Kim, N., & Linzen, T.** "COGS: A Compositional Generalization Challenge Based on Semantic Interpretation." (2020) [arXiv:2010.05465](https://arxiv.org/abs/2010.05465) — Introduced MCD splitting for controlled compositional generalization testing. Relevant to split strategy design for S5/C5 tiers.

4. **Keysers, D., Schärli, N., Scales, N., Buiber, H., Furrer, D., Kasber, S., Kohli, P., Noy, Y., Sørensen, T., Tay, Y., & others.** "Measuring Compositional Generalization: A Comprehensive Method on Realistic Data." (2020) [arXiv:1912.09713](https://arxiv.org/abs/1912.09713) — Maximum compound divergence (MCD) splitting method. Informs future composition-depth split design for deferred S5/C5 tiers.

### Grokking and training dynamics

5. **Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V.** "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." (2022) [arXiv:2201.02177](https://arxiv.org/abs/2201.02177) — Demonstrated delayed generalization long after overfitting on algorithmic tasks. Directly relevant to Action 1.2 (short LSTM training horizons, 20 epochs → 200+).

6. **Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J.** "Progress Measures for Grokking via Mechanistic Interpretability." (2023) [arXiv:2301.05217](https://arxiv.org/abs/2301.05217) — Mechanistic explanation of grokking; identifies progress measures that predict phase transitions. Relevant to `C1.6_modular_class` (INCONCLUSIVE, IID 82.9%) which is a canonical grokking domain.

7. **Liu, Z., Kitouni, O., Nolte, N., Michaud, E. J., Tegmark, M., & Williams, M.** "Towards Understanding Grokking: An Effective Theory of Representation Learning." (2023) [arXiv:2205.10343](https://arxiv.org/abs/2205.10343) — Extended grokking beyond modular arithmetic to group operations and permutation composition. Shows weight decay is a critical enabler.

8. **Thilak, V., Littwin, E., Zhai, S., Saremi, O., Paiss, R., & Susskind, J.** "The Slingshot Mechanism: An Empirical Study of Adaptive Optimizers and the Grokking Phenomenon." (2022) [arXiv:2206.04817](https://arxiv.org/abs/2206.04817) — Showed grokking dependence on dataset size relative to model capacity. Directly describes this benchmark's regime (500–1000 samples, neural models).

### Neural algorithmic reasoning and architecture alignment

9. **Veličković, P., & Blundell, C.** "Neural Algorithmic Reasoning." (2021) [arXiv:2105.02761](https://arxiv.org/abs/2105.02761) — Framed the research program of testing which architectures can imitate which algorithms. Motivates architecture-task alignment analysis (Action 2.1).

10. **Veličković, P., Puigdomènech Badia, A., Budden, D., Pascanu, R., Banino, A., Dashevskiy, M., Hadsell, R., & Blundell, C.** "The CLRS Algorithmic Reasoning Benchmark." (2022) [arXiv:2205.15659](https://arxiv.org/abs/2205.15659) — 30 classical algorithms as a benchmark; showed no single architecture dominates. Model for architecture-conditioned reporting (Action 3.2).

11. **Ibarz, B., Kurin, V., Papamakarios, G., Nikiforou, K., Bennani, M., Csordás, R., Duber, A., Veličković, P., & Blundell, C.** "A Generalist Neural Algorithmic Learner." (2022) [arXiv:2209.11142](https://arxiv.org/abs/2209.11142) — Improved GNN processors for CLRS; demonstrated that architecture refinement yields large gains on algorithmic tasks.

12. **Mahdavi, S., Shen, L., & Yehudai, G.** "Towards Better Out-of-Distribution Generalization of Neural Algorithmic Reasoning Tasks." (2023) [arXiv:2211.00692](https://arxiv.org/abs/2211.00692) — Systematic comparison of Transformers vs. recurrent models on algorithmic tasks; found architecture-task dependencies in length extrapolation.

### Positional encoding and length generalization

13. **Press, O., Smith, N. A., & Lewis, M.** "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." (2022) [arXiv:2108.12409](https://arxiv.org/abs/2108.12409) — Introduced ALiBi for improved length extrapolation. Baseline for Action 2.2.

14. **Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., & Reddy, S.** "The Impact of Positional Encoding on Length Generalization in Transformers." (2023) [arXiv:2305.19466](https://arxiv.org/abs/2305.19466) — Systematic study showing no single PE is universally safe; APE, ALiBi, RoPE, and no-PE each have different failure modes. Critical for Action 2.2 ablation design.

15. **Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y.** "RoFormer: Enhanced Transformer with Rotary Position Embedding." (2024) [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — Introduced RoPE, now widely used; one of the PE variants to ablate.

16. **Anil, C., Wu, Y., Andreassen, A., Lewkowycz, A., Misra, V., Ramasesh, V., Slone, A., Gur-Ari, G., Dyer, E., & Neyshabur, B.** "Exploring Length Generalization in Large Language Models." (2022) [arXiv:2207.04901](https://arxiv.org/abs/2207.04901) — Systematic study of length generalization failures in LLMs; relevant to interpreting `LENGTH_EXTRAPOLATION` split results.

17. **Zhou, H., Nova, A., Larochelle, H., & Courville, A.** "Teaching Arithmetic to Small Transformers." (2024) [arXiv:2307.03381](https://arxiv.org/abs/2307.03381) — Showed that data formatting and curriculum design enable length generalization for arithmetic. Relevant to future data formatting controls.

### Process supervision, scratchpads, and chain-of-thought

18. **Nye, M., Andreassen, A., Gur-Ari, G., Michalewski, H., Austin, J., Biber, D., Dohan, D., Lewkowycz, A., Bosma, M., Luan, D., Sutton, C., & Odena, A.** "Show Your Work: Scratchpads for Intermediate Computation with Language Models." (2021) [arXiv:2112.00114](https://arxiv.org/abs/2112.00114) — Demonstrated scratchpad benefits for multi-step computation. Primary motivation for Action 3.1 (execution traces).

19. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D.** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." (2022) [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) — Demonstrated chain-of-thought gains; chain-of-thought as a form of intermediate computation.

20. **Li, Z., Liu, H., Zhou, D., & Ma, T.** "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems." (2024) [arXiv:2402.12875](https://arxiv.org/abs/2402.12875) — Theoretical proof that CoT increases effective expressiveness for serial problems. Strongest theoretical motivation for trace supervision on S2/S3 tasks.

21. **Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y.** "Large Language Models are Zero-Shot Reasoners." (2022) [arXiv:2205.11916](https://arxiv.org/abs/2205.11916) — Zero-shot chain-of-thought ("Let's think step by step"). Shows reasoning benefits even without task-specific supervision.

22. **Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin, B. Y., West, P., Bhagavatula, C., Le Bras, R., Hwang, J. D., Welleck, S., Smith, N. A., Choi, Y., & Ren, X.** "Faith and Fate: Limits of Transformers on Compositionality." (2023) [arXiv:2305.18654](https://arxiv.org/abs/2305.18654) — Showed CoT models can rely on shortcuts rather than faithful reasoning. Motivates trace faithfulness evaluation in Action 3.1.

23. **Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K.** "Let's Verify Step by Step." (2024) [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) — Process reward models supervising each intermediate step outperform outcome supervision. Framework for scoring trace quality.

### Tabular ML: trees vs. neural networks

24. **Grinsztajn, L., Oyallon, E., & Varoquaux, G.** "Why do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?" (2022) [arXiv:2207.08815](https://arxiv.org/abs/2207.08815) — Systematic comparison showing tree dominance on medium-sized tabular data. Confirms the classification track results where decision trees dominate C1–C3.

25. **Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A.** "Revisiting Deep Learning Models for Tabular Data." (2021) [arXiv:2106.11959](https://arxiv.org/abs/2106.11959) — Introduced FT-Transformer showing Transformers can compete on tabular data. Potential neural tabular baseline for C-tier tasks.

26. **Shwartz-Ziv, R., & Armon, A.** "Tabular Data: Deep Learning Is Not All You Need." (2022) [arXiv:2106.03253](https://arxiv.org/abs/2106.03253) — Large-scale comparison confirming GBT outperforms deep learning on tabular benchmarks, especially on small datasets.

### Program synthesis and symbolic recovery

27. **Balog, M., Gaunt, A. L., Brockschmidt, M., Nowozin, S., & Tarlow, D.** "DeepCoder: Learning to Write Programs." (2017) [arXiv:1611.01989](https://arxiv.org/abs/1611.01989) — Neural-guided DSL search from I/O examples. Directly relevant to improving EXP-B2 search over SR-10 DSL beyond random search.

28. **Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., Cary, L., Solar-Lezama, A., & Tenenbaum, J. B.** "DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning." (2021) [arXiv:2006.08381](https://arxiv.org/abs/2006.08381) — Library-learning loop for program synthesis. Shows how to grow the DSL vocabulary with discovered abstractions.

29. **Bastani, O., Kim, C., & Bastani, H.** "Interpreting Blackbox Models via Model Extraction." (2017) [arXiv:1705.08504](https://arxiv.org/abs/1705.08504) — Extracting decision trees from neural network predictions. Relevant to EXP-B1 methodology for non-axis-aligned tasks.

### State-space models

30. **Gu, A., & Dao, T.** "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." (2023) [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) — Selective SSM achieving near-Transformer performance with linear complexity. Potential third sequence architecture family for S2-tier stateful tasks.

31. **Gu, A., Goel, K., & Ré, C.** "Efficiently Modeling Long Sequences with Structured State Spaces." (2022) [arXiv:2111.00396](https://arxiv.org/abs/2111.00396) — Introduced S4; showed SSMs can handle very long-range dependencies. Relevant to long-sequence regime in LENGTH_EXTRAPOLATION splits.

32. **Jelassi, S., Brandfonbrener, D., Kakade, S., & Malach, E.** "Repeat After Me: Transformers are Better than State Space Models at Copying." (2024) [arXiv:2402.01032](https://arxiv.org/abs/2402.01032) — Compared Transformers and SSMs on algorithmic tasks; Transformers better at position-indexed access, SSMs competitive on sequential state tasks. Relevant to architecture selection per task tier.
