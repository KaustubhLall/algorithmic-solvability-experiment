# Methodology Systematic Review And Benchmark Reset Proposal

> Status: Proposal for review
> Date: 2026-03-26
> Scope: Compare the current repository plan and implementation against the attached methodology feedback and the algorithmic generalization literature, then propose a research-grade reset before benchmark-scale experiments.

## Executive Summary

The repository already gets two foundational choices right:

- It uses deterministic synthetic generators and reference algorithms, which removes label ambiguity.
- It treats out-of-distribution evaluation as central rather than optional, especially through the split generator.

The feedback is directionally correct on the biggest remaining risks, but several points need to be reframed against the actual repository state:

- The current codebase does not yet implement a Transformer, so positional encoding is not a current failure mode in the literal sense. It is, however, a must-control factor before any Transformer-based benchmark claims are made.
- The current harness does not use early stopping, but it still has the same practical blind spot as early stopping: fixed, short training horizons can miss delayed generalization and grokking-style phase transitions.
- Scratchpads should be added as an experimental axis, not treated as a universal requirement. Recent work supports intermediate computation for serial problems, but also shows that scratchpad format can itself become a confound.

Bottom line: the project is ready for pilot experiments, but not yet for strong claims about "task solvability" as a benchmark-level property. Before running TASK-12 and TASK-13 as headline experiments, the benchmark should add a methodology-reset phase covering long-horizon training protocols, process supervision, architecture-conditioned reporting, positional and serialization controls, and stronger statistical reporting discipline.

## What The Current Repository Already Gets Right

### 1. Deterministic task generation and verifiable labels

This is the right foundation for an algorithmic benchmark. The design already assumes that every task has a known reference algorithm and that labels can be re-verified at generation time. That aligns well with the small-algorithm datasets used in grokking and neural algorithmic reasoning work, where clean supervision is a prerequisite for studying learning dynamics rather than dataset noise.

### 2. OOD-focused split design

The current design and TASK-06 implementation are already oriented around systematic generalization rather than IID-only performance. That is a major strength and directly addresses the core lesson from SCAN-style failures: held-out examples from the same support do not establish rule learning.

### 3. Separation of generator, model, evaluator, and reporter

The repository is well-factored for research iteration. Because data generation, splitting, training, evaluation, and reporting are already modular, the methodology can be upgraded without rewriting the whole project.

## Where The Feedback Applies To The Current Repository

The table below distinguishes between the design documents and the code that actually exists today.

| Topic | Current repo state | Does the feedback apply? | Assessment |
|---|---|---|---|
| OOD over IID | Strong in design and already implemented | Yes, positively | This is already a core strength. |
| Deterministic DSLs | Strong in design and already implemented | Yes, positively | This is already a core strength. |
| Grokking / delayed generalization | No early stopping yet, but training is short-horizon and minimally instrumented | Yes | The exact failure mode differs, but the underlying risk remains. |
| Scratchpads / intermediate traces | Absent from schema, registry, harness, evaluation, and reporting | Yes | This is a real gap for multi-step algorithms. |
| Architecture sensitivity | Mentioned in design, but reporting still collapses to a single task verdict; implementation lacks several planned families | Yes | This is a real gap. |
| Positional encoding | No Transformer implementation yet | Not yet in code, yes in future design | This must be controlled before Transformer experiments start. |

## Literature Review: What The Evidence Actually Supports

### 1. IID accuracy is not evidence of rule learning

Lake and Baroni's SCAN paper showed that sequence models can look strong on some held-out splits while failing badly when the test set requires systematic compositional generalization rather than local interpolation. Bastings et al. then showed that SCAN itself can still admit shortcuts, proposing NACS as a harder inverse mapping benchmark. The implication is not just "use OOD splits"; it is also "audit whether your OOD split still contains a shortcut path."

Repository implication:

- Keep adversarial and extrapolation splits.
- Add inverse or anti-shortcut task variants for major task families.
- Treat split design itself as part of the benchmark, not just a preprocessing choice.

### 2. Grokking means training dynamics matter, not just end-state accuracy

Power et al. showed that on small algorithmic datasets, models can move from memorization to perfect generalization long after overfitting appears. The key lesson is broader than early stopping: if training budgets are too short, or if only final checkpoint performance is recorded, the benchmark can misclassify a task as unsolved under a regime that simply did not run long enough.

Repository implication:

- Add long-horizon runs for selected tasks.
- Track train, IID, and OOD curves over optimization steps.
- Report time-to-generalization and best-late OOD accuracy, not just end-of-run accuracy.
- Treat regularization as part of the protocol, not post hoc tuning.

### 3. Intermediate computation helps, but should be studied as a controlled factor

Nye et al. showed that scratchpads can help language models learn intermediate computation. Wei et al. showed strong empirical gains from chain-of-thought prompting. Li et al. gave a theoretical account showing that chain-of-thought can increase the effective expressiveness of low-depth Transformers on inherently serial problems. However, Kazemnejad et al. also found that scratchpads are not uniformly helpful and that their format can strongly affect length generalization outcomes.

Repository implication:

- Add process-supervised variants for tasks with obvious execution traces.
- Compare answer-only, gold-trace, and self-generated-trace settings.
- Evaluate both final-answer correctness and trace faithfulness.
- Do not hard-code "scratchpads on" as the only protocol.

### 4. Architecture-task alignment is real, and benchmark claims must reflect it

Neural algorithmic reasoning work argues that model success depends strongly on architectural bias. The CLRS benchmark makes the same point operational by comparing across families on a shared set of classical algorithms rather than making one global claim from one architecture.

Repository implication:

- Replace single-axis solvability language with architecture-conditioned solvability profiles.
- Match compute budgets and parameter scales across baselines where possible.
- Make architecture family a first-class reporting dimension, not just metadata.

### 5. Positional encoding is a confound, but the fix is not "just use ALiBi or RoPE"

Press et al. showed that ALiBi can improve input-length extrapolation. But later work by Kazemnejad et al. found that positional encoding choice substantially changes length generalization behavior and that common choices such as APE, ALiBi, and Rotary are not universally safe in downstream reasoning settings. The benchmark therefore should not freeze one positional method and call the issue solved.

Repository implication:

- Treat positional encoding as an ablation axis.
- At minimum compare APE, ALiBi, RoPE, and a no-explicit-PE condition where architecturally meaningful.
- Also control serialization format, because format and positional scheme interact.

## Current Repository Gaps Beyond The Attached Feedback

The feedback is strong, but it still misses several benchmark-level issues that matter if this is meant to become a paper-generating research program.

### 1. The implementation is narrower than the design documents suggest

The current harness supports classical tabular models and a targeted pooled LSTM path, but not the broader model matrix described in the design docs. That means the current repo is still an infrastructure prototype rather than a fair cross-architecture benchmark.

### 2. The current LSTM path is not a full algorithm-learning baseline

The implemented LSTM predicts output positions from a pooled encoder representation and fixed training horizon. That is useful for smoke tests, but it is not yet a serious baseline for multi-step algorithmic execution, trace emission, or deep length extrapolation.

### 3. Task-level verdicts are still too collapsed

The report layer currently emits one verdict per task by selecting the best available result. That is convenient for engineering summaries, but scientifically it conflates:

- task difficulty,
- architecture mismatch,
- representation mismatch,
- training-budget insufficiency,
- and missing protocol features such as traces.

### 4. Several evidence criteria in the design are still unmeasured

Counterfactual sensitivity, sample efficiency, transfer, and distractor robustness are part of the design logic, but not yet benchmark-wide measured quantities. That means the current verdict machinery is necessarily provisional.

### 5. Split generation should become split-conditioned sampling for the hardest regimes

The current runner generates a dataset and then derives splits from it. That is fine for early pipeline work, but research-grade OOD evaluation often needs guaranteed sample mass in rare hard regimes. For benchmark claims, the hard splits should be generated with explicit quotas and distribution manifests rather than left to chance from a shared pool.

### 6. DSL task families need semantic deduplication

Once S5 and C5 are expanded, many syntactically different programs may collapse to nearly equivalent functions over the sampled support. Without semantic deduplication or diversity constraints, effective benchmark size can be inflated artificially.

### 7. Representation fairness is under-specified for tabular categoricals

If different baselines see categoricals under materially different encodings, architecture comparisons become hard to interpret. The benchmark needs a representation policy that is fair, documented, and stable across runs.

### 8. Statistical discipline needs to move beyond mean-plus-std

For a benchmark intended to support papers, the reporting layer should add confidence intervals, explicit success thresholds, and benchmark-level significance tests or hierarchical modeling, especially once task families and seeds grow.

## Proposed Benchmark Reset Before TASK-12 And TASK-13

The key recommendation is to insert a methodology-reset phase before full benchmark experiments. The goal is not to stall implementation; it is to prevent the first large experimental pass from producing results that will later need to be retracted or heavily qualified.

### Workstream A: Benchmark Spec v2

Add benchmark metadata fields for:

- algorithm family,
- state requirement,
- expected execution depth,
- memory regime,
- admissible supervision modes,
- extrapolation axes,
- and known shortcut risks.

Also add support for optional gold execution traces at the task-spec level.

### Workstream B: Training Protocol v2

Define a benchmark-wide training protocol with:

- short-horizon pilot runs,
- long-horizon grokking runs on selected tasks,
- regularization sweeps for algorithmic tasks,
- checkpointed OOD evaluation over time,
- and fixed compute-budget accounting.

Outputs should include phase curves, not just final metrics.

### Workstream C: Process Supervision v1

For tasks that admit a reference trace, add three benchmark modes:

1. Answer-only supervision
2. Gold scratchpad or trace supervision
3. Self-generated scratchpad with final-answer supervision

The objective is to measure when traces are necessary, helpful, neutral, or harmful.

### Workstream D: Architecture Matrix v1

Turn the current "model list" into a real benchmark matrix. Suggested initial families:

- linear and tree baselines,
- tabular MLP,
- recurrent sequence baseline,
- encoder-decoder or decoder-only Transformer,
- structure-aware models where task family demands them,
- and one modern long-context or state-space family if feasible.

The important thing is not breadth for its own sake. It is having at least one plausible architecture per task regime.

### Workstream E: Representation And Positional Controls

Before any Transformer results are treated as benchmark evidence, specify:

- serialization formats,
- positional encoding ablations,
- context length policy,
- and training and test length schedules.

This should be treated as part of the benchmark contract, not left to per-model convenience.

### Workstream F: Reporting And Claims v2

Replace the single "solvability score" framing with a matrix that separates:

- answer IID performance,
- answer OOD performance,
- long-horizon best OOD performance,
- architecture sensitivity,
- process-supervision gain,
- positional and serialization sensitivity,
- and statistical confidence.

The benchmark may still compute a composite score, but papers should not lead with a single scalar.

## Concrete Plan Changes Against The Current TASK Structure

The current task graph is strong for engineering, but it should be revised for research use.

| Current plan item | Proposed change |
|---|---|
| TASK-12 Sequence Experiments | Reframe as pilot experiments until Methodology Reset is complete. |
| TASK-13 Classification Experiments | Reframe as pilot experiments until Methodology Reset is complete. |
| TASK-14 Diagnostics | Expand to include grokking trajectories, sample efficiency, counterfactuals, transfer, and positional and serialization ablations. |
| TASK-15 Algorithm Discovery | Keep as optional, but only after benchmark claims stabilize under the upgraded protocol. |

Recommended insertion before the current TASK-12 and TASK-13:

### New Pre-Experiment Phase: Methodology Reset

1. Define benchmark metadata and trace-capable task interfaces.
2. Define architecture matrix and compute-matching rules.
3. Define long-horizon training protocol and checkpoint schedule.
4. Define positional and serialization controls for sequence models.
5. Define upgraded reporting schema and acceptable claim language.

After that:

- Run TASK-12 and TASK-13 as pilot benchmark experiments.
- Use TASK-14 for confirmatory diagnostics and ablations.
- Only then publish benchmark-level solvability claims.

## Research Program: How This Can Become Multiple Papers

### Paper 1: Benchmark And Methodology

Deliverable:

- The benchmark specification, task families, split design, trace modes, architecture matrix, and claim protocol.

Main contribution:

- A research-grade benchmark that avoids common shortcut, training-dynamics, and positional confounds.

### Paper 2: Delayed Generalization And Grokking In Algorithmic Benchmarks

Deliverable:

- Long-horizon training studies across selected deterministic tasks.

Main contribution:

- When and where delayed OOD generalization appears, and how it depends on task family, regularization, and architecture.

### Paper 3: Process Supervision, Scratchpads, And Serial Computation

Deliverable:

- Answer-only versus trace-supervised versus self-generated-trace comparisons.

Main contribution:

- A more precise account of when intermediate computation is necessary, useful, or misleading in algorithmic tasks.

### Paper 4: Architecture And Representation Alignment

Deliverable:

- Cross-family experiments over task classes, positional encodings, and serialization schemes.

Main contribution:

- A map from task structure to successful inductive biases, with benchmark-level evidence rather than anecdote.

### Paper 5: Symbolic Recovery And Program Discovery

Deliverable:

- Symbolic extraction or synthesis only on tasks that have already demonstrated stable algorithmic generalization.

Main contribution:

- Separating "can predict the outputs" from "can recover the compact rule."

## Recommendation

Do not treat the current TASK-12 and TASK-13 plan as the first definitive benchmark pass.

Instead:

1. Keep the current infrastructure as the engineering base.
2. Insert a methodology-reset phase.
3. Run the existing experiment plan as a pilot under that revised protocol.
4. Use the pilot to lock the benchmark before making strong solvability claims.

That path preserves the strong work already done in the repository while making the project credible as a real research initiative rather than just a well-structured experiment harness.

## References

- Lake, B. M., and Baroni, M. "Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks." [arXiv:1711.00350](https://arxiv.org/abs/1711.00350)
- Bastings, J., Baroni, M., Weston, J., Cho, K., and Kiela, D. "Jump to better conclusions: SCAN both left and right." [arXiv:1809.04640](https://arxiv.org/abs/1809.04640)
- Power, A., Burda, Y., Edwards, H., Babuschkin, I., and Misra, V. "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Nye, M., Andreassen, A., Gur-Ari, G., Michalewski, H., Austin, J., et al. "Show Your Work: Scratchpads for Intermediate Computation with Language Models." [arXiv:2112.00114](https://arxiv.org/abs/2112.00114)
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Li, Z., Liu, H., Zhou, D., and Ma, T. "Chain of Thought Empowers Transformers to Solve Inherently Serial Problems." [arXiv:2402.12875](https://arxiv.org/abs/2402.12875)
- Velickovic, P., and Blundell, C. "Neural Algorithmic Reasoning." [arXiv:2105.02761](https://arxiv.org/abs/2105.02761)
- Velickovic, P., Puigdomenech Badia, A., Budden, D., Pascanu, R., Banino, A., Dashevskiy, M., Hadsell, R., and Blundell, C. "The CLRS Algorithmic Reasoning Benchmark." [arXiv:2205.15659](https://arxiv.org/abs/2205.15659)
- Press, O., Smith, N. A., and Lewis, M. "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Kazemnejad, A., Padhi, I., Natesan Ramamurthy, K., Das, P., and Reddy, S. "The Impact of Positional Encoding on Length Generalization in Transformers." [arXiv:2305.19466](https://arxiv.org/abs/2305.19466)
