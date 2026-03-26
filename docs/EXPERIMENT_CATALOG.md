# EXPERIMENT CATALOG: All Experiments, Shared Resources, and Validation

> **STATUS: EXECUTION-READY CATALOG**
> This is the authoritative catalog of every experiment to run, the shared infrastructure
> that supports them, the validation procedures that ensure correctness, and the
> execution plan with task dependencies for implementation.
> Read alongside `EXPERIMENT_DESIGN.md` which provides the rationale and theory.
> All implementation work should be traceable back to entries in this file.
>
> **Last Updated:** 2025-03-25

---

# How to Read This Document

This document has six major sections:

1. **Shared Resources** — code and infrastructure to implement first, shared by all experiments.
2. **Experiment Catalog** — every experiment to run, organized by phase.
3. **Validation Bounds** — how to verify correctness of every component (data, models, evaluation).
4. **Execution Plan** — task-level breakdown with IDs, dependencies, and acceptance criteria.
5. **Deviation Log** — structure for tracking changes from the plan during implementation.
6. **Appendices** — quick-reference tables and checklists.

Each experiment references shared resources by name. Each shared resource lists which experiments depend on it. This ensures nothing is built in isolation or duplicated.

---

# PART 1: SHARED RESOURCES

These are the foundational modules that must be implemented before any experiment runs.
Every resource is used by multiple experiments.

---

## SR-1: Task Registry

### What it is
A central registry where each task is a named entry exposing a standard interface.

### Interface per registered task

```python
class TaskSpec:
    task_id: str                    # unique identifier, e.g., "S1.2_sort"
    tier: str                       # e.g., "S1", "C2"
    track: str                      # "sequence" or "classification"
    description: str                # human-readable description
    input_schema: InputSchema       # defines feature types, shapes, ranges
    output_type: str                # "sequence", "scalar", "class"
    n_classes: int | None           # number of output classes (classification only)
    reference_algorithm: Callable   # the ground-truth function: input → output
    input_sampler: Callable         # generates random valid inputs
    verifier: Callable              # checks if a prediction matches the reference output
    complexity_metadata: dict       # depth, n_features_used, statefulness, etc.
```

### Used by
All experiments. Every experiment looks up its tasks from this registry.

### Validation
See [V-1: Task Registry Validation](#v-1-task-registry-validation).

---

## SR-2: Input Schema System

### What it is
A typed schema that describes the input format for each task.

### For sequence tasks

```python
class SequenceInputSchema:
    element_type: str           # "int", "binary", "char"
    min_length: int
    max_length: int
    value_range: tuple[int, int]
    alphabet: list | None       # for char/token types
```

### For classification tasks

```python
class TabularInputSchema:
    numerical_features: list[NumericalFeatureSpec]   # name, range, distribution
    categorical_features: list[CategoricalFeatureSpec]  # name, cardinality, values
    irrelevant_features: list[FeatureSpec]            # distractors
```

```python
class NumericalFeatureSpec:
    name: str
    min_val: float
    max_val: float
    distribution: str          # "uniform", "normal", "exponential"
    
class CategoricalFeatureSpec:
    name: str
    values: list[str]
    distribution: str          # "uniform", "weighted"
    weights: list[float] | None
```

### Used by
SR-1, SR-3, SR-5. Every task defines its inputs through this schema.

### Validation
See [V-2: Input Schema Validation](#v-2-input-schema-validation).

---

## SR-3: Data Generator

### What it is
A module that takes a TaskSpec, a sample count, and a random seed, and produces a labeled dataset.

### Interface

```python
def generate_dataset(
    task: TaskSpec,
    n_samples: int,
    seed: int,
    noise_config: NoiseConfig | None = None
) -> Dataset:
    """
    Returns:
        Dataset with fields:
            inputs: list of input objects (sequences or feature dicts)
            outputs: list of ground-truth outputs
            metadata: list of per-sample metadata dicts
    """
```

### Key properties
- **Deterministic**: same seed → same data.
- **Verifiable**: every output is checked against the reference algorithm at generation time.
- **Metadata-rich**: every sample carries its complexity metadata.

### Used by
All experiments. Every experiment calls this to produce train/val/test data.

### Validation
See [V-3: Data Generator Validation](#v-3-data-generator-validation).

---

## SR-4: Split Generator

### What it is
A module that takes a generated dataset and produces named train/val/test splits according to a split strategy.

### Interface

```python
def generate_splits(
    dataset: Dataset,
    task: TaskSpec,
    split_strategy: SplitStrategy
) -> dict[str, Dataset]:
    """
    Returns a dict mapping split names to subsets.
    Example keys: "train", "val_iid", "test_iid", 
                  "test_length_extrap", "test_value_extrap",
                  "test_adversarial", "test_distractor"
    """
```

### Supported split strategies

| Strategy | Applicable to | How it works |
|---|---|---|
| `IIDSplit` | all | random 70/15/15 |
| `LengthExtrapolationSplit` | sequence tasks | train on short, test on long |
| `ValueRangeExtrapolationSplit` | both | train on narrow range, test on wide |
| `CompositionSplit` | DSL tasks | train on depth ≤ d, test on d+1 |
| `TemplateHoldoutSplit` | DSL tasks | hold out specific operation combos |
| `CategoryCombinationSplit` | classification tasks | hold out feature-value combos |
| `DistractorSplit` | classification tasks | add irrelevant features in test |
| `ClassImbalanceSplit` | classification tasks | shift class ratios at test time |
| `AdversarialBoundarySplit` | classification tasks | oversample near decision boundary |

### Used by
All experiments. Every experiment specifies which splits to use.

### Validation
See [V-4: Split Generator Validation](#v-4-split-generator-validation).

---

## SR-5: Model Harness

### What it is
A unified training and prediction interface that wraps all model families.

### Interface

```python
class ModelHarness:
    def __init__(self, model_config: ModelConfig):
        ...
    
    def fit(self, train_data: Dataset, val_data: Dataset) -> TrainLog:
        ...
    
    def predict(self, inputs) -> Predictions:
        ...
    
    def predict_proba(self, inputs) -> Probabilities | None:
        ...
```

### Supported model configs

| Model | Config class | Track |
|---|---|---|
| Majority/constant predictor | `ConstantModelConfig` | both |
| Logistic regression | `LogRegConfig` | classification |
| Decision tree | `DecisionTreeConfig` | classification |
| Random forest | `RandomForestConfig` | classification |
| XGBoost / LightGBM | `GBTreeConfig` | classification |
| k-NN | `KNNConfig` | both |
| MLP | `MLPConfig` | both |
| LSTM | `LSTMConfig` | sequence |
| Transformer | `TransformerConfig` | both |

### Hyperparameter defaults
Each config has sensible defaults documented inline. Hyperparameter tuning is a secondary concern — the first goal is to get runs working with reasonable defaults. Tuning can be added later via a sweep config.

### Used by
All experiments.

### Validation
See [V-5: Model Harness Validation](#v-5-model-harness-validation).

---

## SR-6: Evaluation Engine

### What it is
A module that takes predictions, ground-truth outputs, and a task spec, and produces a structured evaluation report.

### Interface

```python
def evaluate(
    predictions: Predictions,
    ground_truth: Dataset,
    task: TaskSpec,
    split_name: str
) -> EvalReport:
    """
    Returns:
        EvalReport with:
            accuracy: float
            per_class_metrics: dict (classification) or None
            confusion_matrix: array (classification) or None
            exact_match: float (sequence)
            token_accuracy: float (sequence) or None
            error_taxonomy: dict[str, int]
            metadata_conditioned_metrics: dict
    """
```

### Used by
All experiments.

### Validation
See [V-6: Evaluation Engine Validation](#v-6-evaluation-engine-validation).

---

## SR-7: Experiment Runner

### What it is
An orchestrator that takes an experiment specification and runs the full pipeline:
generate data → split → train models → evaluate → produce report.

### Interface

```python
def run_experiment(
    experiment_spec: ExperimentSpec,
    seeds: list[int] = [42, 123, 456, 789, 1024]
) -> ExperimentReport:
    """
    Runs the full pipeline for each seed, aggregates results.
    """
```

### ExperimentSpec structure

```python
class ExperimentSpec:
    experiment_id: str
    task_ids: list[str]
    model_configs: list[ModelConfig]
    split_strategies: list[SplitStrategy]
    n_train_samples: int
    n_test_samples: int
    noise_config: NoiseConfig | None
    seeds: list[int]
```

### Used by
All experiments.

### Validation
See [V-7: Experiment Runner Validation](#v-7-experiment-runner-validation).

---

## SR-8: Report Generator

### What it is
Produces structured output (JSON + markdown summary) for each experiment run.

### Output structure

```
results/
├── {experiment_id}/
│   ├── config.json              # full experiment spec
│   ├── summary.md               # human-readable summary
│   ├── per_task/
│   │   ├── {task_id}/
│   │   │   ├── metrics.json     # all metrics per model per split
│   │   │   ├── confusion.png    # confusion matrices (classification)
│   │   │   ├── extrap_curve.png # extrapolation curves
│   │   │   └── errors.json      # error taxonomy
│   ├── comparison.md            # cross-task comparison table
│   └── solvability_verdicts.json  # per-task solvability labels
```

### Used by
All experiments.

### Validation
See [V-8: Report Generator Validation](#v-8-report-generator-validation).

---

## SR-9: Classification Rule DSL

### What it is
A small typed language for specifying classification rules programmatically.

### Why it exists
Many classification tasks (C2–C5) are built from compositions of predicates and combinators. Rather than hand-coding each, define them in a DSL and generate many tasks from one implementation.

### Primitives

```
# Predicates (return bool)
gt(feature, threshold)           # feature > threshold
lt(feature, threshold)           # feature < threshold
eq(feature, value)               # feature == value (categorical)
in_set(feature, values)          # feature in {values}
between(feature, lo, hi)         # lo <= feature <= hi

# Combinators (combine bools)
and(pred1, pred2, ...)
or(pred1, pred2, ...)
not(pred)
xor(pred1, pred2)
k_of_n(k, pred1, pred2, ..., predN)

# Classifiers (return class label)
if_then_else(predicate, class_if_true, class_if_false)
decision_list([(pred1, class1), (pred2, class2), ..., (default, classN)])
decision_tree(node_structure)

# Aggregators (for C4-tier tasks)
mean(feature, group_key)
count(predicate, group_key)
max(feature, group_key)
```

### Program sampling
Generate random valid rule trees with controlled:
- depth (1–5),
- number of features used,
- number of classes,
- predicate types.

### Used by
C2–C5 tier tasks, SR-1 (to register DSL-generated classification tasks).

### Validation
See [V-9: Classification Rule DSL Validation](#v-9-classification-rule-dsl-validation).

---

## SR-10: Sequence DSL

### What it is
A typed DSL for composing integer-list transformations, as described in EXPERIMENT_DESIGN.md.

### Primitives
`map`, `filter`, `sort`, `reverse`, `unique`, `take`, `drop`, `sum`, `count`, `max`, `min`, `parity`, `prefix_sum`, `zip`, `concat`, `mod`, `abs`, `sign`.

### Used by
S3 and S5 tier tasks.

### Validation
See [V-10: Sequence DSL Validation](#v-10-sequence-dsl-validation).

---

## Shared Resource Dependency Map

```
SR-2 (Input Schema)
 ├── SR-1 (Task Registry)     ← uses schemas to define tasks
 ├── SR-3 (Data Generator)    ← uses schemas to sample inputs
 └── SR-5 (Model Harness)     ← uses schemas to configure encoding

SR-9 (Classification DSL) ──→ SR-1 (Task Registry)  ← registers DSL-generated tasks
SR-10 (Sequence DSL) ────────→ SR-1 (Task Registry)  ← registers DSL-generated tasks

SR-1 (Task Registry)
 └── SR-3 (Data Generator)   ← looks up task to generate data

SR-3 (Data Generator)
 └── SR-4 (Split Generator)  ← splits generated datasets

SR-5 (Model Harness)
 └── SR-6 (Evaluation Engine) ← evaluates model predictions

SR-7 (Experiment Runner)
 ├── SR-1 (Task Registry)
 ├── SR-3 (Data Generator)
 ├── SR-4 (Split Generator)
 ├── SR-5 (Model Harness)
 ├── SR-6 (Evaluation Engine)
 └── SR-8 (Report Generator)
```

---

# PART 2: EXPERIMENT CATALOG

Every experiment listed below is a concrete, runnable unit of work.
Experiments are organized into phases. Each phase builds on the previous one.

---

## Phase 1: Pipeline Validation Experiments

These are not scientific experiments. They are engineering checks to confirm the shared resources work correctly before running real experiments.

### EXP-0.1: Smoke Test — Sequence Pipeline

**Goal:** Verify the full sequence pipeline works end-to-end on one trivial task.

| Field | Value |
|---|---|
| **Tasks** | S1.2 (sort) |
| **Models** | MLP, LSTM |
| **Splits** | IID only |
| **Samples** | 1,000 train / 200 test |
| **Seeds** | 1 |
| **Expected result** | LSTM achieves >90% exact match on IID sort of length 4–8. MLP may or may not. |
| **Pass criteria** | Pipeline runs without error. Metrics are computed. Report is generated. |
| **Resources used** | SR-1, SR-2, SR-3, SR-4, SR-5, SR-6, SR-7, SR-8, SR-10 |

### EXP-0.2: Smoke Test — Classification Pipeline

**Goal:** Verify the full classification pipeline works end-to-end on one trivial task.

| Field | Value |
|---|---|
| **Tasks** | C1.1 (numeric threshold) |
| **Models** | Logistic regression, Decision tree, MLP |
| **Splits** | IID only |
| **Samples** | 1,000 train / 200 test |
| **Seeds** | 1 |
| **Expected result** | Decision tree achieves 100% accuracy. Logistic regression achieves 100% (linear boundary). MLP near-perfect. |
| **Pass criteria** | Pipeline runs without error. Confusion matrix is generated. Per-class metrics reported. |
| **Resources used** | SR-1, SR-2, SR-3, SR-4, SR-5, SR-6, SR-7, SR-8, SR-9 |

### EXP-0.3: Smoke Test — Control Tasks

**Goal:** Verify that control tasks (random labels) produce expected low accuracy.

| Field | Value |
|---|---|
| **Tasks** | S0.1 (random labels), C0.1 (random class) |
| **Models** | MLP |
| **Splits** | IID only |
| **Samples** | 1,000 train / 200 test |
| **Seeds** | 1 |
| **Expected result** | Accuracy near chance level (1/n_classes). |
| **Pass criteria** | Accuracy is within expected random range. No suspiciously high accuracy. |
| **Resources used** | SR-1, SR-2, SR-3, SR-4, SR-5, SR-6, SR-7, SR-8 |

---

## Phase 2: Baseline Experiments — Sequence Track

These establish the core results for sequence-symbolic tasks.

### EXP-S1: Simple Sequence Transforms — IID and Extrapolation

**Goal:** Test whether models can learn simple one-step sequence transforms and extrapolate to longer inputs.

| Field | Value |
|---|---|
| **Tasks** | S1.1 (reverse), S1.2 (sort), S1.3 (rotate), S1.4 (count symbol), S1.5 (parity), S1.6 (prefix sum), S1.7 (deduplicate), S1.8 (extrema) |
| **Models** | Constant, Linear, MLP, LSTM, Transformer |
| **Splits** | IID, Length extrapolation, Value-range extrapolation |
| **Samples** | 10,000 train / 2,000 val / 2,000 test per split |
| **Seeds** | 5 |
| **Key question** | Which tasks can Transformer/LSTM solve in-distribution? Which extrapolate? |
| **Resources used** | SR-1 through SR-8, SR-10 |

### EXP-S2: Stateful Sequence Algorithms

**Goal:** Test whether models can learn stateful/iterative algorithms.

| Field | Value |
|---|---|
| **Tasks** | S2.1 (cumulative XOR), S2.2 (balanced parens), S2.3 (running min/max), S2.4 (FSM matcher), S2.5 (checksum), S2.6 (binary addition) |
| **Models** | MLP, LSTM, Transformer |
| **Splits** | IID, Length extrapolation |
| **Samples** | 10,000 train / 2,000 val / 2,000 test per split |
| **Seeds** | 5 |
| **Key question** | Does LSTM outperform Transformer on state-tracking tasks? |
| **Resources used** | SR-1 through SR-8, SR-10 |

### EXP-S3: Compositional Sequence Algorithms

**Goal:** Test whether models can learn multi-step compositional sequence transforms.

| Field | Value |
|---|---|
| **Tasks** | S3.1 (dedup-sort-count), S3.2 (map-filter-reduce), S3.3 (postfix eval), S3.4 (RLE), S3.5 (canonicalize) |
| **Models** | MLP, LSTM, Transformer |
| **Splits** | IID, Length extrapolation, Composition split |
| **Samples** | 10,000 train / 2,000 val / 2,000 test per split |
| **Seeds** | 5 |
| **Key question** | Does accuracy degrade with composition depth? Can models generalize to unseen compositions? |
| **Resources used** | SR-1 through SR-8, SR-10 |

### EXP-S4: Structural / Graph Algorithms (Stretch)

**Goal:** Test whether models can learn graph-level algorithms from serialized input.

| Field | Value |
|---|---|
| **Tasks** | S4.1 (shortest path), S4.2 (connectivity), S4.3 (merge intervals), S4.4 (edit distance), S4.5 (topo sort check) |
| **Models** | LSTM, Transformer |
| **Splits** | IID, Size extrapolation (more nodes/edges) |
| **Samples** | 5,000 train / 1,000 val / 1,000 test per split |
| **Seeds** | 3 |
| **Key question** | Can sequence models solve structural tasks, or is structure-aware encoding essential? |
| **Resources used** | SR-1 through SR-8 |
| **Note** | Lower priority. Run only after EXP-S1 through EXP-S3 are stable. |

### EXP-S5: DSL Program Family

**Goal:** Sweep across DSL-generated programs of varying depth.

| Field | Value |
|---|---|
| **Tasks** | 50–100 DSL-generated programs from SR-10, depths 1–4 |
| **Models** | LSTM, Transformer |
| **Splits** | IID, Composition split, Template holdout |
| **Samples** | 5,000 train / 1,000 val / 1,000 test per program |
| **Seeds** | 3 |
| **Key question** | How does accuracy scale with program depth? Can models transfer across programs? |
| **Resources used** | SR-1 through SR-8, SR-10 |
| **Note** | This is the largest experiment in the sequence track. |

---

## Phase 3: Baseline Experiments — Classification Track

These establish the core results for classification tasks with mixed-type inputs.

### EXP-C1: Single-Rule Threshold Classification

**Goal:** Test whether models can learn the simplest classification rules and extrapolate beyond training value ranges.

| Field | Value |
|---|---|
| **Tasks** | C1.1 (numeric threshold), C1.2 (range binning), C1.3 (categorical match), C1.4 (majority vote), C1.5 (numeric comparison), C1.6 (modular class) |
| **Models** | Constant, Logistic regression, Decision tree, Random forest, XGBoost, MLP |
| **Splits** | IID, Value-range extrapolation, Class imbalance shift |
| **Samples** | 5,000 train / 1,000 val / 1,000 test per split |
| **Seeds** | 5 |
| **Key question** | Which models achieve perfect IID accuracy? Which extrapolate the threshold to unseen value ranges? |
| **Expected insights** | Decision tree should achieve 100% IID. Logistic regression should work on C1.1/C1.5 but struggle on C1.2/C1.6. Extrapolation depends on model type. |
| **Resources used** | SR-1 through SR-8, SR-9 |

### EXP-C2: Multi-Feature Rule Classification

**Goal:** Test models on classification rules that combine multiple features.

| Field | Value |
|---|---|
| **Tasks** | C2.1 (AND), C2.2 (OR), C2.3 (nested if-else), C2.4 (linear boundary), C2.5 (k-of-n), C2.6 (categorical gate) |
| **Models** | Logistic regression, Decision tree, Random forest, XGBoost, MLP |
| **Splits** | IID, Value-range extrapolation, Unseen category combinations, Distractor injection |
| **Samples** | 5,000 train / 1,000 val / 1,000 test per split |
| **Seeds** | 5 |
| **Key question** | Can tree models perfectly learn conjunctive/disjunctive rules? Does adding distractors degrade performance? |
| **Expected insights** | Trees should dominate on axis-aligned rules. MLP should handle C2.4 (linear boundary). C2.6 (categorical gate) is the hardest — it tests conditional thresholds. |
| **Resources used** | SR-1 through SR-8, SR-9 |

### EXP-C3: Feature Interaction Classification

**Goal:** Test models on tasks that require learning feature interactions.

| Field | Value |
|---|---|
| **Tasks** | C3.1 (XOR), C3.2 (parity over categoricals), C3.3 (rank-based), C3.4 (distance-based), C3.5 (interaction polynomial), C3.6 (conditional shift) |
| **Models** | Logistic regression, Decision tree, Random forest, XGBoost, MLP, Transformer |
| **Splits** | IID, Value-range extrapolation, Unseen category combinations, Adversarial boundary |
| **Samples** | 5,000 train / 1,000 val / 1,000 test per split |
| **Seeds** | 5 |
| **Key question** | Do any models learn XOR and parity interactions reliably? Does the Transformer add value over tree ensembles on interaction tasks? |
| **Expected insights** | Linear models will completely fail XOR/parity. Trees need sufficient depth. MLP and Transformer may learn interactions with enough data. |
| **Resources used** | SR-1 through SR-8, SR-9 |

### EXP-C4: Aggregation-Dependent Classification

**Goal:** Test models on tasks that require computing aggregates before classifying.

| Field | Value |
|---|---|
| **Tasks** | C4.1 (group statistics), C4.2 (outlier flag), C4.3 (sequence pattern), C4.4 (cumulative threshold), C4.5 (voting ensemble) |
| **Models** | XGBoost, MLP, LSTM (for C4.3/C4.4), Transformer |
| **Splits** | IID, Value-range extrapolation, Group size extrapolation |
| **Samples** | 5,000 train / 1,000 val / 1,000 test per split |
| **Seeds** | 5 |
| **Key question** | Can models implicitly compute group-level aggregates? Does sequential structure in C4.3/C4.4 require sequence models? |
| **Resources used** | SR-1 through SR-8, SR-9 |

### EXP-C5: Multi-Step Compositional Classification

**Goal:** Test models on classification rules that chain multiple operations.

| Field | Value |
|---|---|
| **Tasks** | C5.1 (feature engineering pipeline), C5.2 (hierarchical classification), C5.3 (multi-stage filter), C5.4 (DSL classification programs), C5.5 (lookup-then-classify) |
| **Models** | XGBoost, MLP, Transformer |
| **Splits** | IID, Rule complexity increase, Value-range extrapolation |
| **Samples** | 5,000 train / 1,000 val / 1,000 test per split |
| **Seeds** | 5 |
| **Key question** | Does accuracy degrade with rule depth? Can models generalize to deeper compositions of known rule primitives? |
| **Resources used** | SR-1 through SR-8, SR-9 |

---

## Phase 4: Cross-Cutting Diagnostic Experiments

These experiments use the results from Phase 2 and 3 to answer deeper questions.

### EXP-D1: Sample Efficiency Comparison

**Goal:** Measure how quickly models learn algorithmic tasks vs. control tasks.

| Field | Value |
|---|---|
| **Tasks** | 3 best-performing tasks from each track + 2 control tasks |
| **Models** | Best model per task from Phase 2/3 |
| **Method** | Train on 100, 500, 1000, 2000, 5000, 10000 samples. Plot learning curves. |
| **Seeds** | 5 |
| **Key question** | Do algorithmic tasks show steeper learning curves than controls? |
| **Resources used** | SR-1 through SR-8 |

### EXP-D2: Distractor Feature Robustness

**Goal:** Measure how adding irrelevant features affects classification accuracy.

| Field | Value |
|---|---|
| **Tasks** | C1.1, C2.1, C2.6, C3.1, C3.6 |
| **Models** | Decision tree, XGBoost, MLP |
| **Method** | Train with 0, 2, 5, 10, 20 distractor features. Report accuracy degradation. |
| **Seeds** | 5 |
| **Key question** | Which models are most robust to distractors? At what distractor count does performance break? |
| **Resources used** | SR-1 through SR-8 |

### EXP-D3: Noise Robustness

**Goal:** Measure how input noise affects model accuracy on algorithmic classification tasks.

| Field | Value |
|---|---|
| **Tasks** | C1.1, C2.1, C3.1, C3.4 |
| **Models** | Decision tree, XGBoost, MLP |
| **Method** | Inject Gaussian noise into numeric features at levels σ = 0, 0.01, 0.05, 0.1, 0.2 (relative to feature range). Report accuracy vs. noise level. |
| **Seeds** | 5 |
| **Key question** | How gracefully does performance degrade? Do boundary-proximity errors dominate? |
| **Resources used** | SR-1 through SR-8 |

### EXP-D4: Feature Importance Alignment

**Goal:** Check whether trained models identify the correct features as important.

| Field | Value |
|---|---|
| **Tasks** | C2.1, C2.6, C3.1 (with 5 distractors each) |
| **Models** | Decision tree, XGBoost, MLP (with SHAP or permutation importance) |
| **Method** | Compare model-derived feature importance ranking to the ground-truth features used by the reference algorithm. |
| **Seeds** | 5 |
| **Key question** | Do models correctly identify which features the algorithm uses? |
| **Resources used** | SR-1 through SR-8 |

### EXP-D5: Solvability Verdict Calibration

**Goal:** Compute the solvability score for every task and verify the scoring system is meaningful.

| Field | Value |
|---|---|
| **Tasks** | All tasks from Phase 2 and 3 |
| **Models** | All models from Phase 2 and 3 |
| **Method** | Compute the solvability score (from EXPERIMENT_DESIGN.md Section 11.5) for each task. Verify that control tasks score WEAK or NEGATIVE, simple algorithmic tasks score MODERATE or STRONG, and difficult tasks score appropriately. |
| **Seeds** | Use results from Phase 2/3 |
| **Key question** | Does the solvability score correctly rank tasks by how algorithmically learnable they are? |
| **Resources used** | SR-6, SR-8 |

---

## Phase 5: Bonus — Algorithm Discovery

### EXP-B1: Rule Extraction from Classification Models

**Goal:** Attempt to recover the hidden classification rule from a trained model.

| Field | Value |
|---|---|
| **Tasks** | Top 5 classification tasks with STRONG solvability score |
| **Models** | Decision tree (direct extraction), XGBoost (approximate extraction), MLP (decision boundary probing) |
| **Method** | Extract the learned decision tree structure. Compare to the reference rule. Measure structural similarity and functional equivalence on a hard test set. |
| **Pass criteria** | Extracted rule matches reference algorithm on >99% of hard test samples. |
| **Resources used** | SR-1 through SR-8, SR-9 |

### EXP-B2: DSL Program Search for Sequence Tasks

**Goal:** Attempt to recover the hidden DSL program from a model's behavior.

| Field | Value |
|---|---|
| **Tasks** | Top 5 sequence tasks with STRONG solvability score from S5 tier |
| **Method** | Use the trained model as an oracle to guide search over SR-10 DSL programs. Score candidate programs by agreement with the model on held-out inputs. Validate best candidate against reference algorithm. |
| **Pass criteria** | Found program is functionally equivalent to the reference on a hard test set. |
| **Resources used** | SR-1 through SR-8, SR-10 |

---

# PART 3: VALIDATION BOUNDS

Every shared resource and every experiment must be validated before results are trusted.
This section defines how to validate each component.

---

## V-1: Task Registry Validation

### What to check

1. **Completeness**: every task listed in EXPERIMENT_DESIGN.md tiers is registered.
2. **Interface compliance**: every registered task has all required fields (task_id, tier, track, reference_algorithm, input_sampler, verifier, complexity_metadata).
3. **Reference algorithm determinism**: calling `reference_algorithm(input)` twice with the same input produces the same output.
4. **Verifier consistency**: `verifier(reference_algorithm(input), reference_algorithm(input))` returns True for all inputs.
5. **Cross-validation**: for at least 5 hand-computed examples per task, the reference algorithm matches the expected output.

### How to check

```python
def validate_task_registry(registry):
    for task in registry.all_tasks():
        # Interface check
        assert hasattr(task, 'task_id')
        assert hasattr(task, 'reference_algorithm')
        assert hasattr(task, 'input_sampler')
        assert hasattr(task, 'verifier')
        
        # Determinism check
        for _ in range(100):
            inp = task.input_sampler(seed=42)
            out1 = task.reference_algorithm(inp)
            out2 = task.reference_algorithm(inp)
            assert out1 == out2, f"Non-deterministic: {task.task_id}"
        
        # Verifier consistency
        for _ in range(100):
            inp = task.input_sampler()
            out = task.reference_algorithm(inp)
            assert task.verifier(out, out), f"Verifier fails on correct output: {task.task_id}"
        
        # Verifier rejects wrong answers
        for _ in range(100):
            inp = task.input_sampler()
            out = task.reference_algorithm(inp)
            wrong = corrupt(out)  # flip a bit, change a class, etc.
            if wrong != out:
                assert not task.verifier(wrong, out), f"Verifier accepts wrong output: {task.task_id}"
```

### Acceptance criteria
All checks pass for every registered task with zero exceptions.

---

## V-2: Input Schema Validation

### What to check

1. **Schema completeness**: every field has a defined type, range, and distribution.
2. **Sampling validity**: sampled inputs conform to the schema (values in range, correct types, correct cardinalities).
3. **Reproducibility**: same seed → same inputs.
4. **Distribution check**: for large samples, the empirical distribution matches the specified distribution (e.g., uniform values are roughly uniform).

### How to check

```python
def validate_input_schema(schema, n=10000, seed=42):
    samples = [schema.sample(seed=seed + i) for i in range(n)]
    
    for s in samples:
        # Type check
        for feat, spec in schema.features():
            assert type(s[feat]) == spec.expected_type
            
            # Range check (numerical)
            if spec.is_numerical:
                assert spec.min_val <= s[feat] <= spec.max_val
            
            # Cardinality check (categorical)
            if spec.is_categorical:
                assert s[feat] in spec.values
    
    # Reproducibility check
    s1 = schema.sample(seed=42)
    s2 = schema.sample(seed=42)
    assert s1 == s2
    
    # Distribution check (Kolmogorov-Smirnov for numerical, chi-squared for categorical)
    # ...
```

### Acceptance criteria
All type, range, and reproducibility checks pass. Distribution checks pass with p > 0.01.

---

## V-3: Data Generator Validation

### What to check

1. **Label correctness**: every generated output matches the reference algorithm applied to the input.
2. **Determinism**: same task + same seed → same dataset.
3. **No data leakage**: inputs in different splits are disjoint.
4. **Metadata correctness**: logged metadata matches actual sample properties.
5. **Class balance**: reported class ratios match actual ratios.

### How to check

```python
def validate_data_generator(task, n=5000, seed=42):
    dataset = generate_dataset(task, n, seed)
    
    # Label correctness: re-verify every sample
    for inp, out in zip(dataset.inputs, dataset.outputs):
        expected = task.reference_algorithm(inp)
        assert task.verifier(out, expected), f"Label mismatch: {inp} → {out} (expected {expected})"
    
    # Determinism
    dataset2 = generate_dataset(task, n, seed)
    assert dataset.inputs == dataset2.inputs
    assert dataset.outputs == dataset2.outputs
    
    # Metadata correctness
    for inp, meta in zip(dataset.inputs, dataset.metadata):
        if 'input_length' in meta:
            assert meta['input_length'] == len(inp)
        if 'n_classes' in meta:
            assert meta['n_classes'] == task.n_classes
    
    # Class balance check
    from collections import Counter
    class_counts = Counter(dataset.outputs)
    reported_balance = dataset.metadata[0].get('class_balance')
    if reported_balance:
        for cls, ratio in reported_balance.items():
            actual_ratio = class_counts[cls] / n
            assert abs(actual_ratio - ratio) < 0.05, f"Class balance mismatch for {cls}"
```

### Acceptance criteria
100% label correctness. Determinism holds exactly. Metadata matches.

---

## V-4: Split Generator Validation

### What to check

1. **Disjointness**: no sample appears in more than one split.
2. **Coverage**: all samples from the source dataset appear in exactly one split.
3. **Split property**: each split satisfies its defining property.
4. **Reproducibility**: same seed → same splits.

### Property checks per split type

| Split | Property to verify |
|---|---|
| IID | Random assignment; approximately target ratios (70/15/15) |
| Length extrapolation | All train samples have length ≤ L_train; all test samples have length > L_train |
| Value-range extrapolation | All train numeric values in [lo_train, hi_train]; test values outside this range |
| Category combination | Held-out combinations never appear in training |
| Distractor injection | Test inputs have additional columns not present in training |
| Class imbalance shift | Training class ratios ≠ test class ratios by specified margin |
| Adversarial boundary | Test samples are closer to the decision boundary than training samples (on average) |
| Composition | Train programs have depth ≤ d; test programs have depth > d |

### Acceptance criteria
All disjointness, coverage, and property checks pass with zero violations.

---

## V-5: Model Harness Validation

### What to check

1. **Overfit test**: on a tiny dataset (50 samples), every model can overfit to 100% training accuracy. If it cannot, the model implementation or encoding is broken.
2. **Random label test**: on random labels, no model achieves significantly above-chance test accuracy. If it does, there is data leakage.
3. **Known-solution test**: on a task with a known-easy solution (e.g., C1.1 numeric threshold), decision tree achieves 100% accuracy. If not, the encoding or training loop is broken.
4. **Prediction shape**: model output shape matches expected output shape for the task.
5. **Reproducibility**: same seed → same trained model → same predictions.

### How to check

```python
def validate_model_harness(model_config, task, n=50, seed=42):
    # Overfit test
    tiny_data = generate_dataset(task, n, seed)
    model = ModelHarness(model_config)
    model.fit(tiny_data, tiny_data)  # train and val are same
    preds = model.predict(tiny_data.inputs)
    train_acc = accuracy(preds, tiny_data.outputs)
    assert train_acc > 0.95, f"Cannot overfit: {model_config.name} on {task.task_id}, acc={train_acc}"
    
    # Random label test
    random_data = generate_dataset(task, 1000, seed)
    random_data.outputs = shuffle(random_data.outputs)
    splits = generate_splits(random_data, task, IIDSplit())
    model2 = ModelHarness(model_config)
    model2.fit(splits['train'], splits['val_iid'])
    test_preds = model2.predict(splits['test_iid'].inputs)
    test_acc = accuracy(test_preds, splits['test_iid'].outputs)
    chance = 1.0 / task.n_classes
    assert test_acc < chance + 0.10, f"Suspiciously high acc on random labels: {test_acc}"
    
    # Prediction shape check
    assert len(preds) == len(tiny_data.inputs)
    
    # Reproducibility
    model3 = ModelHarness(model_config)
    model3.fit(tiny_data, tiny_data)
    preds3 = model3.predict(tiny_data.inputs)
    assert preds == preds3
```

### Acceptance criteria
All four checks pass for every (model, task) combination used in the experiment catalog.

---

## V-6: Evaluation Engine Validation

### What to check

1. **Perfect prediction test**: if predictions exactly match ground truth, all accuracy metrics are 1.0, all error counts are 0.
2. **Worst prediction test**: if predictions are all one wrong class, metrics are correct for that failure mode.
3. **Known confusion test**: create a set of predictions with a known confusion matrix and verify the engine reproduces it exactly.
4. **Metric consistency**: accuracy == (sum of diagonal of confusion matrix) / total.
5. **Classification vs. sequence dispatch**: the engine uses the correct metric set based on task type.

### How to check

```python
def validate_evaluation_engine():
    # Perfect prediction test
    gt = ["A", "B", "A", "B", "A"]
    pred = ["A", "B", "A", "B", "A"]
    report = evaluate(pred, gt, mock_classification_task, "test")
    assert report.accuracy == 1.0
    assert report.per_class_metrics["A"]["f1"] == 1.0
    assert report.per_class_metrics["B"]["f1"] == 1.0
    
    # Worst prediction test
    pred_wrong = ["B", "A", "B", "A", "B"]
    report2 = evaluate(pred_wrong, gt, mock_classification_task, "test")
    assert report2.accuracy == 0.0
    
    # Known confusion test
    gt3 = ["A", "A", "A", "B", "B", "B"]
    pred3 = ["A", "A", "B", "B", "B", "A"]
    report3 = evaluate(pred3, gt3, mock_classification_task, "test")
    assert report3.confusion_matrix == [[2, 1], [1, 2]]
    
    # Metric consistency
    cm = report3.confusion_matrix
    expected_acc = sum(cm[i][i] for i in range(len(cm))) / sum(sum(row) for row in cm)
    assert abs(report3.accuracy - expected_acc) < 1e-9
```

### Acceptance criteria
All checks pass with zero numerical discrepancy (within floating-point tolerance).

---

## V-7: Experiment Runner Validation

### What to check

1. **End-to-end integrity**: run a complete mini-experiment and verify every output artifact exists and is well-formed.
2. **Seed variation**: running with different seeds produces different data but the same task definition.
3. **Multi-seed aggregation**: the aggregated report correctly computes mean and standard deviation across seeds.
4. **No cross-contamination**: results for task A are not mixed into task B's report.

### How to check

Run EXP-0.1 and EXP-0.2 (smoke tests) and manually inspect:

- all expected files exist in the results directory,
- config.json matches the experiment spec,
- metrics.json contains all expected metric keys,
- summary.md is human-readable and internally consistent,
- cross-seed aggregation shows mean ± std.

### Acceptance criteria
All artifacts exist, are correctly structured, and cross-seed aggregation is numerically correct.

---

## V-8: Report Generator Validation

### What to check

1. **File structure**: the expected directory tree is created.
2. **JSON validity**: all JSON files parse without error.
3. **Markdown validity**: summary.md renders correctly.
4. **Metric consistency**: metrics in the summary match metrics in the JSON.
5. **Solvability verdict logic**: the verdict label matches the criteria defined in EXPERIMENT_DESIGN.md Section 9.4.

### Acceptance criteria
All structural and consistency checks pass.

---

## V-9: Classification Rule DSL Validation

### What to check

1. **Type safety**: DSL programs that violate type constraints are rejected at construction time.
2. **Determinism**: applying a rule to the same input always produces the same output.
3. **Coverage**: for any input conforming to the schema, the rule produces a valid class label (no unhandled cases).
4. **Known-rule test**: hand-write 5 rules, generate 1000 samples each, verify all labels match hand-computed expectations.
5. **Depth correctness**: the reported depth of a rule tree matches the actual nesting depth.
6. **Equivalence check**: two syntactically different rules that are semantically equivalent produce the same labels on shared inputs.

### Acceptance criteria
All checks pass. Zero unhandled inputs. Zero mismatches on hand-verified rules.

---

## V-10: Sequence DSL Validation

### What to check

1. **Type safety**: programs that produce type errors (e.g., applying `sort` to a scalar) are rejected.
2. **Determinism**: same program + same input → same output.
3. **Known-program test**: hand-write 5 programs, generate 100 inputs each, verify all outputs.
4. **Depth correctness**: reported program depth matches actual nesting.
5. **Deduplication**: the sampler does not produce semantically equivalent programs for different program IDs (tested empirically on 1000 random inputs).

### Acceptance criteria
All checks pass.

---

## V-GLOBAL: Cross-Component Validation

These checks verify that components work correctly together.

### V-G1: Round-trip check
Generate data (SR-3) → split (SR-4) → train model (SR-5) → evaluate (SR-6) → report (SR-8). Verify the reported accuracy matches what you get by manually comparing predictions to ground truth.

### V-G2: Control task calibration
Control tasks (S0, C0) should produce WEAK or NEGATIVE solvability verdicts. If they produce MODERATE or STRONG, the evaluation pipeline has a bug.

### V-G3: Trivial task ceiling
Tasks like C1.1 (numeric threshold) should produce STRONG solvability for decision trees. If they do not, the pipeline has a bug.

### V-G4: Data-model isolation
Models trained on task A should not accidentally see data from task B. Verify by checking that the experiment runner creates fresh data per task.

---

# PART 4: EXECUTION PLAN

This section breaks the project into **atomic implementation tasks** with unique IDs, explicit dependencies, acceptance criteria, and estimated effort. Use this as the work-tracking backbone.

## Dependency Graph (DAG)

```
TASK-01 ─────────────────────────────────────────────────────────┐
  (Input Schema)                                                 │
    │                                                            │
    ├──► TASK-02 (Classification Rule DSL) ──┐                   │
    │                                        ├──► TASK-04        │
    ├──► TASK-03 (Sequence DSL) ─────────────┘   (Task Registry) │
    │                                              │             │
    │    ┌─────────────────────────────────────────┘             │
    │    │                                                       │
    │    ▼                                                       │
    │  TASK-05 (Data Generator) ──► TASK-06 (Split Generator)    │
    │                                   │                        │
    │                                   ▼                        │
    ├──► TASK-07 (Model Harness) ──► TASK-08 (Eval Engine)       │
    │                                   │                        │
    │    ┌──────────────────────────────┘                        │
    │    ▼                                                       │
    │  TASK-09 (Experiment Runner) ──► TASK-10 (Report Gen)      │
    │                                   │                        │
    │    ┌──────────────────────────────┘                        │
    │    ▼                                                       │
    │  TASK-11 (Smoke Tests: EXP-0.x)                            │
    │    │                                                       │
    │    ├──► TASK-12 (Sequence Experiments: EXP-S1–S5)          │
    │    ├──► TASK-13 (Classification Experiments: EXP-C1–C5)    │
    │    │         │                                             │
    │    │         ├──► TASK-14 (Diagnostics: EXP-D1–D5)         │
    │    │         │         │                                   │
    │    │         │         └──► TASK-15 (Bonus: EXP-B1–B2)     │
    │    │         │                                             │
    └────┴─────────┴─────────────────────────────────────────────┘
```

**Legend:** `A ──► B` means B depends on A (A must be complete before B starts).

---

## Task Table

Every task below must be completed in an order consistent with the dependency graph. Tasks at the same depth with no mutual dependency can be parallelized.

---

### TASK-01: Input Schema System

| Field | Value |
|---|---|
| **Builds** | SR-2 |
| **Depends on** | — (no dependencies) |
| **Deliverables** | `SequenceInputSchema`, `TabularInputSchema`, `NumericalFeatureSpec`, `CategoricalFeatureSpec` classes |
| **Acceptance criteria** | V-2 passes: type checks, range checks, reproducibility, distribution tests (p > 0.01) |
| **Estimated effort** | 0.5 day |
| **Source files** | `src/schemas.py` (suggested) |

---

### TASK-02: Classification Rule DSL

| Field | Value |
|---|---|
| **Builds** | SR-9 |
| **Depends on** | TASK-01 |
| **Deliverables** | Predicate primitives (`gt`, `lt`, `eq`, `in_set`, `between`), combinators (`and`, `or`, `not`, `xor`, `k_of_n`), classifiers (`if_then_else`, `decision_list`, `decision_tree`), aggregators (`mean`, `count`, `max`), rule sampler with depth control |
| **Acceptance criteria** | V-9 passes: type safety, determinism, coverage, 5 hand-verified rules match, depth correctness, equivalence check |
| **Estimated effort** | 1–1.5 days |
| **Source files** | `src/dsl/classification_dsl.py` (suggested) |

---

### TASK-03: Sequence DSL

| Field | Value |
|---|---|
| **Builds** | SR-10 |
| **Depends on** | TASK-01 |
| **Deliverables** | Typed primitives (`map`, `filter`, `sort`, `reverse`, `unique`, `take`, `drop`, `sum`, `count`, `max`, `min`, `parity`, `prefix_sum`, `zip`, `concat`, `mod`, `abs`, `sign`), program sampler with depth/type control, deduplication |
| **Acceptance criteria** | V-10 passes: type safety, determinism, 5 hand-verified programs match, depth correctness, deduplication |
| **Estimated effort** | 1–1.5 days |
| **Source files** | `src/dsl/sequence_dsl.py` (suggested) |

---

### TASK-04: Task Registry

| Field | Value |
|---|---|
| **Builds** | SR-1 |
| **Depends on** | TASK-01, TASK-02, TASK-03 |
| **Deliverables** | `TaskSpec` dataclass, registry with all S0–S5 and C0–C5 tasks registered, lookup by ID/tier/track |
| **Acceptance criteria** | V-1 passes: every task registered, interface compliance, determinism (100 trials), verifier consistency, 5 hand-computed examples per task match |
| **Estimated effort** | 1–1.5 days |
| **Source files** | `src/registry.py`, `src/tasks/` (suggested) |

**Milestone: FOUNDATION COMPLETE** — after TASK-04, you have schemas, DSLs, and all tasks defined. Validate V-1, V-2, V-9, V-10 before proceeding.

---

### TASK-05: Data Generator

| Field | Value |
|---|---|
| **Builds** | SR-3 |
| **Depends on** | TASK-04 |
| **Deliverables** | `generate_dataset(task, n_samples, seed, noise_config)` → `Dataset`, `NoiseConfig` class, metadata recording |
| **Acceptance criteria** | V-3 passes: 100% label correctness (re-verified), determinism, metadata matches, class balance matches |
| **Estimated effort** | 1 day |
| **Source files** | `src/data_generator.py` (suggested) |

---

### TASK-06: Split Generator

| Field | Value |
|---|---|
| **Builds** | SR-4 |
| **Depends on** | TASK-05 |
| **Deliverables** | `generate_splits(dataset, task, split_strategy)` → `dict[str, Dataset]`, all 9 split strategy classes |
| **Acceptance criteria** | V-4 passes: disjointness, coverage, per-strategy property checks, reproducibility |
| **Estimated effort** | 1 day |
| **Source files** | `src/splits.py` (suggested) |

**Milestone: DATA PIPELINE COMPLETE** — after TASK-06, you can generate verified, split data for any task.

---

### TASK-07: Model Harness

| Field | Value |
|---|---|
| **Builds** | SR-5 |
| **Depends on** | TASK-01 (for encoding), TASK-05 (for test data) |
| **Deliverables** | `ModelHarness` with `fit`, `predict`, `predict_proba`; config classes for all 9 model types; encoding logic for sequence and tabular inputs |
| **Acceptance criteria** | V-5 passes for every (model, representative task) pair: overfit test, random label test, known-solution test (C1.1 → 100% for decision tree), prediction shape, reproducibility |
| **Estimated effort** | 2–3 days |
| **Source files** | `src/models/harness.py`, `src/models/configs.py`, `src/models/encoders.py` (suggested) |

---

### TASK-08: Evaluation Engine

| Field | Value |
|---|---|
| **Builds** | SR-6 |
| **Depends on** | TASK-07 |
| **Deliverables** | `evaluate(predictions, ground_truth, task, split_name)` → `EvalReport` with all classification and sequence metrics, error taxonomy, metadata-conditioned breakdown |
| **Acceptance criteria** | V-6 passes: perfect prediction test, worst prediction test, known confusion matrix test, metric consistency, correct dispatch by task type |
| **Estimated effort** | 1 day |
| **Source files** | `src/evaluation.py` (suggested) |

---

### TASK-09: Experiment Runner

| Field | Value |
|---|---|
| **Builds** | SR-7 |
| **Depends on** | TASK-04, TASK-05, TASK-06, TASK-07, TASK-08 |
| **Deliverables** | `run_experiment(experiment_spec, seeds)` → `ExperimentReport`, `ExperimentSpec` dataclass, multi-seed aggregation (mean ± std), progress logging |
| **Acceptance criteria** | V-7 passes: end-to-end artifact integrity, seed variation, multi-seed aggregation, no cross-contamination |
| **Estimated effort** | 1 day |
| **Source files** | `src/runner.py` (suggested) |

---

### TASK-10: Report Generator

| Field | Value |
|---|---|
| **Builds** | SR-8 |
| **Depends on** | TASK-09 |
| **Deliverables** | Structured output directory (`results/{experiment_id}/...`), `config.json`, `summary.md`, per-task `metrics.json`, `confusion.png`, `extrap_curve.png`, `errors.json`, `comparison.md`, `solvability_verdicts.json` |
| **Acceptance criteria** | V-8 passes: file structure correct, JSON valid, markdown renders, metrics consistent, solvability verdict logic matches Section 9.4 of EXPERIMENT_DESIGN.md |
| **Estimated effort** | 1 day |
| **Source files** | `src/reporting.py` (suggested) |

**Milestone: FULL PIPELINE COMPLETE** — after TASK-10, the entire infrastructure is built and individually validated. Run V-Global checks.

---

### TASK-11: Pipeline Smoke Tests

| Field | Value |
|---|---|
| **Runs** | EXP-0.1, EXP-0.2, EXP-0.3 |
| **Depends on** | TASK-10 |
| **Deliverables** | Passing smoke test results in `results/EXP-0.x/` |
| **Acceptance criteria** | EXP-0.1: pipeline runs, LSTM >90% exact match on sort. EXP-0.2: decision tree 100% on C1.1. EXP-0.3: random tasks produce near-chance accuracy. V-G1 through V-G4 all pass. |
| **Estimated effort** | 0.5 day |

**Gate: DO NOT proceed to TASK-12/13 until all smoke tests pass and V-Global checks are green.**

---

### TASK-12: Sequence Track Experiments

| Field | Value |
|---|---|
| **Runs** | EXP-S1, EXP-S2, EXP-S3, EXP-S4 (stretch), EXP-S5 |
| **Depends on** | TASK-11 |
| **Deliverables** | Results in `results/EXP-S{1-5}/`, solvability verdicts per task |
| **Acceptance criteria** | All experiments complete without error. Solvability verdicts are assigned. Results are reproducible across seeds. |
| **Estimated effort** | 2–3 days (compute-dependent) |
| **Note** | EXP-S4 is lower priority — run last within this task, skip if time-constrained. |

---

### TASK-13: Classification Track Experiments

| Field | Value |
|---|---|
| **Runs** | EXP-C1, EXP-C2, EXP-C3, EXP-C4, EXP-C5 |
| **Depends on** | TASK-11 |
| **Deliverables** | Results in `results/EXP-C{1-5}/`, solvability verdicts per task |
| **Acceptance criteria** | All experiments complete without error. Solvability verdicts are assigned. Results are reproducible across seeds. |
| **Estimated effort** | 2–3 days (compute-dependent) |
| **Note** | Can run in parallel with TASK-12. |

---

### TASK-14: Diagnostic Experiments

| Field | Value |
|---|---|
| **Runs** | EXP-D1, EXP-D2, EXP-D3, EXP-D4, EXP-D5 |
| **Depends on** | TASK-12, TASK-13 (uses their results to select tasks/models) |
| **Deliverables** | Results in `results/EXP-D{1-5}/`, learning curves, distractor/noise degradation plots, feature importance comparison, solvability calibration report |
| **Acceptance criteria** | All diagnostic experiments complete. EXP-D5 confirms solvability scores are well-calibrated (controls score WEAK/NEGATIVE, trivial tasks score STRONG). |
| **Estimated effort** | 2–3 days |

---

### TASK-15: Bonus — Algorithm Discovery

| Field | Value |
|---|---|
| **Runs** | EXP-B1, EXP-B2 |
| **Depends on** | TASK-14 (needs STRONG solvability results to select candidates) |
| **Deliverables** | Extracted rules/programs, functional equivalence test results |
| **Acceptance criteria** | At least one extracted rule matches reference algorithm on >99% of hard test samples. |
| **Estimated effort** | 2–3 days |
| **Note** | Optional. Only attempt if TASK-14 identifies tasks with STRONG verdicts. |

---

## Effort Summary

| Phase | Tasks | Scope | Estimated Effort |
|---|---|---|---|
| Foundation | TASK-01 → TASK-04 | Schemas + DSLs + Registry | 3–5 days |
| Data pipeline | TASK-05 → TASK-06 | Data generation + Splits | 2 days |
| Model pipeline | TASK-07 → TASK-08 | Training + Evaluation | 3–4 days |
| Orchestration | TASK-09 → TASK-10 | Runner + Reports | 2 days |
| Validation gate | TASK-11 | Smoke tests + V-Global | 0.5 day |
| Core experiments | TASK-12 + TASK-13 | S-track + C-track (parallel) | 3–4 days |
| Analysis | TASK-14 | Diagnostics | 2–3 days |
| Bonus | TASK-15 | Algorithm discovery | 2–3 days |

**Total: ~18–24 working days.** First validated pipeline (through TASK-11): ~11–14 days.

---

## Implementation Rules

1. **One task at a time.** Do not start a task until all its dependencies are complete and validated.
2. **Validate immediately.** Run the corresponding V-check as soon as a shared resource is built. Do not defer.
3. **Gate before experiments.** TASK-11 is a hard gate. Fix any failures before running core experiments.
4. **Log deviations.** Any change to a task's scope, interface, dependencies, or acceptance criteria must be logged in Part 5 (Deviation Log).
5. **Suggested file paths are suggestions.** The directory structure may be adjusted during implementation — log the actual structure in the deviation log.

---

# PART 5: DEVIATION LOG

This section tracks every change from the original plan that occurs during implementation. It serves as an audit trail and ensures future chats can understand what was actually built vs. what was planned.

## How to Use This Log

When implementing a task, if **any** of the following occur, add an entry:

- A task's scope changed (added/removed deliverables)
- An interface changed from what's specified in Part 1
- A dependency was added or removed
- A task was split into sub-tasks
- A task was deferred or skipped
- An acceptance criterion was relaxed or tightened
- A new shared resource was added
- A bug was found that required architectural change
- An experiment was modified, added, or dropped

## Log Format

Each entry follows this template:

```markdown
### DEV-{NNN}: {Short title}

- **Date:** YYYY-MM-DD
- **Task:** TASK-XX (or SR-XX, EXP-XX, V-XX)
- **Type:** SCOPE_CHANGE | INTERFACE_CHANGE | DEPENDENCY_CHANGE | TASK_SPLIT | DEFERRAL | CRITERIA_CHANGE | NEW_RESOURCE | BUG_FIX | EXPERIMENT_CHANGE
- **What changed:** {description of the deviation}
- **Why:** {rationale}
- **Impact:** {which other tasks/experiments are affected}
- **Resolution:** {what was actually done}
```

## Log Entries

### DEV-001: Added Distribution and ElementType enums to SR-2

- **Date:** 2025-03-25
- **Task:** TASK-01 (SR-2)
- **Type:** SCOPE_CHANGE
- **What changed:** Added `Distribution` enum (`UNIFORM`, `NORMAL`, `EXPONENTIAL`, `WEIGHTED`) and `ElementType` enum (`INT`, `BINARY`, `CHAR`) as proper Python Enum classes. The original SR-2 spec used raw strings for these fields.
- **Why:** Type safety — prevents typos, enables IDE autocomplete, and makes invalid states unrepresentable.
- **Impact:** All downstream modules constructing schemas must import and use these enums instead of raw strings.
- **Resolution:** Enums are defined in `src/schemas.py` alongside the schema classes.

---

### DEV-002: Tuples instead of lists for frozen dataclass compatibility

- **Date:** 2025-03-25
- **Task:** TASK-01 (SR-2)
- **Type:** INTERFACE_CHANGE
- **What changed:** SR-2 spec shows `list[str]` for `CategoricalFeatureSpec.values` and `list[float]` for `weights`. Implementation uses `Tuple[str, ...]` and `Tuple[float, ...]`. Similarly, `TabularInputSchema` feature collections use tuples.
- **Why:** Frozen dataclasses require all fields to be hashable. Lists are not hashable; tuples are.
- **Impact:** All callers must pass tuples when constructing specs. E.g. `values=("a", "b")` not `values=["a", "b"]`.
- **Resolution:** Documented in ADR-006 and TASK-01 log. Minor ergonomic cost, significant safety gain.

---

### DEV-003: Sequence DSL reducers return list[int] instead of int

- **Date:** 2025-03-25
- **Task:** TASK-03 (SR-10)
- **Type:** INTERFACE_CHANGE
- **What changed:** SR-10 spec implies reducers (Sum, Count, Max, Min, Parity) produce a scalar `int`. Implementation wraps output in `[int]` for uniform composability.
- **Why:** All operations being `list[int] → list[int]` eliminates type-mismatch errors in `Compose` and simplifies the type system.
- **Impact:** Downstream consumers must unwrap `[result]` when a scalar is needed. Documented in ADR-010.
- **Resolution:** Accepted as a design improvement. The minor unwrapping cost is worth the type-safety gain.

---

### DEV-004: S4/S5/C4/C5 task tiers deferred

- **Date:** 2025-03-25
- **Task:** TASK-04 (SR-1)
- **Type:** DEFERRAL
- **What changed:** Only S0–S3 and C0–C3 task tiers are registered in the initial registry (28 tasks). S4 (structural/graph), S5 (DSL programs), C4 (stateful aggregation), and C5 (multi-step compositional) are deferred.
- **Why:** Higher-tier tasks require more complex reference implementations. The initial 28 tasks cover all tiers needed for the pipeline (TASK-05 through TASK-11).
- **Impact:** TASK-12/13 (experiment tasks) will need to add higher-tier tasks to the registry.
- **Resolution:** Higher tiers will be added incrementally when experiment tasks are implemented.

---

### DEV-005: DistractorSplit defined but not implemented

- **Date:** 2025-03-25
- **Task:** TASK-06 (SR-4)
- **Type:** DEFERRAL
- **What changed:** The `DISTRACTOR` split strategy is defined in the `SplitStrategy` enum but has no implementation.
- **Why:** Requires `TabularInputSchema.with_extra_irrelevant()` method. Needed only for specific experiments (EXP-D2).
- **Impact:** No impact on pipeline development. Will implement when EXP-D2 is run.
- **Resolution:** Enum value reserved for forward compatibility. Implementation deferred.

---

### DEV-006: Single harness.py instead of planned 3-file split

- **Date:** 2025-03-25
- **Task:** TASK-07 (SR-5)
- **Type:** SCOPE_CHANGE
- **What changed:** The planned file structure had `src/models/harness.py`, `src/models/configs.py`, and `src/models/encoders.py`. All code is in `harness.py` (459 lines).
- **Why:** The total code volume doesn't justify 3 files. Splitting would create unnecessary import complexity.
- **Impact:** All imports come from `src.models.harness`. No API impact. Will refactor if the file grows significantly.
- **Resolution:** Accepted. Single file is cleaner at this scale.

---

## Decision Record

Major architectural or design decisions made during implementation that are not captured in the original plan.

### Format

```markdown
### DEC-{NNN}: {Short title}

- **Date:** YYYY-MM-DD
- **Context:** {what prompted this decision}
- **Options considered:** {list of alternatives}
- **Decision:** {what was chosen}
- **Rationale:** {why}
- **Consequences:** {what this means for the rest of the project}
```

### Entries

_Decisions are logged in `docs/ARCHITECTURE_DECISIONS.md` as ADR-001 through ADR-013._

---

# PART 6: APPENDICES

## Appendix A: Quick Reference — Which Shared Resources Does Each Experiment Use?

| Experiment | SR-1 | SR-2 | SR-3 | SR-4 | SR-5 | SR-6 | SR-7 | SR-8 | SR-9 | SR-10 |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-0.1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| EXP-0.2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-0.3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| EXP-S1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| EXP-S2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| EXP-S3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| EXP-S4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| EXP-S5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| EXP-C1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-C2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-C3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-C4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-C5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-D1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| EXP-D2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-D3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | |
| EXP-D4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-D5 | | | | | | ✓ | | ✓ | | |
| EXP-B1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| EXP-B2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |

---

## Appendix B: Validation Checklist (Printable)

Use this checklist before trusting any experiment results.

```
[ ] V-1:  Task registry — all tasks registered, deterministic, verifier works
[ ] V-2:  Input schemas — types, ranges, reproducibility, distributions checked
[ ] V-3:  Data generator — 100% label correctness, deterministic, metadata correct
[ ] V-4:  Split generator — disjoint, complete, properties hold
[ ] V-5:  Model harness — overfit test, random label test, known-solution test, reproducible
[ ] V-6:  Evaluation engine — perfect/worst/known confusion tests pass
[ ] V-7:  Experiment runner — end-to-end artifacts exist and are correct
[ ] V-8:  Report generator — files exist, JSON valid, metrics consistent
[ ] V-9:  Classification DSL — type safe, deterministic, hand-verified rules match
[ ] V-10: Sequence DSL — type safe, deterministic, hand-verified programs match
[ ] V-G1: Round-trip check passes
[ ] V-G2: Control tasks produce WEAK/NEGATIVE verdicts
[ ] V-G3: Trivial tasks produce STRONG verdicts
[ ] V-G4: No cross-task data contamination
```

---

## Appendix C: Task-to-Experiment Traceability

| Task ID | Builds/Runs | Depends On | Milestone Gate |
|---|---|---|---|
| TASK-01 | SR-2 | — | |
| TASK-02 | SR-9 | TASK-01 | |
| TASK-03 | SR-10 | TASK-01 | |
| TASK-04 | SR-1 | TASK-01, 02, 03 | ✓ FOUNDATION |
| TASK-05 | SR-3 | TASK-04 | |
| TASK-06 | SR-4 | TASK-05 | ✓ DATA PIPELINE |
| TASK-07 | SR-5 | TASK-01, 05 | |
| TASK-08 | SR-6 | TASK-07 | |
| TASK-09 | SR-7 | TASK-04–08 | |
| TASK-10 | SR-8 | TASK-09 | ✓ FULL PIPELINE |
| TASK-11 | EXP-0.x | TASK-10 | ✓ SMOKE TEST GATE |
| TASK-12 | EXP-S1–S5 | TASK-11 | |
| TASK-13 | EXP-C1–C5 | TASK-11 | |
| TASK-14 | EXP-D1–D5 | TASK-12, 13 | |
| TASK-15 | EXP-B1–B2 | TASK-14 | |

---

## Appendix D: How to Use These Documents in New Chats

When starting a new implementation chat, provide the following context:

1. **Always reference both documents:**
   - `docs/EXPERIMENT_DESIGN.md` — the *what* and *why* (design rationale, task definitions, metrics, evidence criteria)
   - `docs/EXPERIMENT_CATALOG.md` — the *how* and *when* (shared resources, experiments, validation, execution plan, deviation log)

2. **Specify which TASK-XX you are implementing.** This scopes the chat to a single deliverable with clear inputs, outputs, and acceptance criteria.

3. **Check the Deviation Log (Part 5) before starting.** Earlier chats may have changed interfaces or added dependencies that affect your task.

4. **After completing a task, update the Deviation Log** if anything differed from the plan. This ensures the next chat has accurate context.

5. **At milestone gates**, review the validation checklist (Appendix B) and confirm all relevant checks pass before proceeding.

### Example prompt for a new implementation chat

```
I am implementing TASK-05 (Data Generator) for the algorithmic solvability experiment.

Reference documents:
- docs/EXPERIMENT_DESIGN.md (design rationale, task tiers, data generation strategy in Section 7)
- docs/EXPERIMENT_CATALOG.md (SR-3 specification in Part 1, validation V-3 in Part 3, 
  TASK-05 details in Part 4)

Dependencies already complete: TASK-01 through TASK-04.
Check Part 5 (Deviation Log) for any changes from the original plan.

Deliverable: generate_dataset() function per the SR-3 interface.
Acceptance: V-3 validation passes (100% label correctness, determinism, metadata).
```
