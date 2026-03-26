"""SR-7: Experiment Runner.

Orchestrator that takes an experiment specification and runs the full pipeline:
generate data → split → train models → evaluate → collect results.

Multi-seed execution with aggregation (mean ± std) across seeds.

Used by: All experiments (EXP-0.x through EXP-B).
Validated by: V-7 (Experiment Runner Validation).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data_generator import generate_dataset
from src.evaluation import EvalReport, evaluate, eval_report_to_dict
from src.models.harness import ModelConfig, ModelHarness
from src.registry import TaskRegistry, TaskSpec, build_default_registry
from src.splits import (
    SplitResult,
    SplitStrategy,
    split_iid,
    split_length,
    split_noise,
    split_value,
)

logger = logging.getLogger(__name__)


# ===================================================================
# ExperimentSpec dataclass
# ===================================================================

@dataclass
class ExperimentSpec:
    """Specification for a single experiment run.

    Attributes:
        experiment_id: Unique identifier, e.g. "EXP-0.1".
        task_ids: List of task IDs to evaluate.
        model_configs: List of model configurations to test.
        split_strategies: List of split strategies to use.
        n_samples: Total number of samples to generate per task.
        train_fraction: Fraction of data for training (IID and noise splits).
        seeds: List of random seeds for multi-seed runs.
        noise_level: Noise level for noise splits (if NOISE in split_strategies).
        length_threshold: Length threshold for length extrapolation splits.
        value_feature: Feature name for value extrapolation splits.
        value_train_range: (lo, hi) for value extrapolation splits.
    """
    experiment_id: str
    task_ids: List[str]
    model_configs: List[ModelConfig]
    split_strategies: List[SplitStrategy]
    n_samples: int = 500
    train_fraction: float = 0.8
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    noise_level: float = 0.1
    length_threshold: Optional[int] = None
    value_feature: Optional[str] = None
    value_train_range: Optional[Tuple[float, float]] = None


# ===================================================================
# Result dataclasses
# ===================================================================

@dataclass
class SingleRunResult:
    """Result of one (task, model, split, seed) combination.

    Attributes:
        task_id: Task ID.
        model_name: Model family name.
        split_strategy: Split strategy used.
        seed: Random seed.
        eval_report: The full evaluation report.
        train_size: Number of training samples.
        test_size: Number of test samples.
        train_time_seconds: Time to train the model.
        split_metadata: Split parameters used for this run.
    """
    task_id: str
    model_name: str
    split_strategy: str
    seed: int
    eval_report: EvalReport
    train_size: int
    test_size: int
    train_time_seconds: float
    split_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedResult:
    """Aggregated metrics across multiple seeds for one (task, model, split).

    Attributes:
        task_id: Task ID.
        model_name: Model family name.
        split_strategy: Split strategy used.
        n_seeds: Number of seeds.
        accuracy_mean: Mean accuracy across seeds.
        accuracy_std: Std of accuracy across seeds.
        per_seed_accuracy: List of per-seed accuracies.
        extra_metrics: Additional aggregated metrics.
    """
    task_id: str
    model_name: str
    split_strategy: str
    n_seeds: int
    accuracy_mean: float
    accuracy_std: float
    per_seed_accuracy: List[float]
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentReport:
    """Full report for an experiment run.

    Attributes:
        experiment_id: Experiment ID.
        spec: The experiment specification.
        seeds_used: The concrete seed list used for execution.
        single_results: All individual (task, model, split, seed) results.
        aggregated_results: Aggregated results across seeds.
        total_time_seconds: Total wall-clock time for the experiment.
    """
    experiment_id: str
    spec: ExperimentSpec
    seeds_used: List[int]
    single_results: List[SingleRunResult]
    aggregated_results: List[AggregatedResult]
    total_time_seconds: float


# ===================================================================
# Split dispatch
# ===================================================================

def _apply_split(
    dataset: Any,
    task: TaskSpec,
    strategy: SplitStrategy,
    spec: ExperimentSpec,
    seed: int,
) -> SplitResult:
    """Apply a split strategy to a dataset.

    Args:
        dataset: The generated dataset.
        task: Task metadata used to choose split-specific behavior such as
            schema-guided categorical noise.
        strategy: Which split strategy to use.
        spec: Experiment spec (for split parameters).
        seed: Random seed.

    Returns:
        A SplitResult.
    """
    if strategy == SplitStrategy.IID:
        return split_iid(dataset, train_fraction=spec.train_fraction, seed=seed)
    elif strategy == SplitStrategy.NOISE:
        return split_noise(
            dataset,
            train_fraction=spec.train_fraction,
            test_noise_level=spec.noise_level,
            seed=seed,
            schema=task.input_schema,
        )
    elif strategy == SplitStrategy.LENGTH_EXTRAPOLATION:
        if spec.length_threshold is None:
            raise ValueError(
                "length_threshold must be set in ExperimentSpec for "
                "LENGTH_EXTRAPOLATION split strategy"
            )
        return split_length(dataset, length_threshold=spec.length_threshold)
    elif strategy == SplitStrategy.VALUE_EXTRAPOLATION:
        if spec.value_feature is None or spec.value_train_range is None:
            raise ValueError(
                "value_feature and value_train_range must be set in ExperimentSpec "
                "for VALUE_EXTRAPOLATION split strategy"
            )
        return split_value(
            dataset,
            feature_name=spec.value_feature,
            train_range=spec.value_train_range,
        )
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")


# ===================================================================
# Core runner logic
# ===================================================================

def _run_single(
    task: TaskSpec,
    model_config: ModelConfig,
    split: SplitResult,
    split_strategy: SplitStrategy,
    seed: int,
) -> SingleRunResult:
    """Run a single (task, model, split, seed) combination.

    Returns:
        A SingleRunResult.
    """
    harness = ModelHarness(model_config)

    start = time.perf_counter()
    pred_result = harness.run(
        train_inputs=split.train_inputs,
        train_outputs=split.train_outputs,
        test_inputs=split.test_inputs,
        test_outputs=split.test_outputs,
    )
    train_time = time.perf_counter() - start

    report = evaluate(
        predictions=pred_result.predictions,
        ground_truth=pred_result.true_labels,
        task=task,
        split_name=f"{split_strategy.value}_seed{seed}",
    )

    return SingleRunResult(
        task_id=task.task_id,
        model_name=pred_result.model_name,
        split_strategy=split_strategy.value,
        seed=seed,
        eval_report=report,
        train_size=split.train_size,
        test_size=split.test_size,
        train_time_seconds=train_time,
        split_metadata=dict(split.split_metadata),
    )


def _aggregate_results(
    results: List[SingleRunResult],
) -> List[AggregatedResult]:
    """Aggregate single-run results across seeds.

    Groups by (task_id, model_name, split_strategy) and computes
    mean ± std of accuracy.
    """
    groups: Dict[Tuple[str, str, str], List[SingleRunResult]] = {}
    for r in results:
        key = (r.task_id, r.model_name, r.split_strategy)
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    aggregated: List[AggregatedResult] = []
    for (task_id, model_name, split_strategy), group in sorted(groups.items()):
        accuracies = [r.eval_report.accuracy for r in group]
        acc_arr = np.array(accuracies)

        extra: Dict[str, Any] = {}

        # Aggregate classification-specific metrics
        macro_f1s = [r.eval_report.macro_f1 for r in group if r.eval_report.macro_f1 is not None]
        if macro_f1s:
            extra["macro_f1_mean"] = float(np.mean(macro_f1s))
            extra["macro_f1_std"] = float(np.std(macro_f1s))

        # Aggregate sequence-specific metrics
        exact_matches = [r.eval_report.exact_match for r in group if r.eval_report.exact_match is not None]
        if exact_matches:
            extra["exact_match_mean"] = float(np.mean(exact_matches))
            extra["exact_match_std"] = float(np.std(exact_matches))

        token_accs = [r.eval_report.token_accuracy for r in group if r.eval_report.token_accuracy is not None]
        if token_accs:
            extra["token_accuracy_mean"] = float(np.mean(token_accs))
            extra["token_accuracy_std"] = float(np.std(token_accs))

        aggregated.append(AggregatedResult(
            task_id=task_id,
            model_name=model_name,
            split_strategy=split_strategy,
            n_seeds=len(group),
            accuracy_mean=float(np.mean(acc_arr)),
            accuracy_std=float(np.std(acc_arr)),
            per_seed_accuracy=accuracies,
            extra_metrics=extra,
        ))

    return aggregated


# ===================================================================
# Main run_experiment function
# ===================================================================

def run_experiment(
    spec: ExperimentSpec,
    seeds: Optional[List[int]] = None,
    registry: Optional[TaskRegistry] = None,
) -> ExperimentReport:
    """Run a full experiment as specified.

    Pipeline per (task, seed, split, model):
    1. Generate dataset for task with seed
    2. Apply split strategy
    3. Train model on train split
    4. Evaluate on test split
    5. Collect results

    After all runs, aggregate across seeds.

    Args:
        spec: The experiment specification.
        seeds: Optional override for the seed list used at execution time.
        registry: Task registry to look up tasks. If None, builds default.

    Returns:
        An ExperimentReport with all single and aggregated results.
    """
    if registry is None:
        registry = build_default_registry()

    active_seeds = list(spec.seeds if seeds is None else seeds)
    start_time = time.perf_counter()
    all_results: List[SingleRunResult] = []

    total_runs = (
        len(spec.task_ids)
        * len(active_seeds)
        * len(spec.split_strategies)
        * len(spec.model_configs)
    )
    run_count = 0

    for task_id in spec.task_ids:
        task = registry.get(task_id)

        for seed in active_seeds:
            dataset = generate_dataset(task, n_samples=spec.n_samples, base_seed=seed)

            for strategy in spec.split_strategies:
                try:
                    split = _apply_split(dataset, task, strategy, spec, seed)
                except (ValueError, TypeError) as e:
                    if str(e).startswith("Unknown split strategy:"):
                        raise
                    logger.warning(
                        "Skipping split %s for task %s seed %d: %s",
                        strategy.value, task_id, seed, e,
                    )
                    run_count += len(spec.model_configs)
                    continue

                if split.train_size == 0 or split.test_size == 0:
                    logger.warning(
                        "Empty train or test set for %s/%s/seed=%d. Skipping.",
                        task_id, strategy.value, seed,
                    )
                    run_count += len(spec.model_configs)
                    continue

                for model_config in spec.model_configs:
                    run_count += 1
                    logger.info(
                        "[%d/%d] %s | %s | %s | seed=%d",
                        run_count, total_runs,
                        task_id, model_config.family.value,
                        strategy.value, seed,
                    )

                    try:
                        result = _run_single(
                            task, model_config, split, strategy, seed,
                        )
                        all_results.append(result)
                    except Exception as e:
                        logger.error(
                            "Error running %s/%s/%s/seed=%d: %s",
                            task_id, model_config.family.value,
                            strategy.value, seed, e,
                        )

    total_time = time.perf_counter() - start_time
    aggregated = _aggregate_results(all_results)

    return ExperimentReport(
        experiment_id=spec.experiment_id,
        spec=spec,
        seeds_used=active_seeds,
        single_results=all_results,
        aggregated_results=aggregated,
        total_time_seconds=total_time,
    )


# ===================================================================
# Utility: report to dict (for JSON serialization)
# ===================================================================

def single_result_to_dict(result: SingleRunResult) -> Dict[str, Any]:
    """Convert a SingleRunResult to a JSON-serializable dict."""
    return {
        "task_id": result.task_id,
        "model_name": result.model_name,
        "split_strategy": result.split_strategy,
        "seed": result.seed,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "train_time_seconds": result.train_time_seconds,
        "split_metadata": result.split_metadata,
        "eval_report": eval_report_to_dict(result.eval_report),
    }


def aggregated_result_to_dict(result: AggregatedResult) -> Dict[str, Any]:
    """Convert an AggregatedResult to a JSON-serializable dict."""
    return {
        "task_id": result.task_id,
        "model_name": result.model_name,
        "split_strategy": result.split_strategy,
        "n_seeds": result.n_seeds,
        "accuracy_mean": result.accuracy_mean,
        "accuracy_std": result.accuracy_std,
        "per_seed_accuracy": result.per_seed_accuracy,
        "extra_metrics": result.extra_metrics,
    }


def experiment_report_to_dict(report: ExperimentReport) -> Dict[str, Any]:
    """Convert an ExperimentReport to a JSON-serializable dict."""
    return {
        "experiment_id": report.experiment_id,
        "spec": {
            "experiment_id": report.spec.experiment_id,
            "task_ids": report.spec.task_ids,
            "model_configs": [
                {
                    "family": mc.family.value,
                    "name": mc.name,
                    "hyperparams": mc.hyperparams,
                }
                for mc in report.spec.model_configs
            ],
            "split_strategies": [s.value for s in report.spec.split_strategies],
            "n_samples": report.spec.n_samples,
            "train_fraction": report.spec.train_fraction,
            "seeds": report.spec.seeds,
            "noise_level": report.spec.noise_level,
            "length_threshold": report.spec.length_threshold,
            "value_feature": report.spec.value_feature,
            "value_train_range": report.spec.value_train_range,
        },
        "seeds_used": report.seeds_used,
        "total_time_seconds": report.total_time_seconds,
        "n_single_results": len(report.single_results),
        "n_aggregated_results": len(report.aggregated_results),
        "single_results": [
            single_result_to_dict(r) for r in report.single_results
        ],
        "aggregated_results": [
            aggregated_result_to_dict(a) for a in report.aggregated_results
        ],
    }
