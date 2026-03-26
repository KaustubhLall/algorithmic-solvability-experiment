"""TASK-13 classification experiment definitions and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from src.models.harness import ModelConfig, ModelFamily
from src.registry import TaskRegistry, build_default_registry
from src.reporting import generate_report
from src.runner import ExperimentReport, ExperimentSpec, run_experiment
from src.splits import SplitStrategy


TASK13_CLASSIFICATION_SEEDS = [42, 123, 456, 789, 1024]
TASK13_CLASSIFICATION_TRAIN_FRACTION = 0.7
TASK13_CLASSIFICATION_NOISE_LEVEL = 0.1
TASK13_CLASSIFICATION_N_SAMPLES = {
    "EXP-C1": 900,
    "EXP-C2": 900,
    "EXP-C3": 1000,
}
TASK13_VALUE_FEATURES = {
    "EXP-C1": "x1",
    "EXP-C2": "x1",
    "EXP-C3": "x1",
}
TASK13_VALUE_TRAIN_RANGES = {
    "EXP-C1": (20.0, 80.0),
    "EXP-C2": (20.0, 80.0),
    "EXP-C3": (20.0, 80.0),
}
TASK13_SPLIT_STRATEGIES = {
    "EXP-C1": [
        SplitStrategy.IID,
        SplitStrategy.VALUE_EXTRAPOLATION,
        SplitStrategy.NOISE,
    ],
    "EXP-C2": [
        SplitStrategy.IID,
        SplitStrategy.VALUE_EXTRAPOLATION,
        SplitStrategy.NOISE,
    ],
    "EXP-C3": [
        SplitStrategy.IID,
        SplitStrategy.VALUE_EXTRAPOLATION,
        SplitStrategy.NOISE,
    ],
}


@dataclass(frozen=True)
class ClassificationExperimentArtifact:
    """Report and artifact directory for a TASK-13 classification experiment."""

    spec: ExperimentSpec
    report: ExperimentReport
    output_dir: Path


def _default_classification_model_configs() -> List[ModelConfig]:
    """Return the validated classification-task model families available today."""

    return [
        ModelConfig(family=ModelFamily.MAJORITY_CLASS),
        ModelConfig(family=ModelFamily.LOGISTIC_REGRESSION),
        ModelConfig(family=ModelFamily.DECISION_TREE),
        ModelConfig(
            family=ModelFamily.RANDOM_FOREST,
            hyperparams={"n_estimators": 200, "max_depth": 12},
        ),
        ModelConfig(
            family=ModelFamily.GRADIENT_BOOSTED_TREES,
            hyperparams={"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05},
        ),
        ModelConfig(
            family=ModelFamily.MLP,
            hyperparams={"hidden_layer_sizes": (128, 64), "max_iter": 400},
        ),
    ]


def _task_ids_by_tier(registry: TaskRegistry, tier: str) -> List[str]:
    return sorted(
        task.task_id for task in registry.by_tier(tier) if task.track == "classification"
    )


def build_classification_experiment_specs(
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, ExperimentSpec]:
    """Return the TASK-13 classification experiment specifications.

    This task runs the currently implemented classification benchmark tiers (C1-C3).
    C4/C5 remain deferred until their registry tasks and split support land.
    """

    active_registry = build_default_registry() if registry is None else registry
    models = _default_classification_model_configs()

    experiment_tasks: Mapping[str, List[str]] = {
        "EXP-C1": _task_ids_by_tier(active_registry, "C1"),
        "EXP-C2": _task_ids_by_tier(active_registry, "C2"),
        "EXP-C3": _task_ids_by_tier(active_registry, "C3"),
    }

    return {
        experiment_id: ExperimentSpec(
            experiment_id=experiment_id,
            task_ids=task_ids,
            model_configs=models,
            split_strategies=list(TASK13_SPLIT_STRATEGIES[experiment_id]),
            n_samples=TASK13_CLASSIFICATION_N_SAMPLES[experiment_id],
            train_fraction=TASK13_CLASSIFICATION_TRAIN_FRACTION,
            seeds=list(TASK13_CLASSIFICATION_SEEDS),
            noise_level=TASK13_CLASSIFICATION_NOISE_LEVEL,
            value_feature=TASK13_VALUE_FEATURES[experiment_id],
            value_train_range=TASK13_VALUE_TRAIN_RANGES[experiment_id],
        )
        for experiment_id, task_ids in experiment_tasks.items()
        if task_ids
    }


def run_classification_experiment(
    spec: ExperimentSpec,
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> ClassificationExperimentArtifact:
    """Run one classification experiment and write report artifacts."""

    active_registry = build_default_registry() if registry is None else registry
    report = run_experiment(spec, registry=active_registry)
    output_dir = generate_report(report, output_root=output_root, registry=active_registry)
    return ClassificationExperimentArtifact(spec=spec, report=report, output_dir=output_dir)


def run_all_classification_experiments(
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, ClassificationExperimentArtifact]:
    """Run the full TASK-13 classification suite for the implemented tiers."""

    active_registry = build_default_registry() if registry is None else registry
    artifacts: Dict[str, ClassificationExperimentArtifact] = {}
    for experiment_id, spec in build_classification_experiment_specs(active_registry).items():
        artifacts[experiment_id] = run_classification_experiment(
            spec,
            output_root=output_root,
            registry=active_registry,
        )
    return artifacts
