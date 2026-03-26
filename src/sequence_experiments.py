"""TASK-12 sequence experiment definitions and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from src.models.harness import ModelConfig, ModelFamily
from src.registry import TaskRegistry, build_default_registry
from src.reporting import generate_report
from src.runner import ExperimentReport, ExperimentSpec, run_experiment
from src.splits import SplitStrategy


TASK12_SEQUENCE_SEEDS = [42, 123, 456]
TASK12_SEQUENCE_TRAIN_FRACTION = 0.7
TASK12_SEQUENCE_N_SAMPLES = {
    "EXP-S1": 600,
    "EXP-S2": 600,
    "EXP-S3": 500,
}
TASK12_LENGTH_THRESHOLDS = {
    "EXP-S1": 8,
    "EXP-S2": 10,
    "EXP-S3": 8,
}
TASK12_VALUE_TRAIN_RANGES = {
    "EXP-S1": (0.0, 6.0),
    "EXP-S3": (0.0, 6.0),
}
TASK12_SPLIT_STRATEGIES = {
    "EXP-S1": [
        SplitStrategy.IID,
        SplitStrategy.LENGTH_EXTRAPOLATION,
        SplitStrategy.VALUE_EXTRAPOLATION,
    ],
    "EXP-S2": [
        SplitStrategy.IID,
        SplitStrategy.LENGTH_EXTRAPOLATION,
    ],
    "EXP-S3": [
        SplitStrategy.IID,
        SplitStrategy.LENGTH_EXTRAPOLATION,
        SplitStrategy.VALUE_EXTRAPOLATION,
    ],
}


@dataclass(frozen=True)
class SequenceExperimentArtifact:
    """Report and artifact directory for a TASK-12 sequence experiment."""

    spec: ExperimentSpec
    report: ExperimentReport
    output_dir: Path


def _default_sequence_model_configs() -> List[ModelConfig]:
    """Return the validated sequence-task model families available today."""

    return [
        ModelConfig(family=ModelFamily.MAJORITY_CLASS),
        ModelConfig(family=ModelFamily.SEQUENCE_BASELINE),
        ModelConfig(
            family=ModelFamily.MLP,
            hyperparams={"hidden_layer_sizes": (64, 32), "max_iter": 300},
        ),
        ModelConfig(
            family=ModelFamily.LSTM,
            hyperparams={
                "epochs": 20,
                "hidden_size": 64,
                "embedding_dim": 32,
                "batch_size": 64,
                "learning_rate": 0.01,
            },
        ),
    ]


def _task_ids_by_tier(registry: TaskRegistry, tier: str) -> List[str]:
    return sorted(task.task_id for task in registry.by_tier(tier) if task.track == "sequence")


def build_sequence_experiment_specs(
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, ExperimentSpec]:
    """Return the TASK-12 sequence experiment specifications.

    This task runs the currently implemented sequence benchmark tiers (S1-S3).
    S4/S5 remain deferred until their registry tasks and split support land.
    """

    active_registry = build_default_registry() if registry is None else registry
    models = _default_sequence_model_configs()

    experiment_tasks: Mapping[str, List[str]] = {
        "EXP-S1": _task_ids_by_tier(active_registry, "S1"),
        "EXP-S2": _task_ids_by_tier(active_registry, "S2"),
        "EXP-S3": _task_ids_by_tier(active_registry, "S3"),
    }

    return {
        experiment_id: ExperimentSpec(
            experiment_id=experiment_id,
            task_ids=task_ids,
            model_configs=models,
            split_strategies=list(TASK12_SPLIT_STRATEGIES[experiment_id]),
            n_samples=TASK12_SEQUENCE_N_SAMPLES[experiment_id],
            train_fraction=TASK12_SEQUENCE_TRAIN_FRACTION,
            seeds=list(TASK12_SEQUENCE_SEEDS),
            length_threshold=TASK12_LENGTH_THRESHOLDS[experiment_id],
            value_feature="sequence_values",
            value_train_range=TASK12_VALUE_TRAIN_RANGES.get(experiment_id),
        )
        for experiment_id, task_ids in experiment_tasks.items()
        if task_ids
    }


def run_sequence_experiment(
    spec: ExperimentSpec,
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> SequenceExperimentArtifact:
    """Run one sequence experiment and write report artifacts."""

    active_registry = build_default_registry() if registry is None else registry
    report = run_experiment(spec, registry=active_registry)
    output_dir = generate_report(report, output_root=output_root, registry=active_registry)
    return SequenceExperimentArtifact(spec=spec, report=report, output_dir=output_dir)


def run_all_sequence_experiments(
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, SequenceExperimentArtifact]:
    """Run the full TASK-12 sequence suite for the implemented tiers."""

    active_registry = build_default_registry() if registry is None else registry
    artifacts: Dict[str, SequenceExperimentArtifact] = {}
    for experiment_id, spec in build_sequence_experiment_specs(active_registry).items():
        artifacts[experiment_id] = run_sequence_experiment(
            spec,
            output_root=output_root,
            registry=active_registry,
        )
    return artifacts
