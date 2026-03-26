"""TASK-11 smoke experiment definitions and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.models.harness import ModelConfig, ModelFamily
from src.registry import TaskRegistry, TaskSpec, build_default_registry
from src.reporting import generate_report
from src.runner import ExperimentReport, ExperimentSpec, run_experiment
from src.schemas import SequenceInputSchema
from src.splits import SplitStrategy


SMOKE_SAMPLE_COUNT = 1200
SMOKE_TRAIN_FRACTION = 1000 / 1200
SMOKE_SEQUENCE_LENGTH_RANGE = (4, 8)
SMOKE_CLASSIFICATION_SEEDS = [42, 43, 44, 45, 46]


@dataclass(frozen=True)
class SmokeExperimentArtifact:
    """Report and artifact directory for a smoke experiment."""

    spec: ExperimentSpec
    report: ExperimentReport
    output_dir: Path


def build_smoke_specs() -> Dict[str, ExperimentSpec]:
    """Return the TASK-11 smoke experiment specifications."""

    return {
        "EXP-0.1": ExperimentSpec(
            experiment_id="EXP-0.1",
            task_ids=["S1.2_sort"],
            model_configs=[
                ModelConfig(family=ModelFamily.MLP),
                ModelConfig(
                    family=ModelFamily.LSTM,
                    hyperparams={
                        "epochs": 100,
                        "hidden_size": 64,
                        "embedding_dim": 32,
                        "batch_size": 64,
                        "learning_rate": 0.01,
                    },
                ),
            ],
            split_strategies=[SplitStrategy.IID],
            n_samples=SMOKE_SAMPLE_COUNT,
            train_fraction=SMOKE_TRAIN_FRACTION,
            seeds=[42],
        ),
        "EXP-0.2": ExperimentSpec(
            experiment_id="EXP-0.2",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[
                ModelConfig(family=ModelFamily.MAJORITY_CLASS),
                ModelConfig(family=ModelFamily.LOGISTIC_REGRESSION),
                ModelConfig(family=ModelFamily.DECISION_TREE),
                ModelConfig(family=ModelFamily.MLP),
            ],
            split_strategies=[SplitStrategy.IID, SplitStrategy.NOISE],
            n_samples=SMOKE_SAMPLE_COUNT,
            train_fraction=SMOKE_TRAIN_FRACTION,
            seeds=list(SMOKE_CLASSIFICATION_SEEDS),
            noise_level=0.05,
        ),
        "EXP-0.3": ExperimentSpec(
            experiment_id="EXP-0.3",
            task_ids=["S0.1_random_labels", "C0.1_random_class"],
            model_configs=[ModelConfig(family=ModelFamily.MLP)],
            split_strategies=[SplitStrategy.IID],
            n_samples=SMOKE_SAMPLE_COUNT,
            train_fraction=SMOKE_TRAIN_FRACTION,
            seeds=[42],
        ),
    }


def build_sequence_smoke_registry(
    registry: Optional[TaskRegistry] = None,
) -> TaskRegistry:
    """Return a registry containing the bounded sort task for EXP-0.1."""

    base_registry = build_default_registry() if registry is None else registry
    sort_task = base_registry.get("S1.2_sort")
    bounded_sort = _clone_sequence_task_with_bounds(
        sort_task,
        min_length=SMOKE_SEQUENCE_LENGTH_RANGE[0],
        max_length=SMOKE_SEQUENCE_LENGTH_RANGE[1],
    )

    smoke_registry = TaskRegistry()
    smoke_registry.register(bounded_sort)
    return smoke_registry


def run_smoke_experiment(
    spec: ExperimentSpec,
    output_root: str | Path = "results",
) -> SmokeExperimentArtifact:
    """Run one smoke experiment and write report artifacts."""

    if spec.experiment_id == "EXP-0.1":
        registry = build_sequence_smoke_registry()
    else:
        registry = build_default_registry()

    report = run_experiment(spec, registry=registry)
    output_dir = generate_report(report, output_root=output_root, registry=registry)
    return SmokeExperimentArtifact(spec=spec, report=report, output_dir=output_dir)


def run_all_smoke_experiments(
    output_root: str | Path = "results",
) -> Dict[str, SmokeExperimentArtifact]:
    """Run the full TASK-11 smoke suite."""

    artifacts: Dict[str, SmokeExperimentArtifact] = {}
    for experiment_id, spec in build_smoke_specs().items():
        artifacts[experiment_id] = run_smoke_experiment(spec, output_root=output_root)
    return artifacts


def _clone_sequence_task_with_bounds(
    task: TaskSpec,
    min_length: int,
    max_length: int,
) -> TaskSpec:
    schema = task.input_schema
    if not isinstance(schema, SequenceInputSchema):
        raise ValueError(f"Task {task.task_id} is not a sequence task")

    bounded_schema = SequenceInputSchema(
        element_type=schema.element_type,
        min_length=min_length,
        max_length=max_length,
        value_range=schema.value_range,
    )
    return TaskSpec(
        task_id=task.task_id,
        tier=task.tier,
        track=task.track,
        description=task.description,
        input_schema=bounded_schema,
        output_type=task.output_type,
        n_classes=task.n_classes,
        reference_algorithm=task.reference_algorithm,
        input_sampler=lambda seed: bounded_schema.sample(seed),
        verifier=task.verifier,
        complexity_metadata=dict(task.complexity_metadata),
    )
