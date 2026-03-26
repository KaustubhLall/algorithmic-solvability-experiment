"""TASK-13 classification experiment tests."""

from __future__ import annotations

import json

from src.classification_experiments import (
    TASK13_CLASSIFICATION_SEEDS,
    build_classification_experiment_specs,
    run_classification_experiment,
)
from src.registry import build_default_registry
from src.splits import SplitStrategy


def test_build_classification_experiment_specs_covers_supported_classification_tiers():
    registry = build_default_registry()

    specs = build_classification_experiment_specs(registry)

    assert set(specs) == {"EXP-C1", "EXP-C2", "EXP-C3"}
    assert specs["EXP-C1"].task_ids == sorted(
        task.task_id for task in registry.by_tier("C1") if task.track == "classification"
    )
    assert specs["EXP-C2"].task_ids == sorted(
        task.task_id for task in registry.by_tier("C2") if task.track == "classification"
    )
    assert specs["EXP-C3"].task_ids == sorted(
        task.task_id for task in registry.by_tier("C3") if task.track == "classification"
    )


def test_classification_experiment_specs_use_supported_models_and_splits():
    specs = build_classification_experiment_specs()
    spec = specs["EXP-C1"]

    assert [config.family.value for config in spec.model_configs] == [
        "majority_class",
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosted_trees",
        "mlp",
    ]
    assert spec.split_strategies == [
        SplitStrategy.IID,
        SplitStrategy.VALUE_EXTRAPOLATION,
        SplitStrategy.NOISE,
    ]
    assert spec.seeds == TASK13_CLASSIFICATION_SEEDS
    assert spec.value_feature == "x1"
    assert spec.value_train_range == (20.0, 80.0)


def test_run_classification_experiment_writes_artifacts(tmp_path):
    registry = build_default_registry()
    spec = build_classification_experiment_specs(registry)["EXP-C1"]
    spec.task_ids = ["C1.1_numeric_threshold"]
    spec.model_configs = spec.model_configs[:3]
    spec.split_strategies = [SplitStrategy.IID, SplitStrategy.NOISE]
    spec.n_samples = 120
    spec.seeds = [7]
    spec.noise_level = 0.2

    artifact = run_classification_experiment(spec, output_root=tmp_path, registry=registry)

    assert artifact.output_dir.exists()
    assert (artifact.output_dir / "config.json").exists()
    assert (artifact.output_dir / "summary.md").exists()
    assert (artifact.output_dir / "comparison.md").exists()
    assert (artifact.output_dir / "solvability_verdicts.json").exists()
    assert (
        artifact.output_dir
        / "per_task"
        / "C1.1_numeric_threshold"
        / "metrics.json"
    ).exists()

    verdicts = json.loads(
        (artifact.output_dir / "solvability_verdicts.json").read_text(encoding="utf-8")
    )
    assert "C1.1_numeric_threshold" in verdicts
