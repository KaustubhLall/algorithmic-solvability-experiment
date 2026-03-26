"""TASK-12 sequence experiment tests."""

from __future__ import annotations

import json

from src.registry import build_default_registry
from src.sequence_experiments import (
    TASK12_SEQUENCE_SEEDS,
    build_sequence_experiment_specs,
    run_sequence_experiment,
)
from src.splits import SplitStrategy


def test_build_sequence_experiment_specs_covers_supported_sequence_tiers():
    registry = build_default_registry()

    specs = build_sequence_experiment_specs(registry)

    assert set(specs) == {"EXP-S1", "EXP-S2", "EXP-S3"}
    assert specs["EXP-S1"].task_ids == sorted(
        task.task_id for task in registry.by_tier("S1") if task.track == "sequence"
    )
    assert specs["EXP-S2"].task_ids == sorted(
        task.task_id for task in registry.by_tier("S2") if task.track == "sequence"
    )
    assert specs["EXP-S3"].task_ids == sorted(
        task.task_id for task in registry.by_tier("S3") if task.track == "sequence"
    )


def test_sequence_experiment_specs_use_supported_models_and_splits():
    specs = build_sequence_experiment_specs()
    spec = specs["EXP-S1"]

    assert [config.family.value for config in spec.model_configs] == [
        "majority_class",
        "sequence_baseline",
        "mlp",
        "lstm",
    ]
    assert spec.split_strategies == [
        SplitStrategy.IID,
        SplitStrategy.LENGTH_EXTRAPOLATION,
        SplitStrategy.VALUE_EXTRAPOLATION,
    ]
    assert spec.seeds == TASK12_SEQUENCE_SEEDS
    assert specs["EXP-S2"].split_strategies == [
        SplitStrategy.IID,
        SplitStrategy.LENGTH_EXTRAPOLATION,
    ]
    assert specs["EXP-S2"].value_train_range is None


def test_run_sequence_experiment_writes_artifacts(tmp_path):
    registry = build_default_registry()
    spec = build_sequence_experiment_specs(registry)["EXP-S1"]
    spec.task_ids = ["S1.2_sort"]
    spec.model_configs = spec.model_configs[:2]
    spec.split_strategies = [SplitStrategy.IID, SplitStrategy.LENGTH_EXTRAPOLATION]
    spec.n_samples = 120
    spec.seeds = [7]
    spec.length_threshold = 8

    artifact = run_sequence_experiment(spec, output_root=tmp_path, registry=registry)

    assert artifact.output_dir.exists()
    assert (artifact.output_dir / "config.json").exists()
    assert (artifact.output_dir / "summary.md").exists()
    assert (artifact.output_dir / "comparison.md").exists()
    assert (artifact.output_dir / "solvability_verdicts.json").exists()
    assert (
        artifact.output_dir
        / "per_task"
        / "S1.2_sort"
        / "metrics.json"
    ).exists()

    verdicts = json.loads(
        (artifact.output_dir / "solvability_verdicts.json").read_text(encoding="utf-8")
    )
    assert "S1.2_sort" in verdicts
