"""V-8: Report Generator Validation Tests.

Tests for SR-8 (Report Generator) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. File structure is created under results/{experiment_id}/...
2. JSON artifacts parse correctly.
3. Markdown summaries stay consistent with metrics JSON.
4. Plot artifacts are generated for supported task types.
5. Solvability verdict logic follows EXPERIMENT_DESIGN.md Section 9.4.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.harness import ModelConfig, ModelFamily
from src.registry import TaskSpec, build_default_registry
from src.reporting import assess_task_solvability, generate_report_artifacts
from src.runner import AggregatedResult, ExperimentSpec, run_experiment
from src.splits import SplitStrategy


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture()
def mini_report(registry):
    spec = ExperimentSpec(
        experiment_id="TEST-REPORTING",
        task_ids=["C1.1_numeric_threshold", "S1.2_sort"],
        model_configs=[
            ModelConfig(family=ModelFamily.MAJORITY_CLASS),
            ModelConfig(family=ModelFamily.DECISION_TREE),
        ],
        split_strategies=[SplitStrategy.IID, SplitStrategy.NOISE],
        n_samples=120,
        noise_level=0.15,
        seeds=[42],
    )
    return run_experiment(spec, registry=registry)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _make_agg(
    task_id: str,
    model_name: str,
    split_name: str,
    accuracy_mean: float,
    accuracy_std: float,
    n_seeds: int = 5,
) -> AggregatedResult:
    return AggregatedResult(
        task_id=task_id,
        model_name=model_name,
        split_strategy=split_name,
        n_seeds=n_seeds,
        accuracy_mean=accuracy_mean,
        accuracy_std=accuracy_std,
        per_seed_accuracy=[accuracy_mean] * n_seeds,
        extra_metrics={},
    )


class TestArtifactGeneration:

    def test_expected_file_structure_created(self, mini_report, registry, tmp_path):
        output_dir = generate_report_artifacts(mini_report, output_root=tmp_path, registry=registry)

        assert output_dir == tmp_path / "TEST-REPORTING"
        assert (output_dir / "config.json").exists()
        assert (output_dir / "summary.md").exists()
        assert (output_dir / "comparison.md").exists()
        assert (output_dir / "solvability_verdicts.json").exists()

        classification_dir = output_dir / "per_task" / "C1.1_numeric_threshold"
        sequence_dir = output_dir / "per_task" / "S1.2_sort"

        for task_dir in [classification_dir, sequence_dir]:
            assert (task_dir / "metrics.json").exists()
            assert (task_dir / "errors.json").exists()
            assert (task_dir / "extrap_curve.png").exists()
            assert (task_dir / "extrap_curve.png").stat().st_size > 0

        assert (classification_dir / "confusion.png").exists()
        assert (classification_dir / "confusion.png").stat().st_size > 0
        assert not (sequence_dir / "confusion.png").exists()

    def test_json_artifacts_parse_and_markdown_matches_metrics(self, mini_report, registry, tmp_path):
        output_dir = generate_report_artifacts(mini_report, output_root=tmp_path, registry=registry)

        config = _load_json(output_dir / "config.json")
        metrics = _load_json(output_dir / "per_task" / "C1.1_numeric_threshold" / "metrics.json")
        errors = _load_json(output_dir / "per_task" / "C1.1_numeric_threshold" / "errors.json")
        verdicts = _load_json(output_dir / "solvability_verdicts.json")
        summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
        comparison_md = (output_dir / "comparison.md").read_text(encoding="utf-8")

        assert config["experiment_id"] == "TEST-REPORTING"
        assert metrics["task_id"] == "C1.1_numeric_threshold"
        assert errors["task_id"] == "C1.1_numeric_threshold"
        assert "C1.1_numeric_threshold" in verdicts["verdicts"]
        assert "S1.2_sort" in comparison_md

        best_iid = max(
            result["accuracy_mean"]
            for result in metrics["aggregated_results"]
            if result["split_strategy"] == "iid"
        )
        assert f"{best_iid:.3f}" in summary_md
        assert "C1.1_numeric_threshold" in summary_md
        assert "TEST-REPORTING" in summary_md

    def test_invalid_experiment_id_is_rejected(self, mini_report, registry, tmp_path):
        mini_report.experiment_id = "../escape"
        with pytest.raises(ValueError, match="experiment_id"):
            generate_report_artifacts(mini_report, output_root=tmp_path, registry=registry)


class TestSolvabilityLogic:

    def test_weak_verdict_requires_only_high_iid(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        verdict = assess_task_solvability(
            task,
            [_make_agg(task.task_id, "mlp", "iid", 0.97, 0.01, n_seeds=1)],
        )

        assert verdict.label == "WEAK"
        assert verdict.chosen_model == "mlp"

    def test_moderate_verdict_requires_criteria_1_to_5(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        verdict = assess_task_solvability(
            task,
            [
                _make_agg(task.task_id, "mlp", "iid", 0.98, 0.01),
                _make_agg(task.task_id, "mlp", "noise", 0.92, 0.02),
                _make_agg(task.task_id, "mlp", "value_extrapolation", 0.89, 0.03),
                _make_agg(task.task_id, "majority_class", "iid", 0.62, 0.01),
                _make_agg(task.task_id, "majority_class", "noise", 0.56, 0.01),
                _make_agg(task.task_id, "majority_class", "value_extrapolation", 0.54, 0.01),
            ],
        )

        assert verdict.label == "MODERATE"
        assert verdict.criteria["criterion_1_high_iid_accuracy"] is True
        assert verdict.criteria["criterion_5_coherent_degradation"] is True

    def test_strong_verdict_requires_two_optional_criteria(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        verdict = assess_task_solvability(
            task,
            [
                _make_agg(task.task_id, "mlp", "iid", 0.99, 0.01),
                _make_agg(task.task_id, "mlp", "noise", 0.92, 0.02),
                _make_agg(task.task_id, "mlp", "distractor", 0.97, 0.02),
                _make_agg(task.task_id, "mlp", "value_extrapolation", 0.90, 0.03),
                _make_agg(task.task_id, "mlp", "composition", 0.87, 0.03),
                _make_agg(task.task_id, "majority_class", "iid", 0.63, 0.01),
                _make_agg(task.task_id, "majority_class", "noise", 0.56, 0.01),
                _make_agg(task.task_id, "majority_class", "distractor", 0.55, 0.01),
                _make_agg(task.task_id, "majority_class", "value_extrapolation", 0.54, 0.01),
                _make_agg(task.task_id, "majority_class", "composition", 0.51, 0.01),
            ],
        )

        assert verdict.label == "STRONG"
        assert verdict.criteria["criterion_7_distractor_robustness"] is True
        assert verdict.criteria["criterion_9_transfer"] is True

    def test_negative_verdict_when_all_models_fail(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        verdict = assess_task_solvability(
            task,
            [
                _make_agg(task.task_id, "mlp", "iid", 0.42, 0.02, n_seeds=1),
                _make_agg(task.task_id, "mlp", "noise", 0.38, 0.03, n_seeds=1),
                _make_agg(task.task_id, "majority_class", "iid", 0.45, 0.01, n_seeds=1),
                _make_agg(task.task_id, "majority_class", "noise", 0.40, 0.01, n_seeds=1),
            ],
        )

        assert verdict.label == "NEGATIVE"
