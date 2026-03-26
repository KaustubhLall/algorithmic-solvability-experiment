"""V-8: Report Generator Validation Tests.

Tests for SR-8 (Report Generator) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. File structure is created as expected.
2. JSON artifacts are valid and complete.
3. Markdown summaries are human-readable and internally consistent.
4. Metrics persisted in markdown match metrics.json.
5. Solvability verdict logic matches the Section 9.4 label semantics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation import EvalReport
from src.models.harness import ModelConfig, ModelFamily
from src.registry import build_default_registry
from src.reporting import compute_solvability_verdict, generate_report, generate_report_artifacts
from src.runner import AggregatedResult, ExperimentReport, ExperimentSpec, SingleRunResult
from src.splits import SplitStrategy


@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture()
def report_spec():
    return ExperimentSpec(
        experiment_id="TEST-REPORTING",
        task_ids=["C1.1_numeric_threshold"],
        model_configs=[
            ModelConfig(family=ModelFamily.MAJORITY_CLASS),
            ModelConfig(family=ModelFamily.DECISION_TREE),
        ],
        split_strategies=[SplitStrategy.IID, SplitStrategy.VALUE_EXTRAPOLATION],
        n_samples=100,
        train_fraction=0.8,
        seeds=[11, 22],
        value_feature="x1",
        value_train_range=(0.0, 50.0),
    )


def _classification_eval_report(
    task_id: str,
    split_name: str,
    accuracy: float,
    confusion_matrix: list[list[int]] | None = None,
    error_taxonomy: dict[str, int] | None = None,
) -> EvalReport:
    return EvalReport(
        task_id=task_id,
        split_name=split_name,
        track="classification",
        n_samples=20,
        accuracy=accuracy,
        per_class_metrics=None,
        macro_f1=accuracy,
        weighted_f1=accuracy,
        confusion_matrix=confusion_matrix,
        class_labels=["A", "B"] if confusion_matrix is not None else None,
        exact_match=None,
        token_accuracy=None,
        error_taxonomy=error_taxonomy or {"correct": int(round(accuracy * 20)), "wrong_class": 20 - int(round(accuracy * 20)), "unknown_class": 0},
        metadata_conditioned_metrics={},
    )


def _agg(
    task_id: str,
    model_name: str,
    split: str,
    mean: float,
    std: float = 0.01,
    n_seeds: int = 5,
) -> AggregatedResult:
    return AggregatedResult(
        task_id=task_id,
        model_name=model_name,
        split_strategy=split,
        n_seeds=n_seeds,
        accuracy_mean=mean,
        accuracy_std=std,
        per_seed_accuracy=[mean] * n_seeds,
        extra_metrics={"macro_f1_mean": mean, "macro_f1_std": std},
    )


def _single(
    task_id: str,
    model_name: str,
    split: str,
    seed: int,
    accuracy: float,
    confusion_matrix: list[list[int]] | None = None,
) -> SingleRunResult:
    return SingleRunResult(
        task_id=task_id,
        model_name=model_name,
        split_strategy=split,
        seed=seed,
        eval_report=_classification_eval_report(
            task_id=task_id,
            split_name=f"{split}_seed{seed}",
            accuracy=accuracy,
            confusion_matrix=confusion_matrix,
        ),
        train_size=80,
        test_size=20,
        train_time_seconds=0.05,
        split_metadata={"seed": seed, "split": split},
    )


@pytest.fixture()
def sample_experiment_report(report_spec):
    aggregated = [
        _agg("C1.1_numeric_threshold", "majority_class", "iid", 0.50, std=0.0, n_seeds=2),
        _agg("C1.1_numeric_threshold", "decision_tree", "iid", 0.97, std=0.01, n_seeds=2),
        _agg("C1.1_numeric_threshold", "decision_tree", "value_extrapolation", 0.88, std=0.02, n_seeds=2),
    ]
    single = [
        _single("C1.1_numeric_threshold", "majority_class", "iid", 11, 0.50, [[5, 5], [5, 5]]),
        _single("C1.1_numeric_threshold", "decision_tree", "iid", 11, 0.95, [[9, 1], [0, 10]]),
        _single("C1.1_numeric_threshold", "decision_tree", "value_extrapolation", 11, 0.90, [[8, 2], [0, 10]]),
        _single("C1.1_numeric_threshold", "majority_class", "iid", 22, 0.50, [[5, 5], [5, 5]]),
        _single("C1.1_numeric_threshold", "decision_tree", "iid", 22, 0.99, [[10, 0], [0, 10]]),
        _single("C1.1_numeric_threshold", "decision_tree", "value_extrapolation", 22, 0.86, [[8, 2], [1, 9]]),
    ]
    return ExperimentReport(
        experiment_id=report_spec.experiment_id,
        spec=report_spec,
        seeds_used=[11, 22],
        single_results=single,
        aggregated_results=aggregated,
        total_time_seconds=1.2345,
    )


class TestGenerateReportArtifacts:

    def test_expected_file_structure_is_created(self, sample_experiment_report, registry, tmp_path):
        out_dir = generate_report(sample_experiment_report, output_root=tmp_path, registry=registry)

        assert out_dir == tmp_path / "TEST-REPORTING"
        assert (out_dir / "config.json").exists()
        assert (out_dir / "summary.md").exists()
        assert (out_dir / "comparison.md").exists()
        assert (out_dir / "solvability_verdicts.json").exists()
        assert (out_dir / "per_task" / "C1.1_numeric_threshold" / "metrics.json").exists()
        assert (out_dir / "per_task" / "C1.1_numeric_threshold" / "errors.json").exists()
        assert (out_dir / "per_task" / "C1.1_numeric_threshold" / "confusion.png").exists()
        assert (out_dir / "per_task" / "C1.1_numeric_threshold" / "extrap_curve.png").exists()

    def test_json_artifacts_are_valid(self, sample_experiment_report, registry, tmp_path):
        out_dir = generate_report(sample_experiment_report, output_root=tmp_path, registry=registry)

        json_paths = [
            out_dir / "config.json",
            out_dir / "solvability_verdicts.json",
            out_dir / "per_task" / "C1.1_numeric_threshold" / "metrics.json",
            out_dir / "per_task" / "C1.1_numeric_threshold" / "errors.json",
        ]
        for path in json_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict)

    def test_markdown_summary_is_human_readable_and_consistent(
        self,
        sample_experiment_report,
        registry,
        tmp_path,
    ):
        out_dir = generate_report(sample_experiment_report, output_root=tmp_path, registry=registry)
        summary = (out_dir / "summary.md").read_text(encoding="utf-8")
        metrics = json.loads(
            (out_dir / "per_task" / "C1.1_numeric_threshold" / "metrics.json").read_text(encoding="utf-8")
        )

        assert "# Experiment Summary: TEST-REPORTING" in summary
        assert "| C1.1_numeric_threshold | C1 | classification | 0.9700 (decision_tree) | 0.8800 (decision_tree) | MODERATE |" in summary
        assert metrics["verdict"]["label"] == "MODERATE"
        assert "Best IID accuracy: 0.9700" in summary
        assert "Best OOD accuracy: 0.8800" in summary

    def test_comparison_markdown_contains_cross_task_table(
        self,
        sample_experiment_report,
        registry,
        tmp_path,
    ):
        out_dir = generate_report(sample_experiment_report, output_root=tmp_path, registry=registry)
        comparison = (out_dir / "comparison.md").read_text(encoding="utf-8")

        assert "# Cross-Task Comparison" in comparison
        assert "| C1.1_numeric_threshold | C1 | classification | 0.9700 | 0.8800 | MODERATE |" in comparison

    def test_invalid_experiment_id_is_rejected(self, sample_experiment_report, registry, tmp_path):
        sample_experiment_report.experiment_id = "../escape"
        with pytest.raises(ValueError, match="experiment_id"):
            generate_report(sample_experiment_report, output_root=tmp_path, registry=registry)

    def test_backward_compatible_artifact_alias(self, sample_experiment_report, registry, tmp_path):
        out_dir = generate_report_artifacts(sample_experiment_report, output_root=tmp_path, registry=registry)
        assert out_dir == tmp_path / "TEST-REPORTING"


class TestVerdictLogic:

    def test_strong_requires_all_minimum_plus_two_optional(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        verdict = compute_solvability_verdict(
            task,
            [
                _agg(task.task_id, "majority_class", "iid", 0.50),
                _agg(task.task_id, "decision_tree", "iid", 0.98),
                _agg(task.task_id, "decision_tree", "value_extrapolation", 0.90),
                _agg(task.task_id, "decision_tree", "distractor", 0.96),
                _agg(task.task_id, "decision_tree", "category_combination", 0.88),
            ],
        )

        assert verdict.label == "STRONG"
        assert verdict.evidence["criterion_7_distractor_robustness"] is True
        assert verdict.evidence["criterion_9_transfer"] is True

    def test_moderate_requires_all_minimum_without_two_optional(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        verdict = compute_solvability_verdict(
            task,
            [
                _agg(task.task_id, "majority_class", "iid", 0.50),
                _agg(task.task_id, "decision_tree", "iid", 0.97),
                _agg(task.task_id, "decision_tree", "value_extrapolation", 0.87),
            ],
        )

        assert verdict.label == "MODERATE"

    def test_weak_requires_high_iid_but_missing_other_evidence(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        verdict = compute_solvability_verdict(
            task,
            [_agg(task.task_id, "decision_tree", "iid", 0.97, n_seeds=1)],
        )

        assert verdict.label == "WEAK"

    def test_negative_when_all_tested_models_fail(self, registry):
        task = registry.get("C3.1_xor")
        verdict = compute_solvability_verdict(
            task,
            [
                _agg(task.task_id, "majority_class", "iid", 0.45),
                _agg(task.task_id, "mlp", "iid", 0.55),
                _agg(task.task_id, "mlp", "value_extrapolation", 0.51),
            ],
        )

        assert verdict.label == "NEGATIVE"

    def test_inconclusive_for_mixed_midrange_results(self, registry):
        task = registry.get("C2.1_and_rule")
        verdict = compute_solvability_verdict(
            task,
            [
                _agg(task.task_id, "majority_class", "iid", 0.50),
                _agg(task.task_id, "decision_tree", "iid", 0.92),
                _agg(task.task_id, "decision_tree", "value_extrapolation", 0.82),
            ],
        )

        assert verdict.label == "INCONCLUSIVE"
