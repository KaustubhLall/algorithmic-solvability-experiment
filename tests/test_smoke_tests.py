"""TASK-11 smoke-suite validation tests."""

from __future__ import annotations

import json

import pytest

from src.data_generator import generate_dataset
from src.evaluation import evaluate
from src.models.harness import ModelConfig, ModelFamily, ModelHarness
from src.registry import build_default_registry
from src.reporting import compute_solvability_verdicts
from src.runner import ExperimentSpec, run_experiment
from src.smoke_tests import run_all_smoke_experiments
from src.splits import SplitStrategy, split_iid


@pytest.fixture(scope="module")
def smoke_suite(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("task11-smoke")
    return run_all_smoke_experiments(output_root=output_root)


def _aggregated_metric(report, task_id: str, model_name: str, split_strategy: str):
    for result in report.aggregated_results:
        if (
            result.task_id == task_id
            and result.model_name == model_name
            and result.split_strategy == split_strategy
        ):
            return result
    raise AssertionError(
        f"Missing aggregated result for {task_id}/{model_name}/{split_strategy}"
    )


class TestSmokeSuiteExecution:

    def test_artifacts_exist_for_all_experiments(self, smoke_suite):
        assert set(smoke_suite) == {"EXP-0.1", "EXP-0.2", "EXP-0.3"}

        for artifact in smoke_suite.values():
            assert artifact.output_dir.exists()
            assert (artifact.output_dir / "config.json").exists()
            assert (artifact.output_dir / "summary.md").exists()
            assert (artifact.output_dir / "comparison.md").exists()
            assert (artifact.output_dir / "solvability_verdicts.json").exists()

    def test_exp_0_1_sequence_smoke_lstm_clears_threshold(self, smoke_suite):
        report = smoke_suite["EXP-0.1"].report
        lstm_result = _aggregated_metric(report, "S1.2_sort", "lstm", "iid")

        assert lstm_result.accuracy_mean > 0.90
        assert lstm_result.extra_metrics["exact_match_mean"] > 0.90
        assert lstm_result.extra_metrics["token_accuracy_mean"] > 0.95

    def test_exp_0_2_classification_smoke_hits_trivial_ceiling(self, smoke_suite):
        artifact = smoke_suite["EXP-0.2"]
        report = artifact.report
        dt_iid = _aggregated_metric(
            report,
            "C1.1_numeric_threshold",
            "decision_tree",
            "iid",
        )
        lr_iid = _aggregated_metric(
            report,
            "C1.1_numeric_threshold",
            "logistic_regression",
            "iid",
        )

        verdicts = compute_solvability_verdicts(report)

        assert dt_iid.accuracy_mean == pytest.approx(1.0)
        assert lr_iid.accuracy_mean == pytest.approx(1.0)
        assert verdicts["C1.1_numeric_threshold"]["label"] in {"MODERATE", "STRONG"}
        assert (
            artifact.output_dir
            / "per_task"
            / "C1.1_numeric_threshold"
            / "confusion.png"
        ).exists()

    def test_exp_0_3_control_tasks_stay_negative(self, smoke_suite):
        report = smoke_suite["EXP-0.3"].report
        verdicts = compute_solvability_verdicts(report)

        assert verdicts["C0.1_random_class"]["label"] in {"WEAK", "NEGATIVE"}
        assert verdicts["S0.1_random_labels"]["label"] in {"WEAK", "NEGATIVE"}
        assert verdicts["C0.1_random_class"]["best_iid_accuracy"] < 0.40
        assert verdicts["S0.1_random_labels"]["best_iid_accuracy"] < 0.10


class TestGlobalValidation:

    def test_vg1_round_trip_accuracy_matches_report(self, tmp_path):
        registry = build_default_registry()
        task = registry.get("C1.1_numeric_threshold")
        dataset = generate_dataset(task, n_samples=120, base_seed=42)
        split = split_iid(dataset, train_fraction=100 / 120, seed=42)

        harness = ModelHarness(ModelConfig(family=ModelFamily.DECISION_TREE))
        prediction_result = harness.run(
            split.train_inputs,
            split.train_outputs,
            split.test_inputs,
            split.test_outputs,
        )
        manual_accuracy = sum(
            1
            for pred, true in zip(
                prediction_result.predictions,
                prediction_result.true_labels,
            )
            if pred == true
        ) / len(prediction_result.predictions)
        eval_report = evaluate(
            prediction_result.predictions,
            prediction_result.true_labels,
            task,
            split_name="iid_seed42",
        )

        spec = ExperimentSpec(
            experiment_id="VG1-CHECK",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
            split_strategies=[SplitStrategy.IID],
            n_samples=120,
            train_fraction=100 / 120,
            seeds=[42],
        )
        experiment_report = run_experiment(spec, registry=registry)

        aggregated = _aggregated_metric(
            experiment_report,
            "C1.1_numeric_threshold",
            "decision_tree",
            "iid",
        )

        assert eval_report.accuracy == pytest.approx(manual_accuracy)
        assert aggregated.accuracy_mean == pytest.approx(manual_accuracy)

    def test_vg4_runner_generates_fresh_data_per_task(self, monkeypatch):
        from src import runner as runner_module

        observed_calls = []
        original_generate_dataset = runner_module.generate_dataset

        def wrapped_generate_dataset(task, n_samples, base_seed=0, noise_level=0.0, verify=True):
            dataset = original_generate_dataset(
                task,
                n_samples=n_samples,
                base_seed=base_seed,
                noise_level=noise_level,
                verify=verify,
            )
            observed_calls.append(
                {
                    "task_id": task.task_id,
                    "dataset": dataset,
                    "sample_task_ids": {sample.task_id for sample in dataset.samples},
                }
            )
            return dataset

        monkeypatch.setattr(runner_module, "generate_dataset", wrapped_generate_dataset)

        spec = ExperimentSpec(
            experiment_id="VG4-CHECK",
            task_ids=["C1.1_numeric_threshold", "C0.1_random_class"],
            model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
            split_strategies=[SplitStrategy.IID],
            n_samples=40,
            train_fraction=0.75,
            seeds=[7],
        )
        run_experiment(spec, registry=build_default_registry())

        assert [call["task_id"] for call in observed_calls] == [
            "C1.1_numeric_threshold",
            "C0.1_random_class",
        ]
        assert observed_calls[0]["dataset"] is not observed_calls[1]["dataset"]
        assert observed_calls[0]["sample_task_ids"] == {"C1.1_numeric_threshold"}
        assert observed_calls[1]["sample_task_ids"] == {"C0.1_random_class"}

    def test_report_summary_is_internally_consistent(self, smoke_suite):
        artifact = smoke_suite["EXP-0.2"]
        summary_text = (artifact.output_dir / "summary.md").read_text(encoding="utf-8")
        metrics_payload = json.loads(
            (
                artifact.output_dir
                / "per_task"
                / "C1.1_numeric_threshold"
                / "metrics.json"
            ).read_text(encoding="utf-8")
        )
        verdict_payload = json.loads(
            (artifact.output_dir / "solvability_verdicts.json").read_text(encoding="utf-8")
        )

        decision_tree_iid = _aggregated_metric(
            artifact.report,
            "C1.1_numeric_threshold",
            "decision_tree",
            "iid",
        )

        assert "C1.1_numeric_threshold" in summary_text
        assert (
            metrics_payload["verdict"]["label"]
            == verdict_payload["C1.1_numeric_threshold"]["label"]
        )
        assert decision_tree_iid.accuracy_mean == pytest.approx(1.0)
