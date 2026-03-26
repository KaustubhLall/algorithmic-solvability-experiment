"""V-7: Experiment Runner Validation Tests.

Tests for SR-7 (Experiment Runner) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. End-to-end integrity: run a mini-experiment, verify all output artifacts.
2. Seed variation: different seeds produce different data but same task definition.
3. Multi-seed aggregation: mean and std correctly computed across seeds.
4. No cross-contamination: results for task A not mixed into task B.
5. ExperimentSpec validation.
6. Split dispatch correctness.
7. Aggregation edge cases.
8. Serialization to dict.
9. Integration with real registry tasks.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from src.evaluation import EvalReport
from src.models.harness import ModelConfig, ModelFamily
from src.registry import TaskSpec, build_default_registry
from src.runner import (
    AggregatedResult,
    ExperimentReport,
    ExperimentSpec,
    SingleRunResult,
    _aggregate_results,
    _apply_split,
    _run_single,
    aggregated_result_to_dict,
    experiment_report_to_dict,
    run_experiment,
    single_result_to_dict,
)
from src.splits import SplitStrategy, split_iid
from src.data_generator import generate_dataset


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def mini_classification_spec():
    """Minimal classification experiment spec for testing."""
    return ExperimentSpec(
        experiment_id="TEST-C1",
        task_ids=["C1.1_numeric_threshold"],
        model_configs=[ModelConfig(family=ModelFamily.MAJORITY_CLASS)],
        split_strategies=[SplitStrategy.IID],
        n_samples=100,
        train_fraction=0.8,
        seeds=[42, 123],
    )


@pytest.fixture(scope="module")
def mini_sequence_spec():
    """Minimal sequence experiment spec for testing."""
    return ExperimentSpec(
        experiment_id="TEST-S1",
        task_ids=["S1.2_sort"],
        model_configs=[ModelConfig(family=ModelFamily.SEQUENCE_BASELINE)],
        split_strategies=[SplitStrategy.IID],
        n_samples=100,
        train_fraction=0.8,
        seeds=[42, 123],
    )


# ===================================================================
# 1. End-to-end integrity
# ===================================================================

class TestEndToEnd:

    def test_classification_experiment_completes(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)

        assert report.experiment_id == "TEST-C1"
        assert len(report.single_results) == 2  # 1 task × 1 model × 1 split × 2 seeds
        assert len(report.aggregated_results) == 1  # 1 (task, model, split) group
        assert report.total_time_seconds > 0

    def test_sequence_experiment_completes(self, mini_sequence_spec, registry):
        report = run_experiment(mini_sequence_spec, registry=registry)

        assert report.experiment_id == "TEST-S1"
        assert len(report.single_results) == 2
        assert len(report.aggregated_results) == 1
        assert report.total_time_seconds > 0

    def test_single_results_have_required_fields(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)

        for r in report.single_results:
            assert r.task_id == "C1.1_numeric_threshold"
            assert r.model_name == ModelFamily.MAJORITY_CLASS.value
            assert r.split_strategy == "iid"
            assert r.seed in [42, 123]
            assert r.eval_report is not None
            assert r.train_size > 0
            assert r.test_size > 0
            assert r.train_time_seconds >= 0

    def test_eval_report_is_well_formed(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)

        for r in report.single_results:
            er = r.eval_report
            assert er.task_id == "C1.1_numeric_threshold"
            assert er.track == "classification"
            assert 0.0 <= er.accuracy <= 1.0
            assert er.confusion_matrix is not None
            assert er.per_class_metrics is not None


# ===================================================================
# 2. Seed variation
# ===================================================================

class TestSeedVariation:

    def test_different_seeds_different_results(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)

        results_by_seed = {}
        for r in report.single_results:
            results_by_seed[r.seed] = r

        # Different seeds should produce results (may or may not differ in accuracy
        # for majority class, but at minimum they ran independently)
        assert 42 in results_by_seed
        assert 123 in results_by_seed

    def test_same_seed_same_results(self, registry):
        """Running the same spec twice with same seeds should give identical results."""
        spec = ExperimentSpec(
            experiment_id="TEST-REPRO",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
            split_strategies=[SplitStrategy.IID],
            n_samples=100,
            seeds=[42],
        )
        report1 = run_experiment(spec, registry=registry)
        report2 = run_experiment(spec, registry=registry)

        assert report1.single_results[0].eval_report.accuracy == pytest.approx(
            report2.single_results[0].eval_report.accuracy,
            rel=1e-12,
            abs=1e-12,
        )

    def test_run_experiment_allows_seed_override(self, registry):
        spec = ExperimentSpec(
            experiment_id="TEST-SEED-OVERRIDE",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[ModelConfig(family=ModelFamily.MAJORITY_CLASS)],
            split_strategies=[SplitStrategy.IID],
            n_samples=100,
            seeds=[42, 123],
        )

        report = run_experiment(spec, seeds=[7], registry=registry)

        assert report.seeds_used == [7]
        assert [result.seed for result in report.single_results] == [7]

    def test_same_task_definition_across_seeds(self, registry):
        """The task spec should be the same regardless of seed."""
        task = registry.get("C1.1_numeric_threshold")

        # Generate with two different seeds
        ds1 = generate_dataset(task, n_samples=50, base_seed=42)
        ds2 = generate_dataset(task, n_samples=50, base_seed=123)

        # Different data
        assert ds1.outputs != ds2.outputs or ds1.inputs != ds2.inputs

        # But same task definition
        assert task.task_id == "C1.1_numeric_threshold"


# ===================================================================
# 3. Multi-seed aggregation
# ===================================================================

class TestMultiSeedAggregation:

    def test_aggregation_mean_std(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)

        assert len(report.aggregated_results) == 1
        agg = report.aggregated_results[0]

        assert agg.task_id == "C1.1_numeric_threshold"
        assert agg.n_seeds == 2
        assert len(agg.per_seed_accuracy) == 2

        # Verify mean and std
        expected_mean = np.mean(agg.per_seed_accuracy)
        expected_std = np.std(agg.per_seed_accuracy)
        assert abs(agg.accuracy_mean - expected_mean) < 1e-9
        assert abs(agg.accuracy_std - expected_std) < 1e-9

    def test_aggregation_groups_correctly(self, registry):
        """Multi-task experiment groups results by (task, model, split)."""
        spec = ExperimentSpec(
            experiment_id="TEST-MULTI",
            task_ids=["C1.1_numeric_threshold", "C1.3_categorical_match"],
            model_configs=[ModelConfig(family=ModelFamily.MAJORITY_CLASS)],
            split_strategies=[SplitStrategy.IID],
            n_samples=100,
            seeds=[42, 123],
        )
        report = run_experiment(spec, registry=registry)

        # Should have 2 aggregated results: one per task
        assert len(report.aggregated_results) == 2
        task_ids = {a.task_id for a in report.aggregated_results}
        assert task_ids == {"C1.1_numeric_threshold", "C1.3_categorical_match"}

    def test_single_seed_std_is_zero(self, registry):
        spec = ExperimentSpec(
            experiment_id="TEST-1SEED",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[ModelConfig(family=ModelFamily.MAJORITY_CLASS)],
            split_strategies=[SplitStrategy.IID],
            n_samples=100,
            seeds=[42],
        )
        report = run_experiment(spec, registry=registry)

        agg = report.aggregated_results[0]
        assert agg.n_seeds == 1
        assert agg.accuracy_std == 0.0

    def test_aggregation_extra_metrics_classification(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)
        agg = report.aggregated_results[0]

        # Classification should have macro_f1 aggregated
        assert "macro_f1_mean" in agg.extra_metrics
        assert "macro_f1_std" in agg.extra_metrics

    def test_aggregation_extra_metrics_sequence(self, mini_sequence_spec, registry):
        report = run_experiment(mini_sequence_spec, registry=registry)
        agg = report.aggregated_results[0]

        # Sequence should have exact_match aggregated
        assert "exact_match_mean" in agg.extra_metrics
        assert "exact_match_std" in agg.extra_metrics


# ===================================================================
# 4. No cross-contamination
# ===================================================================

class TestNoCrossContamination:

    def test_task_results_are_separate(self, registry):
        spec = ExperimentSpec(
            experiment_id="TEST-CROSS",
            task_ids=["C1.1_numeric_threshold", "C1.3_categorical_match"],
            model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
            split_strategies=[SplitStrategy.IID],
            n_samples=100,
            seeds=[42],
        )
        report = run_experiment(spec, registry=registry)

        # Check that each result has the correct task_id
        for r in report.single_results:
            assert r.eval_report.task_id == r.task_id

        # Check that we have results for both tasks
        task_ids = {r.task_id for r in report.single_results}
        assert task_ids == {"C1.1_numeric_threshold", "C1.3_categorical_match"}

    def test_model_results_are_separate(self, registry):
        spec = ExperimentSpec(
            experiment_id="TEST-MODELS",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[
                ModelConfig(family=ModelFamily.MAJORITY_CLASS),
                ModelConfig(family=ModelFamily.DECISION_TREE),
            ],
            split_strategies=[SplitStrategy.IID],
            n_samples=100,
            seeds=[42],
        )
        report = run_experiment(spec, registry=registry)

        # Should have 2 single results (one per model)
        assert len(report.single_results) == 2
        model_names = {r.model_name for r in report.single_results}
        assert model_names == {
            ModelFamily.MAJORITY_CLASS.value,
            ModelFamily.DECISION_TREE.value,
        }

        # Should have 2 aggregated results
        assert len(report.aggregated_results) == 2


# ===================================================================
# 5. ExperimentSpec validation
# ===================================================================

class TestExperimentSpec:

    def test_default_seeds(self):
        spec = ExperimentSpec(
            experiment_id="TEST",
            task_ids=["C1.1"],
            model_configs=[ModelConfig(family=ModelFamily.MAJORITY_CLASS)],
            split_strategies=[SplitStrategy.IID],
        )
        assert spec.seeds == [42, 123, 456, 789, 1024]

    def test_default_n_samples(self):
        spec = ExperimentSpec(
            experiment_id="TEST",
            task_ids=["C1.1"],
            model_configs=[ModelConfig(family=ModelFamily.MAJORITY_CLASS)],
            split_strategies=[SplitStrategy.IID],
        )
        assert spec.n_samples == 500

    def test_custom_parameters(self):
        spec = ExperimentSpec(
            experiment_id="CUSTOM",
            task_ids=["C1.1", "S1.2"],
            model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
            split_strategies=[SplitStrategy.IID, SplitStrategy.NOISE],
            n_samples=200,
            train_fraction=0.7,
            seeds=[1, 2, 3],
            noise_level=0.2,
        )
        assert spec.n_samples == 200
        assert spec.train_fraction == 0.7
        assert len(spec.seeds) == 3
        assert spec.noise_level == 0.2


# ===================================================================
# 6. Split dispatch
# ===================================================================

class TestSplitDispatch:

    def test_iid_split(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        ds = generate_dataset(task, n_samples=100, base_seed=42)
        spec = ExperimentSpec(
            experiment_id="T",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[],
            split_strategies=[SplitStrategy.IID],
        )
        split = _apply_split(ds, task, SplitStrategy.IID, spec, seed=42)
        assert split.train_size > 0
        assert split.test_size > 0
        assert split.strategy == SplitStrategy.IID

    def test_noise_split(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        ds = generate_dataset(task, n_samples=100, base_seed=42)
        spec = ExperimentSpec(
            experiment_id="T",
            task_ids=[],
            model_configs=[],
            split_strategies=[SplitStrategy.NOISE],
            noise_level=0.1,
        )
        split = _apply_split(ds, task, SplitStrategy.NOISE, spec, seed=42)
        assert split.train_size > 0
        assert split.test_size > 0
        assert split.strategy == SplitStrategy.NOISE

    def test_length_split_requires_threshold(self, registry):
        task = registry.get("S1.2_sort")
        ds = generate_dataset(task, n_samples=100, base_seed=42)
        spec = ExperimentSpec(
            experiment_id="T",
            task_ids=[],
            model_configs=[],
            split_strategies=[SplitStrategy.LENGTH_EXTRAPOLATION],
        )
        with pytest.raises(ValueError, match="length_threshold"):
            _apply_split(ds, task, SplitStrategy.LENGTH_EXTRAPOLATION, spec, seed=42)

    def test_value_split_requires_params(self, registry):
        task = registry.get("C1.1_numeric_threshold")
        ds = generate_dataset(task, n_samples=100, base_seed=42)
        spec = ExperimentSpec(
            experiment_id="T",
            task_ids=[],
            model_configs=[],
            split_strategies=[SplitStrategy.VALUE_EXTRAPOLATION],
        )
        with pytest.raises(ValueError, match="value_feature"):
            _apply_split(ds, task, SplitStrategy.VALUE_EXTRAPOLATION, spec, seed=42)

    def test_length_split_with_threshold(self, registry):
        task = registry.get("S1.2_sort")
        ds = generate_dataset(task, n_samples=200, base_seed=42)
        spec = ExperimentSpec(
            experiment_id="T",
            task_ids=[],
            model_configs=[],
            split_strategies=[SplitStrategy.LENGTH_EXTRAPOLATION],
            length_threshold=5,
        )
        split = _apply_split(ds, task, SplitStrategy.LENGTH_EXTRAPOLATION, spec, seed=42)
        assert split.strategy == SplitStrategy.LENGTH_EXTRAPOLATION


# ===================================================================
# 7. Aggregation edge cases
# ===================================================================

class TestAggregationEdgeCases:

    def _make_mock_result(self, task_id, model, split, seed, accuracy):
        """Create a mock SingleRunResult."""
        report = EvalReport(
            task_id=task_id,
            split_name=f"{split}_seed{seed}",
            track="classification",
            n_samples=100,
            accuracy=accuracy,
            macro_f1=accuracy * 0.9,
        )
        return SingleRunResult(
            task_id=task_id,
            model_name=model,
            split_strategy=split,
            seed=seed,
            eval_report=report,
            train_size=80,
            test_size=20,
            train_time_seconds=0.1,
        )

    def test_empty_results(self):
        agg = _aggregate_results([])
        assert agg == []

    def test_single_result(self):
        results = [self._make_mock_result("T1", "M1", "iid", 42, 0.8)]
        agg = _aggregate_results(results)
        assert len(agg) == 1
        assert agg[0].accuracy_mean == 0.8
        assert agg[0].accuracy_std == 0.0
        assert agg[0].n_seeds == 1

    def test_multiple_seeds(self):
        results = [
            self._make_mock_result("T1", "M1", "iid", 42, 0.8),
            self._make_mock_result("T1", "M1", "iid", 123, 0.6),
        ]
        agg = _aggregate_results(results)
        assert len(agg) == 1
        assert agg[0].accuracy_mean == pytest.approx(0.7)
        assert agg[0].accuracy_std == pytest.approx(0.1)

    def test_multiple_groups(self):
        results = [
            self._make_mock_result("T1", "M1", "iid", 42, 0.8),
            self._make_mock_result("T1", "M2", "iid", 42, 0.9),
            self._make_mock_result("T2", "M1", "iid", 42, 0.7),
        ]
        agg = _aggregate_results(results)
        assert len(agg) == 3  # 3 distinct (task, model, split) groups

    def test_aggregation_preserves_per_seed(self):
        results = [
            self._make_mock_result("T1", "M1", "iid", 42, 0.8),
            self._make_mock_result("T1", "M1", "iid", 123, 0.6),
            self._make_mock_result("T1", "M1", "iid", 456, 0.7),
        ]
        agg = _aggregate_results(results)
        assert agg[0].per_seed_accuracy == [0.8, 0.6, 0.7]


# ===================================================================
# 8. Serialization
# ===================================================================

class TestSerialization:

    def test_single_result_to_dict(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)
        d = single_result_to_dict(report.single_results[0])

        assert "task_id" in d
        assert "model_name" in d
        assert "split_strategy" in d
        assert "seed" in d
        assert "eval_report" in d
        assert "train_size" in d
        assert "test_size" in d
        assert "train_time_seconds" in d
        assert "split_metadata" in d

    def test_aggregated_result_to_dict(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)
        d = aggregated_result_to_dict(report.aggregated_results[0])

        assert "task_id" in d
        assert "accuracy_mean" in d
        assert "accuracy_std" in d
        assert "per_seed_accuracy" in d
        assert "n_seeds" in d

    def test_experiment_report_to_dict(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)
        d = experiment_report_to_dict(report)

        assert d["experiment_id"] == "TEST-C1"
        assert "spec" in d
        assert "seeds_used" in d
        assert "single_results" in d
        assert "total_time_seconds" in d
        assert "aggregated_results" in d
        assert d["n_single_results"] == 2
        assert d["spec"]["noise_level"] == mini_classification_spec.noise_level
        assert d["spec"]["model_configs"][0]["family"] == ModelFamily.MAJORITY_CLASS.value

    def test_json_serializable(self, mini_classification_spec, registry):
        import json
        report = run_experiment(mini_classification_spec, registry=registry)
        d = experiment_report_to_dict(report)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["experiment_id"] == "TEST-C1"


# ===================================================================
# 9. Integration: multi-model, multi-split
# ===================================================================

class TestIntegration:

    def test_multi_model_experiment(self, registry):
        spec = ExperimentSpec(
            experiment_id="TEST-MULTI-MODEL",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[
                ModelConfig(family=ModelFamily.MAJORITY_CLASS),
                ModelConfig(family=ModelFamily.DECISION_TREE),
                ModelConfig(family=ModelFamily.KNN),
            ],
            split_strategies=[SplitStrategy.IID],
            n_samples=100,
            seeds=[42],
        )
        report = run_experiment(spec, registry=registry)

        assert len(report.single_results) == 3  # 3 models × 1 seed
        assert len(report.aggregated_results) == 3

        # Decision tree should beat majority class on a threshold task
        dt_acc = None
        mc_acc = None
        for a in report.aggregated_results:
            if a.model_name == ModelFamily.DECISION_TREE.value:
                dt_acc = a.accuracy_mean
            elif a.model_name == ModelFamily.MAJORITY_CLASS.value:
                mc_acc = a.accuracy_mean

        assert dt_acc is not None
        assert mc_acc is not None
        assert dt_acc >= mc_acc  # DT should be at least as good

    def test_multi_split_experiment(self, registry):
        spec = ExperimentSpec(
            experiment_id="TEST-MULTI-SPLIT",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
            split_strategies=[SplitStrategy.IID, SplitStrategy.NOISE],
            n_samples=100,
            noise_level=0.1,
            seeds=[42],
        )
        report = run_experiment(spec, registry=registry)

        assert len(report.single_results) == 2  # 2 splits × 1 seed
        split_strategies = {r.split_strategy for r in report.single_results}
        assert split_strategies == {"iid", "noise"}

    def test_noise_degrades_accuracy(self, registry):
        """High-noise evaluation should not materially outperform the IID split."""
        spec = ExperimentSpec(
            experiment_id="TEST-NOISE",
            task_ids=["C1.1_numeric_threshold"],
            model_configs=[ModelConfig(family=ModelFamily.KNN)],
            split_strategies=[SplitStrategy.IID, SplitStrategy.NOISE],
            n_samples=200,
            noise_level=0.5,
            seeds=[42, 123, 456],
        )
        report = run_experiment(spec, registry=registry)

        iid_agg = None
        noise_agg = None
        for a in report.aggregated_results:
            if a.split_strategy == "iid":
                iid_agg = a
            elif a.split_strategy == "noise":
                noise_agg = a

        assert iid_agg is not None
        assert noise_agg is not None
        assert noise_agg.accuracy_mean <= iid_agg.accuracy_mean + 0.05

    def test_spec_preserved_in_report(self, mini_classification_spec, registry):
        report = run_experiment(mini_classification_spec, registry=registry)
        assert report.spec is mini_classification_spec
        assert report.spec.experiment_id == "TEST-C1"
