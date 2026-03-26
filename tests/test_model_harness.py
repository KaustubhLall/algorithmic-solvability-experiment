"""V-5: Model Harness Validation Tests.

Tests for SR-5 (Model Harness) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. Every model family can be instantiated and produces predictions.
2. Predictions are the correct shape and type.
3. Majority class baseline always predicts the mode.
4. Decision tree achieves perfect accuracy on a trivially separable task.
5. InputEncoder and LabelEncoder work correctly.
6. ModelHarness end-to-end pipeline works.
7. run_models convenience function works.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from src.data_generator import generate_dataset
from src.models.harness import (
    BaseModel,
    InputEncoder,
    LabelEncoder,
    MajorityClassModel,
    ModelConfig,
    ModelFamily,
    ModelHarness,
    PredictionResult,
    SklearnModelWrapper,
    build_model,
    run_models,
)
from src.registry import build_default_registry
from src.splits import split_iid


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def threshold_split(registry):
    task = registry.get("C1.1_numeric_threshold")
    ds = generate_dataset(task, n_samples=300, base_seed=42)
    return split_iid(ds, train_fraction=0.8, seed=42)


@pytest.fixture(scope="module")
def sort_split(registry):
    task = registry.get("S1.2_sort")
    ds = generate_dataset(task, n_samples=200, base_seed=42)
    return split_iid(ds, train_fraction=0.8, seed=42)


# ===================================================================
# 1. Model Instantiation
# ===================================================================

class TestModelInstantiation:

    @pytest.mark.parametrize("family", [
        ModelFamily.MAJORITY_CLASS,
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.DECISION_TREE,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.KNN,
        ModelFamily.GRADIENT_BOOSTED_TREES,
        ModelFamily.MLP,
        ModelFamily.SEQUENCE_BASELINE,
    ])
    def test_build_model(self, family):
        config = ModelConfig(family=family)
        model = build_model(config)
        assert isinstance(model, BaseModel)
        assert model.name() is not None

    def test_unknown_family_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            config = ModelConfig(family="nonexistent")
            build_model(config)

    def test_custom_hyperparams(self):
        config = ModelConfig(
            family=ModelFamily.DECISION_TREE,
            hyperparams={"max_depth": 3},
        )
        model = build_model(config)
        assert isinstance(model, BaseModel)


# ===================================================================
# 2. Prediction Shape and Type
# ===================================================================

class TestPredictionShape:

    def test_classification_predictions_shape(self, threshold_split):
        config = ModelConfig(family=ModelFamily.DECISION_TREE)
        harness = ModelHarness(config)
        result = harness.run(
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            threshold_split.test_inputs,
            threshold_split.test_outputs,
        )
        assert len(result.predictions) == threshold_split.test_size
        assert len(result.true_labels) == threshold_split.test_size
        assert all(isinstance(p, str) for p in result.predictions)

    def test_sequence_predictions_shape(self, sort_split):
        config = ModelConfig(family=ModelFamily.SEQUENCE_BASELINE)
        harness = ModelHarness(config)
        result = harness.run(
            sort_split.train_inputs,
            sort_split.train_outputs,
            sort_split.test_inputs,
            sort_split.test_outputs,
        )
        assert len(result.predictions) == sort_split.test_size


# ===================================================================
# 3. Majority Class Baseline
# ===================================================================

class TestMajorityClass:

    def test_predicts_mode(self):
        X = np.zeros((10, 2))
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
        model = MajorityClassModel()
        model.fit(X, y)
        preds = model.predict(np.zeros((5, 2)))
        assert all(p == 0 for p in preds)

    def test_via_harness(self, threshold_split):
        config = ModelConfig(family=ModelFamily.MAJORITY_CLASS)
        harness = ModelHarness(config)
        result = harness.run(
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            threshold_split.test_inputs,
            threshold_split.test_outputs,
        )
        # All predictions should be the same class
        unique_preds = set(result.predictions)
        assert len(unique_preds) == 1

    def test_predict_before_fit_raises_runtime_error(self):
        model = MajorityClassModel()
        with pytest.raises(RuntimeError, match="fit before predict"):
            model.predict(np.zeros((2, 3)))


# ===================================================================
# 4. Decision Tree on Trivial Task
# ===================================================================

class TestDecisionTreePerfect:

    def test_perfect_on_threshold(self, threshold_split):
        """Decision tree should achieve near-perfect accuracy on C1.1_numeric_threshold."""
        config = ModelConfig(family=ModelFamily.DECISION_TREE)
        harness = ModelHarness(config)
        result = harness.run(
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            threshold_split.test_inputs,
            threshold_split.test_outputs,
        )
        correct = sum(1 for p, t in zip(result.predictions, result.true_labels) if p == t)
        accuracy = correct / len(result.predictions)
        assert accuracy > 0.95, f"Decision tree accuracy {accuracy:.2%} too low on trivial task"


# ===================================================================
# 5. InputEncoder
# ===================================================================

class TestInputEncoder:

    def test_tabular_encoding(self):
        inputs = [
            {"x1": 1.0, "x2": 2.0, "cat": "A"},
            {"x1": 3.0, "x2": 4.0, "cat": "B"},
            {"x1": 5.0, "x2": 6.0, "cat": "A"},
        ]
        enc = InputEncoder()
        X = enc.fit_transform(inputs)
        assert X.shape == (3, 3)
        # cat is label-encoded: A=0, B=1
        assert X[0, 0] == 0.0  # cat=A -> 0
        assert X[1, 0] == 1.0  # cat=B -> 1

    def test_sequence_encoding(self):
        inputs = [[1, 2, 3], [4, 5], [0, 0, 0, 0]]
        enc = InputEncoder()
        X = enc.fit_transform(inputs)
        assert X.shape == (3, 8)
        # First row: len=3, mean=2, min=1, max=3, sum=6, first=1, last=3
        assert X[0, 0] == 3.0  # length
        assert X[0, 5] == 6.0  # sum

    def test_empty_raises(self):
        enc = InputEncoder()
        with pytest.raises(ValueError, match="empty"):
            enc.fit([])

    def test_transform_before_fit_raises(self):
        enc = InputEncoder()
        with pytest.raises(RuntimeError, match="fit"):
            enc.transform([{"x": 1.0}])

    def test_transform_empty_sequence_batch_after_fit(self):
        enc = InputEncoder()
        enc.fit([[1, 2, 3]])
        X = enc.transform([])
        assert X.shape == (0, 8)


# ===================================================================
# 6. LabelEncoder
# ===================================================================

class TestLabelEncoder:

    def test_encode_decode(self):
        enc = LabelEncoder()
        labels = ["A", "B", "C", "A", "B"]
        encoded = enc.fit_transform(labels)
        assert len(encoded) == 5
        decoded = enc.inverse_transform(encoded)
        assert decoded == labels

    def test_n_classes(self):
        enc = LabelEncoder()
        enc.fit(["A", "B", "C"])
        assert enc.n_classes == 3

    def test_unseen_label(self):
        enc = LabelEncoder()
        enc.fit(["A", "B"])
        result = enc.transform(["C"])
        assert result[0] == -1

    def test_inverse_unseen(self):
        enc = LabelEncoder()
        enc.fit(["A", "B"])
        result = enc.inverse_transform(np.array([99]))
        assert result[0] == "UNKNOWN"


# ===================================================================
# 7. End-to-End ModelHarness
# ===================================================================

class TestModelHarnessE2E:

    def test_prediction_result_fields(self, threshold_split):
        config = ModelConfig(family=ModelFamily.DECISION_TREE)
        harness = ModelHarness(config)
        result = harness.run(
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            threshold_split.test_inputs,
            threshold_split.test_outputs,
        )
        assert isinstance(result, PredictionResult)
        assert result.model_name == "decision_tree"
        assert result.train_size == threshold_split.train_size
        assert result.test_size == threshold_split.test_size

    def test_model_name_property(self):
        config = ModelConfig(family=ModelFamily.RANDOM_FOREST, name="rf_v1")
        harness = ModelHarness(config)
        assert harness.model_name == "rf_v1"

    def test_string_family_is_normalized(self):
        config = ModelConfig(family="decision_tree")
        assert config.family == ModelFamily.DECISION_TREE
        harness = ModelHarness(config)
        assert harness.model_name == "decision_tree"

    def test_empty_test_split_returns_empty_predictions(self, threshold_split):
        config = ModelConfig(family=ModelFamily.DECISION_TREE)
        harness = ModelHarness(config)
        result = harness.run(
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            [],
            [],
        )
        assert result.predictions == []
        assert result.true_labels == []
        assert result.test_size == 0

    def test_train_length_mismatch_raises(self, threshold_split):
        config = ModelConfig(family=ModelFamily.DECISION_TREE)
        harness = ModelHarness(config)
        with pytest.raises(ValueError, match="train_inputs length"):
            harness.run(
                threshold_split.train_inputs[:-1],
                threshold_split.train_outputs,
                threshold_split.test_inputs,
                threshold_split.test_outputs,
            )

    def test_test_length_mismatch_raises(self, threshold_split):
        config = ModelConfig(family=ModelFamily.DECISION_TREE)
        harness = ModelHarness(config)
        with pytest.raises(ValueError, match="test_inputs length"):
            harness.run(
                threshold_split.train_inputs,
                threshold_split.train_outputs,
                threshold_split.test_inputs[:-1],
                threshold_split.test_outputs,
            )


# ===================================================================
# 8. run_models Convenience
# ===================================================================

class TestRunModels:

    def test_run_multiple_models(self, threshold_split):
        configs = [
            ModelConfig(family=ModelFamily.MAJORITY_CLASS),
            ModelConfig(family=ModelFamily.DECISION_TREE),
            ModelConfig(family=ModelFamily.LOGISTIC_REGRESSION),
        ]
        results = run_models(
            configs,
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            threshold_split.test_inputs,
            threshold_split.test_outputs,
        )
        assert len(results) == 3
        names = [r.model_name for r in results]
        assert "majority_class" in names
        assert "decision_tree" in names
        assert "logistic_regression" in names

    def test_all_results_have_correct_size(self, threshold_split):
        configs = [
            ModelConfig(family=ModelFamily.MAJORITY_CLASS),
            ModelConfig(family=ModelFamily.KNN),
        ]
        results = run_models(
            configs,
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            threshold_split.test_inputs,
            threshold_split.test_outputs,
        )
        for r in results:
            assert len(r.predictions) == threshold_split.test_size
            assert len(r.true_labels) == threshold_split.test_size


# ===================================================================
# 9. Multiple Model Families on Classification
# ===================================================================

class TestMultipleFamilies:

    @pytest.mark.parametrize("family", [
        ModelFamily.LOGISTIC_REGRESSION,
        ModelFamily.DECISION_TREE,
        ModelFamily.RANDOM_FOREST,
        ModelFamily.KNN,
        ModelFamily.GRADIENT_BOOSTED_TREES,
        ModelFamily.MLP,
    ])
    def test_family_runs_on_classification(self, family, threshold_split):
        config = ModelConfig(family=family)
        harness = ModelHarness(config)
        result = harness.run(
            threshold_split.train_inputs,
            threshold_split.train_outputs,
            threshold_split.test_inputs,
            threshold_split.test_outputs,
        )
        assert len(result.predictions) == threshold_split.test_size
        # At least some predictions should be valid class labels
        valid_labels = set(str(o) for o in threshold_split.train_outputs)
        assert any(p in valid_labels for p in result.predictions)
