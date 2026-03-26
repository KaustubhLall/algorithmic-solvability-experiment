"""SR-5: Model Harness.

Unified interface for training and predicting with multiple model families.
Wraps sklearn, XGBoost, LightGBM, and simple neural models behind a common API.

Used by: SR-6 (Evaluation Engine), SR-7 (Experiment Runner).
Validated by: V-5 (Model Harness Validation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ===================================================================
# Model configuration
# ===================================================================

class ModelFamily(str, Enum):
    """Supported model families."""
    MAJORITY_CLASS = "majority_class"
    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    KNN = "knn"
    GRADIENT_BOOSTED_TREES = "gradient_boosted_trees"
    MLP = "mlp"
    # Sequence models (placeholder for future)
    SEQUENCE_BASELINE = "sequence_baseline"


@dataclass
class ModelConfig:
    """Configuration for a model instance.

    Attributes:
        family: Which model family.
        hyperparams: Model-specific hyperparameters.
        name: Human-readable name (auto-generated if not set).
    """
    family: Union[ModelFamily, str]
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.family, str):
            try:
                self.family = ModelFamily(self.family)
            except ValueError as exc:
                raise ValueError(f"Unknown model family: {self.family}") from exc
        if self.name is None:
            self.name = self.family.value


# ===================================================================
# Abstract model interface
# ===================================================================

class BaseModel(ABC):
    """Abstract interface that all models must implement."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on features X and labels y."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for features X."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        ...


# ===================================================================
# Encoding utilities
# ===================================================================

class InputEncoder:
    """Encodes raw task inputs into numeric feature matrices for sklearn-style models.

    For tabular inputs: numerical features pass through, categorical features
    are label-encoded.

    For sequence inputs: extracts summary features (length, mean, min, max,
    sum, std, first, last).
    """

    def __init__(self) -> None:
        self._categorical_maps: Dict[str, Dict[str, int]] = {}
        self._feature_names: List[str] = []
        self._fitted = False

    def fit(self, inputs: List[Any]) -> "InputEncoder":
        """Learn encoding from training inputs."""
        if not inputs:
            raise ValueError("Cannot fit on empty inputs")

        sample = inputs[0]
        if isinstance(sample, dict):
            self._fit_tabular(inputs)
        elif isinstance(sample, list):
            self._fit_sequence(inputs)
        else:
            raise ValueError(f"Unsupported input type: {type(sample)}")

        self._fitted = True
        return self

    def transform(self, inputs: List[Any]) -> np.ndarray:
        """Transform inputs into a numeric feature matrix."""
        if not self._fitted:
            raise RuntimeError("InputEncoder must be fit before transform")
        if not inputs:
            return np.zeros((0, len(self._feature_names)), dtype=np.float64)

        sample = inputs[0]
        if isinstance(sample, dict):
            return self._transform_tabular(inputs)
        elif isinstance(sample, list):
            return self._transform_sequence(inputs)
        else:
            raise ValueError(f"Unsupported input type: {type(sample)}")

    def fit_transform(self, inputs: List[Any]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(inputs).transform(inputs)

    def _fit_tabular(self, inputs: List[Dict[str, Any]]) -> None:
        self._feature_names = sorted(inputs[0].keys())
        self._categorical_maps = {}
        for fname in self._feature_names:
            sample_val = inputs[0][fname]
            if isinstance(sample_val, str):
                unique_vals = sorted(set(row[fname] for row in inputs))
                self._categorical_maps[fname] = {v: i for i, v in enumerate(unique_vals)}

    def _transform_tabular(self, inputs: List[Dict[str, Any]]) -> np.ndarray:
        n = len(inputs)
        m = len(self._feature_names)
        X = np.zeros((n, m), dtype=np.float64)
        for i, row in enumerate(inputs):
            for j, fname in enumerate(self._feature_names):
                val = row[fname]
                if fname in self._categorical_maps:
                    X[i, j] = self._categorical_maps[fname].get(str(val), -1)
                else:
                    X[i, j] = float(val)
        return X

    def _fit_sequence(self, inputs: List[List[int]]) -> None:
        self._feature_names = [
            "length", "mean", "std", "min", "max", "sum", "first", "last"
        ]

    def _transform_sequence(self, inputs: List[List[int]]) -> np.ndarray:
        n = len(inputs)
        X = np.zeros((n, 8), dtype=np.float64)
        for i, seq in enumerate(inputs):
            if len(seq) == 0:
                continue
            arr = np.array(seq, dtype=np.float64)
            X[i, 0] = len(seq)
            X[i, 1] = np.mean(arr)
            X[i, 2] = np.std(arr) if len(arr) > 1 else 0.0
            X[i, 3] = np.min(arr)
            X[i, 4] = np.max(arr)
            X[i, 5] = np.sum(arr)
            X[i, 6] = arr[0]
            X[i, 7] = arr[-1]
        return X


class LabelEncoder:
    """Encodes string labels to integers and back."""

    def __init__(self) -> None:
        self._label_to_int: Dict[str, int] = {}
        self._int_to_label: Dict[int, str] = {}
        self._fitted = False

    def fit(self, labels: List[Any]) -> "LabelEncoder":
        unique = sorted(set(str(l) for l in labels))
        self._label_to_int = {l: i for i, l in enumerate(unique)}
        self._int_to_label = {i: l for l, i in self._label_to_int.items()}
        self._fitted = True
        return self

    def transform(self, labels: List[Any]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fit before transform")
        return np.array([self._label_to_int.get(str(l), -1) for l in labels])

    def inverse_transform(self, encoded: np.ndarray) -> List[str]:
        if not self._fitted:
            raise RuntimeError("LabelEncoder must be fit before inverse_transform")
        return [self._int_to_label.get(int(v), "UNKNOWN") for v in encoded]

    def fit_transform(self, labels: List[Any]) -> np.ndarray:
        return self.fit(labels).transform(labels)

    @property
    def n_classes(self) -> int:
        return len(self._label_to_int)


# ===================================================================
# Concrete model implementations
# ===================================================================

class MajorityClassModel(BaseModel):
    """Always predicts the most common training label."""

    def __init__(self) -> None:
        self._majority: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        values, counts = np.unique(y, return_counts=True)
        self._majority = int(values[np.argmax(counts)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._majority is None:
            raise RuntimeError("MajorityClassModel must be fit before predict")
        return np.full(X.shape[0], self._majority)

    def name(self) -> str:
        return "majority_class"


class SklearnModelWrapper(BaseModel):
    """Wraps any sklearn-compatible estimator."""

    def __init__(self, estimator: Any, model_name: str) -> None:
        self._estimator = estimator
        self._model_name = model_name

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._estimator.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict(X)

    def name(self) -> str:
        return self._model_name


class SequenceBaselineModel(BaseModel):
    """Baseline for sequence tasks: encodes sequences as summary features,
    then uses a simple classifier on the encoded features.

    For sequence-to-sequence tasks, this predicts based on input statistics only,
    so it will fail on most tasks. That's intentional — it's a baseline.
    """

    def __init__(self) -> None:
        from sklearn.tree import DecisionTreeClassifier
        self._clf = DecisionTreeClassifier(max_depth=10, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._clf.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def name(self) -> str:
        return "sequence_baseline"


# ===================================================================
# Model factory
# ===================================================================

def build_model(config: ModelConfig) -> BaseModel:
    """Create a model instance from a configuration.

    Args:
        config: Model configuration.

    Returns:
        A BaseModel instance ready for fit/predict.
    """
    hp = config.hyperparams

    if config.family == ModelFamily.MAJORITY_CLASS:
        return MajorityClassModel()

    elif config.family == ModelFamily.LOGISTIC_REGRESSION:
        from sklearn.linear_model import LogisticRegression
        return SklearnModelWrapper(
            LogisticRegression(max_iter=hp.get("max_iter", 1000), random_state=42),
            config.name or "logistic_regression",
        )

    elif config.family == ModelFamily.DECISION_TREE:
        from sklearn.tree import DecisionTreeClassifier
        return SklearnModelWrapper(
            DecisionTreeClassifier(
                max_depth=hp.get("max_depth", 10),
                random_state=42,
            ),
            config.name or "decision_tree",
        )

    elif config.family == ModelFamily.RANDOM_FOREST:
        from sklearn.ensemble import RandomForestClassifier
        return SklearnModelWrapper(
            RandomForestClassifier(
                n_estimators=hp.get("n_estimators", 100),
                max_depth=hp.get("max_depth", 10),
                random_state=42,
                n_jobs=-1,
            ),
            config.name or "random_forest",
        )

    elif config.family == ModelFamily.KNN:
        from sklearn.neighbors import KNeighborsClassifier
        return SklearnModelWrapper(
            KNeighborsClassifier(
                n_neighbors=hp.get("n_neighbors", 5),
            ),
            config.name or "knn",
        )

    elif config.family == ModelFamily.GRADIENT_BOOSTED_TREES:
        from sklearn.ensemble import GradientBoostingClassifier
        return SklearnModelWrapper(
            GradientBoostingClassifier(
                n_estimators=hp.get("n_estimators", 100),
                max_depth=hp.get("max_depth", 5),
                learning_rate=hp.get("learning_rate", 0.1),
                random_state=42,
            ),
            config.name or "gradient_boosted_trees",
        )

    elif config.family == ModelFamily.MLP:
        from sklearn.neural_network import MLPClassifier
        return SklearnModelWrapper(
            MLPClassifier(
                hidden_layer_sizes=hp.get("hidden_layer_sizes", (64, 32)),
                max_iter=hp.get("max_iter", 500),
                random_state=42,
            ),
            config.name or "mlp",
        )

    elif config.family == ModelFamily.SEQUENCE_BASELINE:
        return SequenceBaselineModel()

    else:
        raise ValueError(f"Unknown model family: {config.family}")


# ===================================================================
# Model Harness: high-level train/predict pipeline
# ===================================================================

@dataclass
class PredictionResult:
    """Result of running a model on a dataset split.

    Attributes:
        model_name: Name of the model.
        predictions: Predicted labels (strings).
        true_labels: Ground truth labels (strings).
        train_size: Number of training samples.
        test_size: Number of test samples.
    """
    model_name: str
    predictions: List[str]
    true_labels: List[str]
    train_size: int
    test_size: int


class ModelHarness:
    """High-level harness: encode → train → predict → decode.

    Handles the full pipeline from raw task inputs/outputs to string predictions.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model = build_model(config)
        self._input_encoder = InputEncoder()
        self._label_encoder = LabelEncoder()

    def run(
        self,
        train_inputs: List[Any],
        train_outputs: List[Any],
        test_inputs: List[Any],
        test_outputs: List[Any],
    ) -> PredictionResult:
        """Full pipeline: encode, fit, predict, decode.

        Args:
            train_inputs: Raw training inputs.
            train_outputs: Training labels (any type, will be stringified).
            test_inputs: Raw test inputs.
            test_outputs: Ground truth test labels.

        Returns:
            PredictionResult with string predictions and true labels.
        """
        # Stringify outputs for uniform handling
        train_labels_str = [str(o) for o in train_outputs]
        test_labels_str = [str(o) for o in test_outputs]

        # Encode inputs
        X_train = self._input_encoder.fit_transform(train_inputs)
        X_test = self._input_encoder.transform(test_inputs)

        # Encode labels
        y_train = self._label_encoder.fit_transform(train_labels_str)

        # Train
        self._model.fit(X_train, y_train)

        if len(test_inputs) == 0:
            return PredictionResult(
                model_name=self._model.name(),
                predictions=[],
                true_labels=test_labels_str,
                train_size=len(train_inputs),
                test_size=0,
            )

        # Predict
        y_pred_encoded = self._model.predict(X_test)
        predictions = self._label_encoder.inverse_transform(y_pred_encoded)

        return PredictionResult(
            model_name=self._model.name(),
            predictions=predictions,
            true_labels=test_labels_str,
            train_size=len(train_inputs),
            test_size=len(test_inputs),
        )

    @property
    def model_name(self) -> str:
        return self._model.name()


# ===================================================================
# Convenience: run multiple models on same split
# ===================================================================

def run_models(
    configs: List[ModelConfig],
    train_inputs: List[Any],
    train_outputs: List[Any],
    test_inputs: List[Any],
    test_outputs: List[Any],
) -> List[PredictionResult]:
    """Run multiple models on the same train/test split.

    Returns:
        List of PredictionResult, one per model config.
    """
    results = []
    for config in configs:
        harness = ModelHarness(config)
        result = harness.run(train_inputs, train_outputs, test_inputs, test_outputs)
        results.append(result)
    return results
