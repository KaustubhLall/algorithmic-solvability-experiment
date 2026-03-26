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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    LSTM = "lstm"
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


class _SequenceLSTMNet:
    """Small pooled encoder used for sequence smoke tests."""

    def __init__(
        self,
        vocab_size: int,
        output_vocab_size: int,
        max_output_len: int,
        pad_idx: int,
        embedding_dim: int,
        hidden_size: int,
    ) -> None:
        import torch
        from torch import nn

        feature_dim = (hidden_size * 2) + (vocab_size - 1)

        class SequenceEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(
                    vocab_size,
                    embedding_dim,
                    padding_idx=pad_idx,
                )
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(feature_dim, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_size * 2,
                        max_output_len * output_vocab_size,
                    ),
                )

            def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
                embedded = self.embedding(tokens)
                packed = nn.utils.rnn.pack_padded_sequence(
                    embedded,
                    lengths.cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                encoded, _ = self.lstm(packed)
                encoded, _ = nn.utils.rnn.pad_packed_sequence(
                    encoded,
                    batch_first=True,
                    total_length=tokens.shape[1],
                )
                time_steps = encoded.shape[1]
                mask = (
                    torch.arange(time_steps, device=encoded.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
                counts = torch.zeros(
                    tokens.shape[0],
                    vocab_size - 1,
                    device=tokens.device,
                    dtype=encoded.dtype,
                )
                valid_tokens = tokens.clamp(min=1) - 1
                counts.scatter_add_(
                    1,
                    valid_tokens,
                    mask.to(encoded.dtype),
                )
                counts = counts / lengths.unsqueeze(1)
                logits = self.head(torch.cat([pooled, counts], dim=1))
                return logits.view(-1, max_output_len, output_vocab_size)

        self.module = SequenceEncoder()


class LSTMSequenceModel(BaseModel):
    """PyTorch LSTM encoder for sequence-to-sequence smoke tests.

    The model consumes raw integer sequences, pools bidirectional LSTM states,
    and predicts each output position independently. That is sufficient for the
    bounded smoke-test regime while keeping the public harness API unchanged.
    """

    def __init__(
        self,
        model_name: str = "lstm",
        embedding_dim: int = 32,
        hidden_size: int = 64,
        epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._hidden_size = hidden_size
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._random_state = random_state

        self._pad_idx = 0
        self._token_offset = 1
        self._max_output_len = 0
        self._same_length_outputs = False
        self._trained = False
        self._network: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        train_inputs = self._normalize_sequences(X, "train_inputs")
        train_outputs = self._normalize_sequences(y, "train_outputs")
        if not train_inputs:
            raise ValueError("LSTM sequence model requires non-empty training data")

        all_tokens = [token for seq in train_inputs + train_outputs for token in seq]
        if not all_tokens:
            raise ValueError("LSTM sequence model requires at least one token")

        min_token = min(all_tokens)
        max_token = max(all_tokens)
        self._token_offset = 1 - min_token if min_token <= 0 else 1
        vocab_size = max_token + self._token_offset + 1
        self._max_output_len = max(len(seq) for seq in train_outputs)
        self._same_length_outputs = all(
            len(inp) == len(out)
            for inp, out in zip(train_inputs, train_outputs)
        )

        torch.manual_seed(self._random_state)

        train_input_tensor, train_input_lengths = self._encode_sequences(
            train_inputs,
            max(len(seq) for seq in train_inputs),
        )
        train_output_tensor, _ = self._encode_sequences(
            train_outputs,
            self._max_output_len,
        )

        sequence_net = _SequenceLSTMNet(
            vocab_size=vocab_size,
            output_vocab_size=vocab_size,
            max_output_len=self._max_output_len,
            pad_idx=self._pad_idx,
            embedding_dim=self._embedding_dim,
            hidden_size=self._hidden_size,
        )
        self._network = sequence_net.module
        self._network.train()

        optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=self._learning_rate,
        )
        dataset = TensorDataset(
            train_input_tensor,
            train_input_lengths,
            train_output_tensor,
        )
        loader = DataLoader(
            dataset,
            batch_size=min(self._batch_size, len(dataset)),
            shuffle=True,
        )

        for _ in range(self._epochs):
            for batch_inputs, batch_lengths, batch_targets in loader:
                optimizer.zero_grad()
                logits = self._network(batch_inputs, batch_lengths)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    batch_targets.reshape(-1),
                    ignore_index=self._pad_idx,
                )
                loss.backward()
                optimizer.step()

        self._trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        if not self._trained or self._network is None:
            raise RuntimeError("LSTMSequenceModel must be fit before predict")

        inputs = self._normalize_sequences(X, "test_inputs")
        if not inputs:
            return np.array([], dtype=object)

        input_tensor, input_lengths = self._encode_sequences(
            inputs,
            max(len(seq) for seq in inputs),
        )

        self._network.eval()
        with torch.no_grad():
            logits = self._network(input_tensor, input_lengths)
            encoded_predictions = logits.argmax(dim=-1).cpu().tolist()

        decoded_predictions: List[List[int]] = []
        for encoded, source in zip(encoded_predictions, inputs):
            if self._same_length_outputs:
                target_len = len(source)
            else:
                target_len = self._first_pad_index(encoded)

            decoded_predictions.append([
                int(token - self._token_offset)
                for token in encoded[:target_len]
                if token != self._pad_idx
            ])

        return np.array(decoded_predictions, dtype=object)

    def name(self) -> str:
        return self._model_name

    def _normalize_sequences(
        self,
        values: Union[np.ndarray, Sequence[Any]],
        field_name: str,
    ) -> List[List[int]]:
        raw_values = values.tolist() if isinstance(values, np.ndarray) else list(values)
        sequences: List[List[int]] = []
        for seq in raw_values:
            if not isinstance(seq, list):
                raise ValueError(
                    f"LSTM sequence model requires list[int] {field_name}, "
                    f"got {type(seq)}"
                )
            try:
                sequences.append([int(token) for token in seq])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"LSTM sequence model requires integer tokens in {field_name}"
                ) from exc
        return sequences

    def _encode_sequences(
        self,
        sequences: List[List[int]],
        max_len: int,
    ) -> Tuple[Any, Any]:
        import torch

        encoded = np.full((len(sequences), max_len), self._pad_idx, dtype=np.int64)
        lengths = np.zeros(len(sequences), dtype=np.int64)

        for idx, seq in enumerate(sequences):
            lengths[idx] = len(seq)
            for pos, token in enumerate(seq[:max_len]):
                encoded[idx, pos] = self._encode_token(token)

        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    def _encode_token(self, token: int) -> int:
        encoded = token + self._token_offset
        return encoded if encoded > self._pad_idx else self._pad_idx

    def _first_pad_index(self, encoded_sequence: List[int]) -> int:
        for idx, token in enumerate(encoded_sequence):
            if token == self._pad_idx:
                return idx
        return len(encoded_sequence)


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

    elif config.family == ModelFamily.LSTM:
        return LSTMSequenceModel(
            model_name=config.name or "lstm",
            embedding_dim=hp.get("embedding_dim", 32),
            hidden_size=hp.get("hidden_size", 64),
            epochs=hp.get("epochs", 60),
            batch_size=hp.get("batch_size", 64),
            learning_rate=hp.get("learning_rate", 0.01),
            random_state=hp.get("random_state", 42),
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
        if len(train_inputs) != len(train_outputs):
            raise ValueError(
                f"train_inputs length ({len(train_inputs)}) != "
                f"train_outputs length ({len(train_outputs)})"
            )
        if len(test_inputs) != len(test_outputs):
            raise ValueError(
                f"test_inputs length ({len(test_inputs)}) != "
                f"test_outputs length ({len(test_outputs)})"
            )

        # Stringify outputs for uniform handling
        train_labels_str = [str(o) for o in train_outputs]
        test_labels_str = [str(o) for o in test_outputs]

        if self._config.family == ModelFamily.LSTM:
            self._model.fit(train_inputs, train_outputs)

            if len(test_inputs) == 0:
                return PredictionResult(
                    model_name=self._model.name(),
                    predictions=[],
                    true_labels=test_labels_str,
                    train_size=len(train_inputs),
                    test_size=0,
                )

            raw_predictions = self._model.predict(test_inputs)
            predictions = [str(prediction) for prediction in raw_predictions]
            return PredictionResult(
                model_name=self._model.name(),
                predictions=predictions,
                true_labels=test_labels_str,
                train_size=len(train_inputs),
                test_size=len(test_inputs),
            )

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
