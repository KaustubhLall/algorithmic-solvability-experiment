"""SR-4: Split Generator.

Creates train/test splits with controlled distribution shifts for evaluating
algorithmic generalization. Supports multiple split strategies:

- IID: standard random split
- Length extrapolation: train on short sequences, test on longer
- Value extrapolation: train on narrow value range, test on wider
- Composition extrapolation: train on simple compositions, test on deeper
- Distractor: add irrelevant features to test set only
- Noise: add input noise to test set only

Used by: SR-7 (Experiment Runner).
Validated by: V-4 (Split Generator Validation).
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data_generator import Dataset, Sample
from src.schemas import TabularInputSchema


# ===================================================================
# Split strategy enum
# ===================================================================

class SplitStrategy(str, Enum):
    IID = "iid"
    LENGTH_EXTRAPOLATION = "length_extrapolation"
    VALUE_EXTRAPOLATION = "value_extrapolation"
    DISTRACTOR = "distractor"
    NOISE = "noise"


# ===================================================================
# Split result
# ===================================================================

@dataclass
class SplitResult:
    """Result of splitting a dataset.

    Attributes:
        train: Training samples.
        test: Test samples.
        strategy: Which split strategy was used.
        split_metadata: Details about the split (e.g., split ratio, thresholds).
    """
    train: List[Sample]
    test: List[Sample]
    strategy: SplitStrategy
    split_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def train_size(self) -> int:
        return len(self.train)

    @property
    def test_size(self) -> int:
        return len(self.test)

    @property
    def train_inputs(self) -> List[Any]:
        return [s.input_data for s in self.train]

    @property
    def train_outputs(self) -> List[Any]:
        return [s.output_data for s in self.train]

    @property
    def test_inputs(self) -> List[Any]:
        return [s.input_data for s in self.test]

    @property
    def test_outputs(self) -> List[Any]:
        return [s.output_data for s in self.test]


# ===================================================================
# Split Generator
# ===================================================================

class SplitGenerator:
    """Generates train/test splits from datasets."""

    def split_iid(
        self,
        dataset: Dataset,
        train_fraction: float = 0.8,
        seed: int = 42,
    ) -> SplitResult:
        """Standard IID random split.

        Args:
            dataset: The dataset to split.
            train_fraction: Fraction of data for training.
            seed: Random seed for shuffling.

        Returns:
            SplitResult with IID train/test split.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

        rng = np.random.default_rng(seed)
        indices = list(range(len(dataset)))
        rng.shuffle(indices)

        split_idx = int(len(indices) * train_fraction)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        return SplitResult(
            train=[dataset.samples[i] for i in train_indices],
            test=[dataset.samples[i] for i in test_indices],
            strategy=SplitStrategy.IID,
            split_metadata={
                "train_fraction": train_fraction,
                "seed": seed,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
            },
        )

    def split_length_extrapolation(
        self,
        dataset: Dataset,
        length_threshold: int,
    ) -> SplitResult:
        """Split by sequence length: train on short, test on long.

        Requires sequence inputs (list type).

        Args:
            dataset: The dataset to split.
            length_threshold: Sequences with len <= threshold go to train,
                              len > threshold go to test.

        Returns:
            SplitResult with length-based split.
        """
        train, test = [], []
        for s in dataset.samples:
            if isinstance(s.input_data, list):
                if len(s.input_data) <= length_threshold:
                    train.append(s)
                else:
                    test.append(s)
            else:
                raise ValueError(
                    f"Length extrapolation requires list inputs, got {type(s.input_data)}"
                )

        return SplitResult(
            train=train,
            test=test,
            strategy=SplitStrategy.LENGTH_EXTRAPOLATION,
            split_metadata={
                "length_threshold": length_threshold,
                "train_size": len(train),
                "test_size": len(test),
            },
        )

    def split_value_extrapolation(
        self,
        dataset: Dataset,
        feature_name: str,
        train_range: Tuple[float, float],
    ) -> SplitResult:
        """Split by feature value range: train on narrow, test on wider.

        For tabular data: samples where feature is within train_range go to train.
        For sequence data: samples where all values are within train_range go to train.

        Args:
            dataset: The dataset to split.
            feature_name: Feature to split on (for tabular) or ignored (for sequence).
            train_range: (lo, hi) range for training data.

        Returns:
            SplitResult with value-based split.
        """
        lo, hi = train_range
        if not isinstance(lo, numbers.Real) or not isinstance(hi, numbers.Real):
            raise ValueError(
                f"train_range must contain numeric bounds, got {train_range}"
            )
        if lo > hi:
            raise ValueError(f"train_range lower bound ({lo}) > upper bound ({hi})")
        train, test = [], []

        for s in dataset.samples:
            if isinstance(s.input_data, dict):
                val = s.input_data.get(feature_name)
                if val is not None and isinstance(val, numbers.Real):
                    if lo <= val <= hi:
                        train.append(s)
                    else:
                        test.append(s)
                else:
                    train.append(s)  # non-numeric features go to train
            elif isinstance(s.input_data, list):
                if any(not isinstance(x, numbers.Real) for x in s.input_data):
                    test.append(s)
                elif all(lo <= x <= hi for x in s.input_data):
                    train.append(s)
                else:
                    test.append(s)
            else:
                train.append(s)

        return SplitResult(
            train=train,
            test=test,
            strategy=SplitStrategy.VALUE_EXTRAPOLATION,
            split_metadata={
                "feature_name": feature_name,
                "train_range": train_range,
                "train_size": len(train),
                "test_size": len(test),
            },
        )

    def split_with_noise(
        self,
        dataset: Dataset,
        train_fraction: float = 0.8,
        test_noise_level: float = 0.1,
        seed: int = 42,
        schema: Optional[TabularInputSchema] = None,
    ) -> SplitResult:
        """IID split where test inputs have added noise.

        Train data is clean. Test inputs are perturbed but labels remain correct
        (they reference the clean input).

        Args:
            dataset: The dataset to split.
            train_fraction: Fraction for training.
            test_noise_level: Noise level applied to test inputs.
            seed: Random seed.
            schema: Optional tabular schema used to sample valid categorical
                perturbations for tabular inputs.

        Returns:
            SplitResult with noisy test inputs.
        """
        if not 0.0 <= test_noise_level <= 1.0:
            raise ValueError(
                f"test_noise_level must be in [0, 1], got {test_noise_level}"
            )
        base_split = self.split_iid(dataset, train_fraction, seed)

        rng = np.random.default_rng(seed + 2**30)
        noisy_test = []
        for s in base_split.test:
            noisy_input = self._add_noise(
                s.input_data,
                test_noise_level,
                rng,
                schema=schema,
            )
            noisy_sample = Sample(
                input_data=noisy_input,
                output_data=s.output_data,
                task_id=s.task_id,
                seed=s.seed,
                metadata={**s.metadata, "noise_level": test_noise_level},
            )
            noisy_test.append(noisy_sample)

        return SplitResult(
            train=base_split.train,
            test=noisy_test,
            strategy=SplitStrategy.NOISE,
            split_metadata={
                "train_fraction": train_fraction,
                "test_noise_level": test_noise_level,
                "seed": seed,
                "train_size": base_split.train_size,
                "test_size": len(noisy_test),
            },
        )

    def _add_noise(
        self,
        inp: Any,
        noise_level: float,
        rng: np.random.Generator,
        schema: Optional[TabularInputSchema] = None,
    ) -> Any:
        """Add noise to an input."""
        if isinstance(inp, list):
            result = list(inp)
            for i in range(len(result)):
                if rng.random() < noise_level and isinstance(result[i], int):
                    result[i] = result[i] + int(rng.integers(-2, 3))
            return result
        elif isinstance(inp, dict):
            result = dict(inp)
            categorical_values: Dict[str, Tuple[str, ...]] = {}
            if isinstance(schema, TabularInputSchema):
                categorical_values = {
                    feature.name: feature.values
                    for feature in schema.categorical_features
                }
            for key, val in result.items():
                should_perturb = rng.random() < noise_level
                if should_perturb and isinstance(val, numbers.Real):
                    scale = max(abs(val) * 0.1, 1.0)
                    result[key] = float(val) + float(rng.normal(0, scale))
                elif (
                    should_perturb
                    and isinstance(val, str)
                    and key in categorical_values
                ):
                    alternatives = [
                        candidate
                        for candidate in categorical_values[key]
                        if candidate != val
                    ]
                    if alternatives:
                        result[key] = alternatives[int(rng.integers(0, len(alternatives)))]
            return result
        return inp


# ===================================================================
# Convenience functions
# ===================================================================

def split_iid(dataset: Dataset, train_fraction: float = 0.8, seed: int = 42) -> SplitResult:
    return SplitGenerator().split_iid(dataset, train_fraction, seed)


def split_length(dataset: Dataset, length_threshold: int) -> SplitResult:
    return SplitGenerator().split_length_extrapolation(dataset, length_threshold)


def split_value(dataset: Dataset, feature_name: str, train_range: Tuple[float, float]) -> SplitResult:
    return SplitGenerator().split_value_extrapolation(dataset, feature_name, train_range)


def split_noise(
    dataset: Dataset,
    train_fraction: float = 0.8,
    test_noise_level: float = 0.1,
    seed: int = 42,
    schema: Optional[TabularInputSchema] = None,
) -> SplitResult:
    return SplitGenerator().split_with_noise(
        dataset,
        train_fraction,
        test_noise_level,
        seed,
        schema=schema,
    )
