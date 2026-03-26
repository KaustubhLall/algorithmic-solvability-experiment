"""SR-2: Input Schema System.

Typed schemas that describe input formats for sequence and classification tasks.
Each schema can sample valid inputs deterministically given a seed.

Used by: SR-1 (Task Registry), SR-3 (Data Generator), SR-5 (Model Harness).
Validated by: V-2 (Input Schema Validation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Distribution(str, Enum):
    """Supported sampling distributions."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    WEIGHTED = "weighted"


class ElementType(str, Enum):
    """Element types for sequence inputs."""
    INT = "int"
    BINARY = "binary"
    CHAR = "char"


# ---------------------------------------------------------------------------
# Feature specs (building blocks for TabularInputSchema)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericalFeatureSpec:
    """Specification for a single numerical feature.

    Attributes:
        name: Feature name (unique within a schema).
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
        distribution: Sampling distribution.
    """
    name: str
    min_val: float
    max_val: float
    distribution: Distribution = Distribution.UNIFORM

    def __post_init__(self) -> None:
        if self.min_val > self.max_val:
            raise ValueError(
                f"NumericalFeatureSpec '{self.name}': "
                f"min_val ({self.min_val}) > max_val ({self.max_val})"
            )
        if self.distribution not in (
            Distribution.UNIFORM,
            Distribution.NORMAL,
            Distribution.EXPONENTIAL,
        ):
            raise ValueError(
                f"NumericalFeatureSpec '{self.name}': unsupported distribution {self.distribution}"
            )

    @property
    def is_numerical(self) -> bool:
        return True

    @property
    def is_categorical(self) -> bool:
        return False

    @property
    def expected_type(self) -> tuple[type, ...]:
        return (int, float)

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a single value from this feature's distribution."""
        if self.distribution == Distribution.UNIFORM:
            return float(rng.uniform(self.min_val, self.max_val))
        elif self.distribution == Distribution.NORMAL:
            # Mean at center, std = range/6 so ~99.7% falls in range; clip to bounds
            mean = (self.min_val + self.max_val) / 2.0
            std = (self.max_val - self.min_val) / 6.0
            val = float(rng.normal(mean, std)) if std > 0 else mean
            return float(np.clip(val, self.min_val, self.max_val))
        elif self.distribution == Distribution.EXPONENTIAL:
            # Exponential scaled to [min_val, max_val]
            # Lambda chosen so that mean is at 1/3 of range (right-skewed)
            range_size = self.max_val - self.min_val
            if range_size == 0:
                return self.min_val
            lam = 3.0 / range_size
            val = float(rng.exponential(1.0 / lam))
            return float(np.clip(val + self.min_val, self.min_val, self.max_val))
        else:
            raise ValueError(f"Unsupported distribution for numerical feature: {self.distribution}")


@dataclass(frozen=True)
class CategoricalFeatureSpec:
    """Specification for a single categorical feature.

    Attributes:
        name: Feature name (unique within a schema).
        values: Tuple of possible categorical values.
        distribution: "uniform" or "weighted".
        weights: Tuple of probability weights (must sum to 1 if provided).
            Required when distribution is WEIGHTED.
    """
    name: str
    values: Tuple[str, ...]
    distribution: Distribution = Distribution.UNIFORM
    weights: Optional[Tuple[float, ...]] = None

    def __post_init__(self) -> None:
        if len(self.values) == 0:
            raise ValueError(f"CategoricalFeatureSpec '{self.name}': values must not be empty")
        if len(self.values) != len(set(self.values)):
            raise ValueError(f"CategoricalFeatureSpec '{self.name}': values must be unique")
        if self.distribution == Distribution.WEIGHTED:
            if self.weights is None:
                raise ValueError(
                    f"CategoricalFeatureSpec '{self.name}': weights required for WEIGHTED distribution"
                )
            if len(self.weights) != len(self.values):
                raise ValueError(
                    f"CategoricalFeatureSpec '{self.name}': "
                    f"weights length ({len(self.weights)}) != values length ({len(self.values)})"
                )
            if any(not math.isfinite(weight) for weight in self.weights):
                raise ValueError(
                    f"CategoricalFeatureSpec '{self.name}': weights must be finite"
                )
            if any(weight < 0 for weight in self.weights):
                raise ValueError(
                    f"CategoricalFeatureSpec '{self.name}': weights must be non-negative"
                )
            if not math.isclose(sum(self.weights), 1.0, rel_tol=1e-6):
                raise ValueError(
                    f"CategoricalFeatureSpec '{self.name}': weights must sum to 1.0, got {sum(self.weights)}"
                )
        if self.distribution not in (Distribution.UNIFORM, Distribution.WEIGHTED):
            raise ValueError(
                f"CategoricalFeatureSpec '{self.name}': "
                f"distribution must be UNIFORM or WEIGHTED, got {self.distribution}"
            )

    @property
    def is_numerical(self) -> bool:
        return False

    @property
    def is_categorical(self) -> bool:
        return True

    @property
    def expected_type(self) -> type:
        return str

    @property
    def cardinality(self) -> int:
        return len(self.values)

    def sample(self, rng: np.random.Generator) -> str:
        """Sample a single value from this feature's distribution."""
        if self.distribution == Distribution.UNIFORM:
            idx = int(rng.integers(0, len(self.values)))
            return self.values[idx]
        elif self.distribution == Distribution.WEIGHTED:
            if self.weights is None:
                raise ValueError(
                    f"CategoricalFeatureSpec '{self.name}': weights required for WEIGHTED distribution"
                )
            idx = int(rng.choice(len(self.values), p=list(self.weights)))
            return self.values[idx]
        else:
            raise ValueError(f"Unsupported distribution for categorical feature: {self.distribution}")


# Union type for any feature spec
FeatureSpec = Union[NumericalFeatureSpec, CategoricalFeatureSpec]


# ---------------------------------------------------------------------------
# SequenceInputSchema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequenceInputSchema:
    """Schema for sequence-track task inputs.

    Describes variable-length sequences of typed elements.

    Attributes:
        element_type: Type of each element ("int", "binary", "char").
        min_length: Minimum sequence length (inclusive, >= 1).
        max_length: Maximum sequence length (inclusive).
        value_range: (low, high) inclusive range for int/binary elements.
        alphabet: Tuple of valid characters/tokens for char-type elements.
    """
    element_type: ElementType
    min_length: int
    max_length: int
    value_range: Tuple[int, int] = (0, 9)
    alphabet: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if self.min_length < 1:
            raise ValueError(f"min_length must be >= 1, got {self.min_length}")
        if self.min_length > self.max_length:
            raise ValueError(
                f"min_length ({self.min_length}) > max_length ({self.max_length})"
            )
        if self.element_type == ElementType.CHAR and self.alphabet is None:
            raise ValueError("alphabet is required for char element_type")
        if self.element_type == ElementType.BINARY and self.value_range != (0, 1):
            raise ValueError(
                f"binary element_type requires value_range=(0, 1), got {self.value_range}"
            )
        if self.value_range[0] > self.value_range[1]:
            raise ValueError(
                f"value_range[0] ({self.value_range[0]}) > value_range[1] ({self.value_range[1]})"
            )

    def sample(self, seed: int) -> List[Any]:
        """Sample a single valid input sequence.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            A list of elements conforming to this schema.
        """
        rng = np.random.default_rng(seed)
        length = int(rng.integers(self.min_length, self.max_length + 1))
        return self._sample_with_rng(rng, length)

    def sample_with_length(self, seed: int, length: int) -> List[Any]:
        """Sample a sequence of a specific length.

        Args:
            seed: Random seed for reproducibility.
            length: Exact length of the sequence.

        Returns:
            A list of elements of the given length.
        """
        if length < self.min_length or length > self.max_length:
            raise ValueError(
                f"Requested length {length} outside schema range "
                f"[{self.min_length}, {self.max_length}]"
            )
        rng = np.random.default_rng(seed)
        return self._sample_with_rng(rng, length)

    def sample_batch(self, seed: int, n: int) -> List[List[Any]]:
        """Sample n input sequences.

        Args:
            seed: Random seed for reproducibility.
            n: Number of sequences to sample.

        Returns:
            List of n sequences.
        """
        rng = np.random.default_rng(seed)
        results = []
        for _ in range(n):
            length = int(rng.integers(self.min_length, self.max_length + 1))
            results.append(self._sample_with_rng(rng, length))
        return results

    def _sample_with_rng(self, rng: np.random.Generator, length: int) -> List[Any]:
        """Internal: sample a sequence using an existing RNG."""
        if self.element_type == ElementType.INT:
            return [int(x) for x in rng.integers(self.value_range[0], self.value_range[1] + 1, size=length)]
        elif self.element_type == ElementType.BINARY:
            return [int(x) for x in rng.integers(0, 2, size=length)]
        elif self.element_type == ElementType.CHAR:
            if self.alphabet is None:
                raise ValueError("alphabet is required for char element_type")
            indices = rng.integers(0, len(self.alphabet), size=length)
            return [self.alphabet[int(i)] for i in indices]
        else:
            raise ValueError(f"Unsupported element_type: {self.element_type}")

    def validate_input(self, inp: List[Any]) -> bool:
        """Check whether an input conforms to this schema.

        Args:
            inp: Input to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(inp, list):
            return False
        if not (self.min_length <= len(inp) <= self.max_length):
            return False
        for elem in inp:
            if self.element_type == ElementType.INT:
                if not isinstance(elem, int):
                    return False
                if not (self.value_range[0] <= elem <= self.value_range[1]):
                    return False
            elif self.element_type == ElementType.BINARY:
                if elem not in (0, 1):
                    return False
            elif self.element_type == ElementType.CHAR:
                if self.alphabet is None:
                    return False
                if elem not in self.alphabet:
                    return False
        return True

    def features(self):
        """Yield (name, spec) pairs for compatibility with generic iteration.

        For sequence schemas this is a no-op since there are no named features.
        """
        return iter([])


# ---------------------------------------------------------------------------
# TabularInputSchema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TabularInputSchema:
    """Schema for classification-track task inputs.

    Describes fixed-width rows with named numerical and categorical features,
    plus optional irrelevant (distractor) features.

    Attributes:
        numerical_features: Specs for numerical features.
        categorical_features: Specs for categorical features.
        irrelevant_features: Specs for distractor features (can be numerical or categorical).
    """
    numerical_features: Tuple[NumericalFeatureSpec, ...] = ()
    categorical_features: Tuple[CategoricalFeatureSpec, ...] = ()
    irrelevant_features: Tuple[FeatureSpec, ...] = ()

    def __post_init__(self) -> None:
        # Check for duplicate feature names
        all_names = [f.name for f in self.all_feature_specs]
        if len(all_names) != len(set(all_names)):
            seen = set()
            for n in all_names:
                if n in seen:
                    raise ValueError(f"Duplicate feature name: '{n}'")
                seen.add(n)

    @property
    def all_feature_specs(self) -> List[FeatureSpec]:
        """All feature specs in order: numerical, categorical, irrelevant."""
        specs: List[FeatureSpec] = []
        specs.extend(self.numerical_features)
        specs.extend(self.categorical_features)
        specs.extend(self.irrelevant_features)
        return specs

    @property
    def relevant_feature_specs(self) -> List[FeatureSpec]:
        """Only the non-distractor feature specs."""
        specs: List[FeatureSpec] = []
        specs.extend(self.numerical_features)
        specs.extend(self.categorical_features)
        return specs

    @property
    def n_features(self) -> int:
        """Total number of features (including irrelevant)."""
        return len(self.numerical_features) + len(self.categorical_features) + len(self.irrelevant_features)

    @property
    def n_relevant_features(self) -> int:
        """Number of non-distractor features."""
        return len(self.numerical_features) + len(self.categorical_features)

    def sample(self, seed: int) -> Dict[str, Any]:
        """Sample a single valid input row.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Dict mapping feature names to sampled values.
        """
        rng = np.random.default_rng(seed)
        return self._sample_with_rng(rng)

    def sample_batch(self, seed: int, n: int) -> List[Dict[str, Any]]:
        """Sample n input rows.

        Args:
            seed: Random seed for reproducibility.
            n: Number of rows to sample.

        Returns:
            List of n dicts.
        """
        rng = np.random.default_rng(seed)
        return [self._sample_with_rng(rng) for _ in range(n)]

    def _sample_with_rng(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Internal: sample a row using an existing RNG."""
        row: Dict[str, Any] = {}
        for spec in self.all_feature_specs:
            row[spec.name] = spec.sample(rng)
        return row

    def validate_input(self, inp: Dict[str, Any]) -> bool:
        """Check whether an input dict conforms to this schema.

        Args:
            inp: Input dict to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(inp, dict):
            return False
        for spec in self.all_feature_specs:
            if spec.name not in inp:
                return False
            val = inp[spec.name]
            if spec.is_numerical:
                if not isinstance(val, spec.expected_type):
                    return False
                if not isinstance(spec, NumericalFeatureSpec):
                    return False
                if not (spec.min_val <= float(val) <= spec.max_val):
                    return False
            elif spec.is_categorical:
                if not isinstance(val, str):
                    return False
                if not isinstance(spec, CategoricalFeatureSpec):
                    return False
                if val not in spec.values:
                    return False
        return True

    def features(self):
        """Yield (name, spec) pairs for all features."""
        for spec in self.all_feature_specs:
            yield spec.name, spec

    def with_extra_irrelevant(self, extra: Sequence[FeatureSpec]) -> "TabularInputSchema":
        """Return a new schema with additional irrelevant features appended.

        Used by DistractorSplit to inject extra features at test time.

        Args:
            extra: Additional distractor feature specs to add.

        Returns:
            New TabularInputSchema with the extra irrelevant features.
        """
        return TabularInputSchema(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            irrelevant_features=tuple(self.irrelevant_features) + tuple(extra),
        )


# ---------------------------------------------------------------------------
# InputSchema union type (for type annotations in downstream modules)
# ---------------------------------------------------------------------------

InputSchema = Union[SequenceInputSchema, TabularInputSchema]
