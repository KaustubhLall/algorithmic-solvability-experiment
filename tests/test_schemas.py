"""V-2: Input Schema Validation Tests.

Tests for SR-2 (Input Schema System) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. Schema completeness — every field has a defined type, range, and distribution.
2. Sampling validity — sampled inputs conform to the schema.
3. Reproducibility — same seed → same inputs.
4. Distribution check — empirical distribution matches specified distribution (p > 0.01).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from scipy import stats

from src.schemas import (
    CategoricalFeatureSpec,
    Distribution,
    ElementType,
    NumericalFeatureSpec,
    SequenceInputSchema,
    TabularInputSchema,
)


# ===================================================================
# Fixtures: reusable schema instances
# ===================================================================

@pytest.fixture
def int_sequence_schema() -> SequenceInputSchema:
    return SequenceInputSchema(
        element_type=ElementType.INT,
        min_length=3,
        max_length=10,
        value_range=(0, 9),
    )


@pytest.fixture
def binary_sequence_schema() -> SequenceInputSchema:
    return SequenceInputSchema(
        element_type=ElementType.BINARY,
        min_length=1,
        max_length=20,
        value_range=(0, 1),
    )


@pytest.fixture
def char_sequence_schema() -> SequenceInputSchema:
    return SequenceInputSchema(
        element_type=ElementType.CHAR,
        min_length=2,
        max_length=8,
        value_range=(0, 25),
        alphabet=tuple("abcdefghijklmnopqrstuvwxyz"),
    )


@pytest.fixture
def simple_tabular_schema() -> TabularInputSchema:
    return TabularInputSchema(
        numerical_features=(
            NumericalFeatureSpec(name="age", min_val=0.0, max_val=100.0, distribution=Distribution.UNIFORM),
            NumericalFeatureSpec(name="income", min_val=0.0, max_val=200000.0, distribution=Distribution.NORMAL),
        ),
        categorical_features=(
            CategoricalFeatureSpec(name="color", values=("red", "green", "blue"), distribution=Distribution.UNIFORM),
            CategoricalFeatureSpec(
                name="size",
                values=("S", "M", "L"),
                distribution=Distribution.WEIGHTED,
                weights=(0.2, 0.5, 0.3),
            ),
        ),
    )


@pytest.fixture
def tabular_with_distractors() -> TabularInputSchema:
    return TabularInputSchema(
        numerical_features=(
            NumericalFeatureSpec(name="x1", min_val=-10.0, max_val=10.0),
        ),
        categorical_features=(
            CategoricalFeatureSpec(name="cat1", values=("A", "B")),
        ),
        irrelevant_features=(
            NumericalFeatureSpec(name="noise1", min_val=0.0, max_val=1.0),
            CategoricalFeatureSpec(name="noise2", values=("X", "Y", "Z")),
        ),
    )


# ===================================================================
# 1. Construction & Validation Checks
# ===================================================================

class TestConstructionValidation:
    """Test that schemas reject invalid configurations at construction time."""

    def test_numerical_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="min_val.*max_val"):
            NumericalFeatureSpec(name="bad", min_val=10.0, max_val=5.0)

    def test_categorical_empty_values_raises(self):
        with pytest.raises(ValueError, match="values must not be empty"):
            CategoricalFeatureSpec(name="bad", values=())

    def test_categorical_duplicate_values_raises(self):
        with pytest.raises(ValueError, match="values must be unique"):
            CategoricalFeatureSpec(name="bad", values=("a", "a"))

    def test_categorical_weighted_no_weights_raises(self):
        with pytest.raises(ValueError, match="weights required"):
            CategoricalFeatureSpec(name="bad", values=("a", "b"), distribution=Distribution.WEIGHTED)

    def test_categorical_weights_wrong_length_raises(self):
        with pytest.raises(ValueError, match="weights length"):
            CategoricalFeatureSpec(name="bad", values=("a", "b"), distribution=Distribution.WEIGHTED, weights=(0.5,))

    def test_categorical_weights_not_sum_1_raises(self):
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            CategoricalFeatureSpec(name="bad", values=("a", "b"), distribution=Distribution.WEIGHTED, weights=(0.3, 0.3))

    def test_categorical_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            CategoricalFeatureSpec(
                name="bad",
                values=("a", "b"),
                distribution=Distribution.WEIGHTED,
                weights=(1.1, -0.1),
            )

    def test_categorical_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="UNIFORM or WEIGHTED"):
            CategoricalFeatureSpec(name="bad", values=("a", "b"), distribution=Distribution.NORMAL)

    def test_numerical_invalid_distribution_raises(self):
        with pytest.raises(ValueError, match="unsupported distribution"):
            NumericalFeatureSpec(name="bad", min_val=0.0, max_val=1.0, distribution=Distribution.WEIGHTED)

    def test_sequence_min_length_zero_raises(self):
        with pytest.raises(ValueError, match="min_length must be >= 1"):
            SequenceInputSchema(element_type=ElementType.INT, min_length=0, max_length=5)

    def test_sequence_min_gt_max_length_raises(self):
        with pytest.raises(ValueError, match="min_length.*max_length"):
            SequenceInputSchema(element_type=ElementType.INT, min_length=10, max_length=5)

    def test_sequence_char_no_alphabet_raises(self):
        with pytest.raises(ValueError, match="alphabet is required"):
            SequenceInputSchema(element_type=ElementType.CHAR, min_length=1, max_length=5)

    def test_sequence_binary_wrong_range_raises(self):
        with pytest.raises(ValueError, match="binary.*value_range"):
            SequenceInputSchema(element_type=ElementType.BINARY, min_length=1, max_length=5, value_range=(0, 5))

    def test_sequence_inverted_value_range_raises(self):
        with pytest.raises(ValueError, match="value_range"):
            SequenceInputSchema(element_type=ElementType.INT, min_length=1, max_length=5, value_range=(10, 5))

    def test_tabular_duplicate_feature_names_raises(self):
        with pytest.raises(ValueError, match="Duplicate feature name"):
            TabularInputSchema(
                numerical_features=(
                    NumericalFeatureSpec(name="x", min_val=0.0, max_val=1.0),
                    NumericalFeatureSpec(name="x", min_val=0.0, max_val=1.0),
                ),
            )

    def test_tabular_duplicate_across_types_raises(self):
        with pytest.raises(ValueError, match="Duplicate feature name"):
            TabularInputSchema(
                numerical_features=(NumericalFeatureSpec(name="x", min_val=0.0, max_val=1.0),),
                categorical_features=(CategoricalFeatureSpec(name="x", values=("a", "b")),),
            )


# ===================================================================
# 2. Sampling Validity — Type & Range Checks
# ===================================================================

class TestSequenceSamplingValidity:
    """Sampled sequence inputs conform to the schema."""

    N_SAMPLES = 500

    def test_int_sequence_values_in_range(self, int_sequence_schema: SequenceInputSchema):
        for i in range(self.N_SAMPLES):
            seq = int_sequence_schema.sample(seed=i)
            assert int_sequence_schema.validate_input(seq), f"Invalid sample at seed={i}: {seq}"

    def test_int_sequence_types(self, int_sequence_schema: SequenceInputSchema):
        seq = int_sequence_schema.sample(seed=0)
        assert isinstance(seq, list)
        for elem in seq:
            assert isinstance(elem, int)

    def test_int_sequence_length_bounds(self, int_sequence_schema: SequenceInputSchema):
        for i in range(self.N_SAMPLES):
            seq = int_sequence_schema.sample(seed=i)
            assert 3 <= len(seq) <= 10, f"Length {len(seq)} outside [3, 10] at seed={i}"

    def test_binary_sequence_values(self, binary_sequence_schema: SequenceInputSchema):
        for i in range(self.N_SAMPLES):
            seq = binary_sequence_schema.sample(seed=i)
            assert all(elem in (0, 1) for elem in seq), f"Non-binary value at seed={i}: {seq}"

    def test_char_sequence_values(self, char_sequence_schema: SequenceInputSchema):
        for i in range(self.N_SAMPLES):
            seq = char_sequence_schema.sample(seed=i)
            assert all(isinstance(c, str) and c in "abcdefghijklmnopqrstuvwxyz" for c in seq)

    def test_sample_with_length(self, int_sequence_schema: SequenceInputSchema):
        for length in range(3, 11):
            seq = int_sequence_schema.sample_with_length(seed=42, length=length)
            assert len(seq) == length

    def test_sample_with_length_out_of_range_raises(self, int_sequence_schema: SequenceInputSchema):
        with pytest.raises(ValueError, match="outside schema range"):
            int_sequence_schema.sample_with_length(seed=42, length=100)

    def test_batch_sampling(self, int_sequence_schema: SequenceInputSchema):
        batch = int_sequence_schema.sample_batch(seed=42, n=100)
        assert len(batch) == 100
        for seq in batch:
            assert int_sequence_schema.validate_input(seq)


class TestTabularSamplingValidity:
    """Sampled tabular inputs conform to the schema."""

    N_SAMPLES = 500

    def test_all_features_present(self, simple_tabular_schema: TabularInputSchema):
        row = simple_tabular_schema.sample(seed=0)
        assert "age" in row
        assert "income" in row
        assert "color" in row
        assert "size" in row

    def test_numerical_types_and_ranges(self, simple_tabular_schema: TabularInputSchema):
        for i in range(self.N_SAMPLES):
            row = simple_tabular_schema.sample(seed=i)
            assert isinstance(row["age"], float)
            assert 0.0 <= row["age"] <= 100.0
            assert isinstance(row["income"], float)
            assert 0.0 <= row["income"] <= 200000.0

    def test_categorical_values(self, simple_tabular_schema: TabularInputSchema):
        for i in range(self.N_SAMPLES):
            row = simple_tabular_schema.sample(seed=i)
            assert row["color"] in ("red", "green", "blue")
            assert row["size"] in ("S", "M", "L")

    def test_validate_input(self, simple_tabular_schema: TabularInputSchema):
        for i in range(self.N_SAMPLES):
            row = simple_tabular_schema.sample(seed=i)
            assert simple_tabular_schema.validate_input(row)

    def test_validate_rejects_missing_feature(self, simple_tabular_schema: TabularInputSchema):
        row = simple_tabular_schema.sample(seed=0)
        del row["age"]
        assert not simple_tabular_schema.validate_input(row)

    def test_validate_rejects_out_of_range(self, simple_tabular_schema: TabularInputSchema):
        row = simple_tabular_schema.sample(seed=0)
        row["age"] = 999.0
        assert not simple_tabular_schema.validate_input(row)

    def test_validate_rejects_wrong_category(self, simple_tabular_schema: TabularInputSchema):
        row = simple_tabular_schema.sample(seed=0)
        row["color"] = "purple"
        assert not simple_tabular_schema.validate_input(row)

    def test_irrelevant_features_sampled(self, tabular_with_distractors: TabularInputSchema):
        row = tabular_with_distractors.sample(seed=0)
        assert "noise1" in row
        assert "noise2" in row
        assert isinstance(row["noise1"], float)
        assert row["noise2"] in ("X", "Y", "Z")

    def test_batch_sampling(self, simple_tabular_schema: TabularInputSchema):
        batch = simple_tabular_schema.sample_batch(seed=42, n=100)
        assert len(batch) == 100
        for row in batch:
            assert simple_tabular_schema.validate_input(row)

    def test_n_features(self, tabular_with_distractors: TabularInputSchema):
        assert tabular_with_distractors.n_features == 4
        assert tabular_with_distractors.n_relevant_features == 2

    def test_features_iteration(self, simple_tabular_schema: TabularInputSchema):
        names = [name for name, _ in simple_tabular_schema.features()]
        assert names == ["age", "income", "color", "size"]


# ===================================================================
# 3. Reproducibility — same seed → same inputs
# ===================================================================

class TestReproducibility:
    """Same seed produces identical samples."""

    def test_sequence_reproducibility(self, int_sequence_schema: SequenceInputSchema):
        s1 = int_sequence_schema.sample(seed=42)
        s2 = int_sequence_schema.sample(seed=42)
        assert s1 == s2

    def test_sequence_different_seeds_differ(self, int_sequence_schema: SequenceInputSchema):
        s1 = int_sequence_schema.sample(seed=42)
        s2 = int_sequence_schema.sample(seed=43)
        # Very unlikely to be equal for different seeds
        assert s1 != s2

    def test_sequence_batch_reproducibility(self, int_sequence_schema: SequenceInputSchema):
        b1 = int_sequence_schema.sample_batch(seed=42, n=50)
        b2 = int_sequence_schema.sample_batch(seed=42, n=50)
        assert b1 == b2

    def test_tabular_reproducibility(self, simple_tabular_schema: TabularInputSchema):
        r1 = simple_tabular_schema.sample(seed=42)
        r2 = simple_tabular_schema.sample(seed=42)
        assert r1 == r2

    def test_tabular_different_seeds_differ(self, simple_tabular_schema: TabularInputSchema):
        r1 = simple_tabular_schema.sample(seed=42)
        r2 = simple_tabular_schema.sample(seed=43)
        assert r1 != r2

    def test_tabular_batch_reproducibility(self, simple_tabular_schema: TabularInputSchema):
        b1 = simple_tabular_schema.sample_batch(seed=42, n=50)
        b2 = simple_tabular_schema.sample_batch(seed=42, n=50)
        assert b1 == b2


# ===================================================================
# 4. Distribution Checks (KS for numerical, chi-squared for categorical)
# ===================================================================

class TestDistributions:
    """For large samples, empirical distribution matches specified distribution."""

    N_SAMPLES = 10000
    P_THRESHOLD = 0.01  # per V-2 acceptance criteria

    def test_uniform_numerical_distribution(self):
        spec = NumericalFeatureSpec(name="x", min_val=0.0, max_val=10.0, distribution=Distribution.UNIFORM)
        schema = TabularInputSchema(numerical_features=(spec,))
        samples = schema.sample_batch(seed=42, n=self.N_SAMPLES)
        values = [s["x"] for s in samples]

        # KS test against uniform(0, 10)
        ks_stat, p_val = stats.kstest(values, "uniform", args=(0.0, 10.0))
        assert p_val > self.P_THRESHOLD, f"Uniform KS test failed: p={p_val:.6f}"

    def test_normal_numerical_distribution(self):
        spec = NumericalFeatureSpec(name="x", min_val=-100.0, max_val=100.0, distribution=Distribution.NORMAL)
        schema = TabularInputSchema(numerical_features=(spec,))
        samples = schema.sample_batch(seed=42, n=self.N_SAMPLES)
        values = [s["x"] for s in samples]

        # For the normal distribution, mean should be near 0, std near ~33.3
        mean = np.mean(values)
        assert abs(mean) < 5.0, f"Normal mean too far from 0: {mean}"

    def test_exponential_numerical_distribution(self):
        spec = NumericalFeatureSpec(name="x", min_val=0.0, max_val=100.0, distribution=Distribution.EXPONENTIAL)
        schema = TabularInputSchema(numerical_features=(spec,))
        samples = schema.sample_batch(seed=42, n=self.N_SAMPLES)
        values = [s["x"] for s in samples]

        # Right-skewed: median < mean
        median = np.median(values)
        mean = np.mean(values)
        assert median < mean, f"Exponential not right-skewed: median={median}, mean={mean}"

    def test_uniform_categorical_distribution(self):
        spec = CategoricalFeatureSpec(name="c", values=("A", "B", "C", "D"), distribution=Distribution.UNIFORM)
        schema = TabularInputSchema(categorical_features=(spec,))
        samples = schema.sample_batch(seed=42, n=self.N_SAMPLES)
        values = [s["c"] for s in samples]

        counts = Counter(values)
        expected = self.N_SAMPLES / 4
        observed = [counts.get(v, 0) for v in ("A", "B", "C", "D")]

        # Chi-squared test
        chi2_stat, p_val = stats.chisquare(observed, [expected] * 4)
        assert p_val > self.P_THRESHOLD, f"Uniform categorical chi2 test failed: p={p_val:.6f}"

    def test_weighted_categorical_distribution(self):
        weights = (0.1, 0.2, 0.3, 0.4)
        spec = CategoricalFeatureSpec(
            name="c",
            values=("A", "B", "C", "D"),
            distribution=Distribution.WEIGHTED,
            weights=weights,
        )
        schema = TabularInputSchema(categorical_features=(spec,))
        samples = schema.sample_batch(seed=42, n=self.N_SAMPLES)
        values = [s["c"] for s in samples]

        counts = Counter(values)
        expected = [w * self.N_SAMPLES for w in weights]
        observed = [counts.get(v, 0) for v in ("A", "B", "C", "D")]

        chi2_stat, p_val = stats.chisquare(observed, expected)
        assert p_val > self.P_THRESHOLD, f"Weighted categorical chi2 test failed: p={p_val:.6f}"

    def test_sequence_length_distribution_uniform(self, int_sequence_schema: SequenceInputSchema):
        """Sequence lengths should be roughly uniform across [min_length, max_length]."""
        batch = int_sequence_schema.sample_batch(seed=42, n=self.N_SAMPLES)
        lengths = [len(s) for s in batch]

        n_bins = int_sequence_schema.max_length - int_sequence_schema.min_length + 1
        expected = self.N_SAMPLES / n_bins
        counts = Counter(lengths)
        observed = [counts.get(l, 0) for l in range(int_sequence_schema.min_length, int_sequence_schema.max_length + 1)]

        chi2_stat, p_val = stats.chisquare(observed, [expected] * n_bins)
        assert p_val > self.P_THRESHOLD, f"Length distribution chi2 test failed: p={p_val:.6f}"

    def test_sequence_int_value_distribution_uniform(self):
        """Integer element values should be roughly uniform across value_range."""
        schema = SequenceInputSchema(
            element_type=ElementType.INT,
            min_length=100,
            max_length=100,
            value_range=(0, 9),
        )
        batch = schema.sample_batch(seed=42, n=100)
        all_values = [elem for seq in batch for elem in seq]

        n_bins = 10  # values 0-9
        expected = len(all_values) / n_bins
        counts = Counter(all_values)
        observed = [counts.get(v, 0) for v in range(10)]

        chi2_stat, p_val = stats.chisquare(observed, [expected] * n_bins)
        assert p_val > self.P_THRESHOLD, f"Int value distribution chi2 test failed: p={p_val:.6f}"


# ===================================================================
# 5. Property Checks
# ===================================================================

class TestSchemaProperties:
    """Test derived properties and helper methods."""

    def test_numerical_feature_spec_properties(self):
        spec = NumericalFeatureSpec(name="x", min_val=0.0, max_val=1.0)
        assert spec.is_numerical is True
        assert spec.is_categorical is False
        assert spec.expected_type == (int, float)

    def test_categorical_feature_spec_properties(self):
        spec = CategoricalFeatureSpec(name="c", values=("a", "b", "c"))
        assert spec.is_numerical is False
        assert spec.is_categorical is True
        assert spec.expected_type is str
        assert spec.cardinality == 3

    def test_with_extra_irrelevant(self, simple_tabular_schema: TabularInputSchema):
        extra = [NumericalFeatureSpec(name="distractor1", min_val=0.0, max_val=1.0)]
        extended = simple_tabular_schema.with_extra_irrelevant(extra)
        assert extended.n_features == simple_tabular_schema.n_features + 1
        assert extended.n_relevant_features == simple_tabular_schema.n_relevant_features
        row = extended.sample(seed=0)
        assert "distractor1" in row

    def test_sequence_features_iteration_empty(self, int_sequence_schema: SequenceInputSchema):
        feats = list(int_sequence_schema.features())
        assert feats == []

    def test_equal_min_max_numerical(self):
        spec = NumericalFeatureSpec(name="const", min_val=5.0, max_val=5.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            assert spec.sample(rng) == 5.0

    def test_equal_min_max_sequence_length(self):
        schema = SequenceInputSchema(
            element_type=ElementType.INT,
            min_length=5,
            max_length=5,
            value_range=(0, 9),
        )
        for i in range(100):
            assert len(schema.sample(seed=i)) == 5
