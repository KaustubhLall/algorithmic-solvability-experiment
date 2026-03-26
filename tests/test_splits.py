"""V-4: Split Generator Validation Tests.

Tests for SR-4 (Split Generator) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. IID split: correct sizes, no overlap, reproducibility.
2. Length extrapolation: train has short, test has long.
3. Value extrapolation: train has narrow range, test has wider.
4. Noise split: train is clean, test has noise, labels unchanged.
5. Edge cases: empty splits, boundary values.
"""

from __future__ import annotations

import pytest

from src.data_generator import Dataset, Sample, generate_dataset
from src.registry import build_default_registry
from src.splits import (
    SplitGenerator,
    SplitResult,
    SplitStrategy,
    split_iid,
    split_length,
    split_noise,
    split_value,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture(scope="module")
def sort_dataset(registry):
    task = registry.get("S1.2_sort")
    return generate_dataset(task, n_samples=200, base_seed=42)


@pytest.fixture(scope="module")
def threshold_dataset(registry):
    task = registry.get("C1.1_numeric_threshold")
    return generate_dataset(task, n_samples=200, base_seed=42)


@pytest.fixture(scope="module")
def categorical_match_dataset(registry):
    task = registry.get("C1.3_categorical_match")
    return generate_dataset(task, n_samples=200, base_seed=42)


# ===================================================================
# 1. IID Split
# ===================================================================

class TestIIDSplit:

    def test_correct_sizes(self, sort_dataset):
        result = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        assert result.train_size + result.test_size == len(sort_dataset)
        assert result.train_size == 160
        assert result.test_size == 40

    def test_no_overlap(self, sort_dataset):
        result = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        train_seeds = {s.seed for s in result.train}
        test_seeds = {s.seed for s in result.test}
        assert train_seeds.isdisjoint(test_seeds)

    def test_all_samples_present(self, sort_dataset):
        result = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        all_seeds = {s.seed for s in result.train} | {s.seed for s in result.test}
        expected_seeds = {s.seed for s in sort_dataset.samples}
        assert all_seeds == expected_seeds

    def test_reproducible(self, sort_dataset):
        r1 = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        r2 = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        assert [s.seed for s in r1.train] == [s.seed for s in r2.train]
        assert [s.seed for s in r1.test] == [s.seed for s in r2.test]

    def test_different_seed_different_split(self, sort_dataset):
        r1 = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        r2 = split_iid(sort_dataset, train_fraction=0.8, seed=99)
        train1 = [s.seed for s in r1.train]
        train2 = [s.seed for s in r2.train]
        assert train1 != train2

    def test_strategy_is_iid(self, sort_dataset):
        result = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        assert result.strategy == SplitStrategy.IID

    def test_invalid_fraction_raises(self, sort_dataset):
        with pytest.raises(ValueError, match="train_fraction"):
            split_iid(sort_dataset, train_fraction=0.0)
        with pytest.raises(ValueError, match="train_fraction"):
            split_iid(sort_dataset, train_fraction=1.0)

    def test_metadata(self, sort_dataset):
        result = split_iid(sort_dataset, train_fraction=0.7, seed=99)
        assert result.split_metadata["train_fraction"] == 0.7
        assert result.split_metadata["seed"] == 99

    def test_accessors(self, sort_dataset):
        result = split_iid(sort_dataset, train_fraction=0.8, seed=42)
        assert len(result.train_inputs) == result.train_size
        assert len(result.train_outputs) == result.train_size
        assert len(result.test_inputs) == result.test_size
        assert len(result.test_outputs) == result.test_size

    def test_tabular_iid(self, threshold_dataset):
        result = split_iid(threshold_dataset, train_fraction=0.8, seed=42)
        assert result.train_size + result.test_size == len(threshold_dataset)


# ===================================================================
# 2. Length Extrapolation
# ===================================================================

class TestLengthExtrapolation:

    def test_short_train_long_test(self, sort_dataset):
        result = split_length(sort_dataset, length_threshold=8)
        for s in result.train:
            assert len(s.input_data) <= 8
        for s in result.test:
            assert len(s.input_data) > 8

    def test_all_samples_present(self, sort_dataset):
        result = split_length(sort_dataset, length_threshold=8)
        total = result.train_size + result.test_size
        assert total == len(sort_dataset)

    def test_strategy_is_length(self, sort_dataset):
        result = split_length(sort_dataset, length_threshold=8)
        assert result.strategy == SplitStrategy.LENGTH_EXTRAPOLATION

    def test_metadata(self, sort_dataset):
        result = split_length(sort_dataset, length_threshold=7)
        assert result.split_metadata["length_threshold"] == 7

    def test_extreme_threshold_all_train(self, sort_dataset):
        result = split_length(sort_dataset, length_threshold=100)
        assert result.train_size == len(sort_dataset)
        assert result.test_size == 0

    def test_extreme_threshold_all_test(self, sort_dataset):
        result = split_length(sort_dataset, length_threshold=0)
        assert result.train_size == 0
        assert result.test_size == len(sort_dataset)


# ===================================================================
# 3. Value Extrapolation
# ===================================================================

class TestValueExtrapolation:

    def test_tabular_value_split(self, threshold_dataset):
        result = split_value(threshold_dataset, feature_name="x1", train_range=(20.0, 80.0))
        for s in result.train:
            assert 20.0 <= s.input_data["x1"] <= 80.0
        for s in result.test:
            assert s.input_data["x1"] < 20.0 or s.input_data["x1"] > 80.0

    def test_all_samples_present(self, threshold_dataset):
        result = split_value(threshold_dataset, feature_name="x1", train_range=(20.0, 80.0))
        assert result.train_size + result.test_size == len(threshold_dataset)

    def test_strategy_is_value(self, threshold_dataset):
        result = split_value(threshold_dataset, feature_name="x1", train_range=(20.0, 80.0))
        assert result.strategy == SplitStrategy.VALUE_EXTRAPOLATION

    def test_sequence_value_split(self, sort_dataset):
        result = split_value(sort_dataset, feature_name="", train_range=(2, 7))
        for s in result.train:
            assert all(2 <= x <= 7 for x in s.input_data)

    def test_metadata(self, threshold_dataset):
        result = split_value(threshold_dataset, feature_name="x1", train_range=(30.0, 70.0))
        assert result.split_metadata["feature_name"] == "x1"
        assert result.split_metadata["train_range"] == (30.0, 70.0)

    def test_non_numeric_train_range_raises(self, threshold_dataset):
        with pytest.raises(ValueError, match="numeric bounds"):
            split_value(threshold_dataset, feature_name="x1", train_range=("low", 70.0))

    def test_inverted_train_range_raises(self, threshold_dataset):
        with pytest.raises(ValueError, match="lower bound"):
            split_value(threshold_dataset, feature_name="x1", train_range=(80.0, 20.0))

    def test_non_numeric_sequence_values_go_to_test(self):
        dataset = Dataset(
            task_id="synthetic",
            samples=[
                Sample(input_data=[1, 2, 3], output_data=[1], task_id="synthetic", seed=0),
                Sample(input_data=[1, "oops", 3], output_data=[1], task_id="synthetic", seed=1),
            ],
        )
        result = split_value(dataset, feature_name="", train_range=(0, 5))
        assert [sample.seed for sample in result.train] == [0]
        assert [sample.seed for sample in result.test] == [1]


# ===================================================================
# 4. Noise Split
# ===================================================================

class TestNoiseSplit:

    def test_train_is_clean(self, sort_dataset, registry):
        task = registry.get("S1.2_sort")
        result = split_noise(sort_dataset, train_fraction=0.8, test_noise_level=0.5, seed=42)
        for s in result.train:
            clean = task.input_sampler(s.seed)
            assert s.input_data == clean

    def test_test_has_noise(self, sort_dataset, registry):
        task = registry.get("S1.2_sort")
        result = split_noise(sort_dataset, train_fraction=0.8, test_noise_level=0.5, seed=42)
        n_modified = 0
        for s in result.test:
            clean = task.input_sampler(s.seed)
            if s.input_data != clean:
                n_modified += 1
        assert n_modified > 0, "Test set should have some noisy inputs"

    def test_labels_unchanged(self, sort_dataset, registry):
        task = registry.get("S1.2_sort")
        result = split_noise(sort_dataset, train_fraction=0.8, test_noise_level=0.5, seed=42)
        for s in result.test:
            clean = task.input_sampler(s.seed)
            expected = task.reference_algorithm(clean)
            assert s.output_data == expected

    def test_strategy_is_noise(self, sort_dataset):
        result = split_noise(sort_dataset, test_noise_level=0.1, seed=42)
        assert result.strategy == SplitStrategy.NOISE

    def test_noise_metadata(self, sort_dataset):
        result = split_noise(sort_dataset, test_noise_level=0.3, seed=42)
        assert result.split_metadata["test_noise_level"] == 0.3

    def test_zero_noise_is_clean(self, sort_dataset, registry):
        task = registry.get("S1.2_sort")
        result = split_noise(sort_dataset, test_noise_level=0.0, seed=42)
        for s in result.test:
            clean = task.input_sampler(s.seed)
            assert s.input_data == clean

    def test_invalid_noise_level_raises(self, sort_dataset):
        with pytest.raises(ValueError, match="test_noise_level"):
            split_noise(sort_dataset, test_noise_level=-0.1, seed=42)
        with pytest.raises(ValueError, match="test_noise_level"):
            split_noise(sort_dataset, test_noise_level=1.1, seed=42)


# ===================================================================
# 5. Split on Classification Data
# ===================================================================

class TestClassificationSplits:

    def test_iid_preserves_labels(self, threshold_dataset, registry):
        task = registry.get("C1.1_numeric_threshold")
        result = split_iid(threshold_dataset, train_fraction=0.8, seed=42)
        for s in result.train + result.test:
            clean = task.input_sampler(s.seed)
            expected = task.reference_algorithm(clean)
            assert s.output_data == expected

    def test_noise_on_tabular(self, threshold_dataset, registry):
        task = registry.get("C1.1_numeric_threshold")
        result = split_noise(threshold_dataset, test_noise_level=0.3, seed=42)
        # Labels should still reference clean inputs
        for s in result.test:
            clean = task.input_sampler(s.seed)
            expected = task.reference_algorithm(clean)
            assert s.output_data == expected

    def test_noise_on_categorical_tabular_uses_schema_values(
        self,
        categorical_match_dataset,
        registry,
    ):
        task = registry.get("C1.3_categorical_match")
        cat1_spec = next(
            feature
            for feature in task.input_schema.categorical_features
            if feature.name == "cat1"
        )
        allowed_values = set(cat1_spec.values)
        result = split_noise(
            categorical_match_dataset,
            test_noise_level=1.0,
            seed=42,
            schema=task.input_schema,
        )

        modified = 0
        for s in result.test:
            clean = task.input_sampler(s.seed)
            expected = task.reference_algorithm(clean)
            assert s.output_data == expected
            assert s.input_data["cat1"] in allowed_values
            if s.input_data != clean:
                modified += 1

        assert modified > 0
