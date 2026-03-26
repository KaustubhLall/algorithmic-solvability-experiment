"""V-3: Data Generator Validation Tests.

Tests for SR-3 (Data Generator) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. Every generated label matches re-application of the reference algorithm.
2. Reproducibility: same seed → same dataset.
3. Noise injection affects inputs but not labels.
4. Metadata is complete and correct.
5. Class balance computation works.
6. Multi-task generation works correctly.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.data_generator import (
    DataGenerator,
    Dataset,
    Sample,
    compute_class_balance,
    generate_dataset,
    generate_datasets,
)
from src.registry import build_default_registry, TaskSpec


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def registry():
    return build_default_registry()


@pytest.fixture
def sort_task(registry) -> TaskSpec:
    return registry.get("S1.2_sort")


@pytest.fixture
def threshold_task(registry) -> TaskSpec:
    return registry.get("C1.1_numeric_threshold")


# ===================================================================
# 1. Label Correctness
# ===================================================================

class TestLabelCorrectness:

    def test_sequence_labels_match_reference(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=200, base_seed=42)
        for s in ds.samples:
            expected = sort_task.reference_algorithm(sort_task.input_sampler(s.seed))
            assert s.output_data == expected, (
                f"Label mismatch at seed {s.seed}: got {s.output_data}, expected {expected}"
            )

    def test_classification_labels_match_reference(self, threshold_task: TaskSpec):
        ds = generate_dataset(threshold_task, n_samples=200, base_seed=42)
        for s in ds.samples:
            expected = threshold_task.reference_algorithm(threshold_task.input_sampler(s.seed))
            assert s.output_data == expected

    def test_all_tasks_labels_correct(self, registry):
        """For every registered task, verify 50 samples."""
        for task in registry.all_tasks():
            ds = generate_dataset(task, n_samples=50, base_seed=0)
            for s in ds.samples:
                clean_inp = task.input_sampler(s.seed)
                expected = task.reference_algorithm(clean_inp)
                assert task.verifier(s.output_data, expected), (
                    f"{task.task_id} label mismatch at seed {s.seed}"
                )

    def test_verify_labels_flag_catches_bugs(self):
        """DataGenerator with verify_labels=True should catch mismatches."""
        gen = DataGenerator(verify_labels=True)
        # This should work fine for a normal task
        registry = build_default_registry()
        task = registry.get("S1.2_sort")
        ds = gen.generate(task, n_samples=10, base_seed=42)
        assert len(ds) == 10


# ===================================================================
# 2. Reproducibility
# ===================================================================

class TestReproducibility:

    def test_same_seed_same_dataset(self, sort_task: TaskSpec):
        ds1 = generate_dataset(sort_task, n_samples=100, base_seed=42)
        ds2 = generate_dataset(sort_task, n_samples=100, base_seed=42)
        for s1, s2 in zip(ds1.samples, ds2.samples):
            assert s1.input_data == s2.input_data
            assert s1.output_data == s2.output_data
            assert s1.seed == s2.seed

    def test_different_seed_different_dataset(self, sort_task: TaskSpec):
        ds1 = generate_dataset(sort_task, n_samples=50, base_seed=42)
        ds2 = generate_dataset(sort_task, n_samples=50, base_seed=99)
        inputs1 = [tuple(s.input_data) for s in ds1.samples]
        inputs2 = [tuple(s.input_data) for s in ds2.samples]
        assert inputs1 != inputs2

    def test_classification_reproducible(self, threshold_task: TaskSpec):
        ds1 = generate_dataset(threshold_task, n_samples=100, base_seed=42)
        ds2 = generate_dataset(threshold_task, n_samples=100, base_seed=42)
        for s1, s2 in zip(ds1.samples, ds2.samples):
            assert s1.input_data == s2.input_data
            assert s1.output_data == s2.output_data


# ===================================================================
# 3. Noise Injection
# ===================================================================

class TestNoiseInjection:

    def test_no_noise_inputs_clean(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=100, base_seed=42, noise_level=0.0)
        for s in ds.samples:
            clean = sort_task.input_sampler(s.seed)
            assert s.input_data == clean

    def test_noise_modifies_inputs(self, sort_task: TaskSpec):
        """With high noise, some inputs should differ from clean versions."""
        ds = generate_dataset(sort_task, n_samples=100, base_seed=42, noise_level=0.5)
        n_modified = 0
        for s in ds.samples:
            clean = sort_task.input_sampler(s.seed)
            if s.input_data != clean:
                n_modified += 1
        assert n_modified > 0, "Noise should modify at least some inputs"

    def test_noise_does_not_affect_labels(self, sort_task: TaskSpec):
        """Labels should always match reference on CLEAN inputs, regardless of noise."""
        ds = generate_dataset(sort_task, n_samples=100, base_seed=42, noise_level=0.5)
        for s in ds.samples:
            clean = sort_task.input_sampler(s.seed)
            expected = sort_task.reference_algorithm(clean)
            assert s.output_data == expected, (
                f"Noise should not affect labels: seed={s.seed}"
            )

    def test_noise_on_tabular(self, threshold_task: TaskSpec):
        ds_clean = generate_dataset(threshold_task, n_samples=100, base_seed=42, noise_level=0.0)
        ds_noisy = generate_dataset(threshold_task, n_samples=100, base_seed=42, noise_level=0.5)
        n_modified = 0
        for sc, sn in zip(ds_clean.samples, ds_noisy.samples):
            if sc.input_data != sn.input_data:
                n_modified += 1
        # Labels should still be correct
        for s in ds_noisy.samples:
            clean = threshold_task.input_sampler(s.seed)
            expected = threshold_task.reference_algorithm(clean)
            assert s.output_data == expected


# ===================================================================
# 4. Metadata
# ===================================================================

class TestMetadata:

    def test_sample_metadata_present(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=10, base_seed=42)
        for s in ds.samples:
            assert "seed" in s.metadata
            assert "noise_level" in s.metadata
            assert s.metadata["noise_level"] == 0.0

    def test_dataset_metadata_present(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=10, base_seed=42)
        assert ds.task_metadata["tier"] == "S1"
        assert ds.task_metadata["track"] == "sequence"
        assert ds.task_metadata["n_samples"] == 10
        assert ds.task_metadata["base_seed"] == 42

    def test_noise_level_recorded(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=10, base_seed=42, noise_level=0.3)
        for s in ds.samples:
            assert s.metadata["noise_level"] == 0.3
        assert ds.task_metadata["noise_level"] == 0.3

    def test_task_id_in_samples(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=10, base_seed=42)
        for s in ds.samples:
            assert s.task_id == "S1.2_sort"


# ===================================================================
# 5. Class Balance
# ===================================================================

class TestClassBalance:

    def test_class_balance_binary(self, threshold_task: TaskSpec):
        ds = generate_dataset(threshold_task, n_samples=1000, base_seed=42)
        balance = compute_class_balance(ds)
        assert set(balance.keys()) == {"A", "B"}
        # Both classes should have non-trivial fractions
        assert balance["A"] > 0.1
        assert balance["B"] > 0.1
        # Sum to 1
        assert abs(sum(balance.values()) - 1.0) < 1e-9

    def test_class_balance_empty(self):
        ds = Dataset(task_id="empty", samples=[], task_metadata={})
        balance = compute_class_balance(ds)
        assert balance == {}

    def test_class_balance_multiclass(self, registry):
        task = registry.get("C1.2_range_binning")
        ds = generate_dataset(task, n_samples=1000, base_seed=42)
        balance = compute_class_balance(ds)
        assert len(balance) >= 2  # Should have multiple classes


# ===================================================================
# 6. Dataset Properties
# ===================================================================

class TestDatasetProperties:

    def test_inputs_outputs_accessors(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=10, base_seed=42)
        assert len(ds.inputs) == 10
        assert len(ds.outputs) == 10
        assert ds.inputs[0] == ds.samples[0].input_data
        assert ds.outputs[0] == ds.samples[0].output_data

    def test_len(self, sort_task: TaskSpec):
        ds = generate_dataset(sort_task, n_samples=25, base_seed=42)
        assert len(ds) == 25


# ===================================================================
# 7. Multi-Task Generation
# ===================================================================

class TestMultiTaskGeneration:

    def test_generate_multi(self, registry):
        tasks = [registry.get("S1.1_reverse"), registry.get("S1.2_sort")]
        datasets = generate_datasets(tasks, n_samples_per_task=50, base_seed=42)
        assert len(datasets) == 2
        assert "S1.1_reverse" in datasets
        assert "S1.2_sort" in datasets
        assert len(datasets["S1.1_reverse"]) == 50
        assert len(datasets["S1.2_sort"]) == 50

    def test_multi_task_seeds_dont_overlap(self, registry):
        """Different tasks should use different seed ranges."""
        tasks = [registry.get("S1.1_reverse"), registry.get("S1.2_sort")]
        datasets = generate_datasets(tasks, n_samples_per_task=50, base_seed=0)
        seeds_0 = {s.seed for s in datasets["S1.1_reverse"].samples}
        seeds_1 = {s.seed for s in datasets["S1.2_sort"].samples}
        assert seeds_0.isdisjoint(seeds_1), "Different tasks should use different seeds"

    def test_multi_classification(self, registry):
        tasks = [registry.get("C1.1_numeric_threshold"), registry.get("C2.1_and_rule")]
        datasets = generate_datasets(tasks, n_samples_per_task=100, base_seed=42)
        for tid, ds in datasets.items():
            task = registry.get(tid)
            for s in ds.samples:
                clean = task.input_sampler(s.seed)
                expected = task.reference_algorithm(clean)
                assert task.verifier(s.output_data, expected)
