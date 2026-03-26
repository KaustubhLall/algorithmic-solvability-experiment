"""V-1: Task Registry Validation Tests.

Tests for SR-1 (Task Registry) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. Every registered task has a unique ID.
2. Every task's reference algorithm is deterministic.
3. Every task's input sampler produces valid inputs (conforming to schema).
4. Every task's verifier agrees with the reference algorithm.
5. Expected tiers and tracks are present.
6. Registry operations (get, by_tier, by_track) work correctly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.registry import (
    TaskRegistry,
    TaskSpec,
    _stable_hash_int,
    build_default_registry,
    exact_match_verifier,
    classification_verifier,
)
from src.schemas import ElementType, SequenceInputSchema, TabularInputSchema


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def registry() -> TaskRegistry:
    """Build the default registry once for all tests in this module."""
    return build_default_registry()


# ===================================================================
# 1. Registry Structure
# ===================================================================

class TestRegistryStructure:

    def test_registry_not_empty(self, registry: TaskRegistry):
        assert len(registry) > 0

    def test_all_task_ids_unique(self, registry: TaskRegistry):
        ids = registry.task_ids()
        assert len(ids) == len(set(ids))

    def test_expected_sequence_tiers_present(self, registry: TaskRegistry):
        for tier in ["S0", "S1", "S2", "S3"]:
            tasks = registry.by_tier(tier)
            assert len(tasks) > 0, f"Tier {tier} has no tasks"

    def test_expected_classification_tiers_present(self, registry: TaskRegistry):
        for tier in ["C0", "C1", "C2", "C3"]:
            tasks = registry.by_tier(tier)
            assert len(tasks) > 0, f"Tier {tier} has no tasks"

    def test_sequence_track_tasks(self, registry: TaskRegistry):
        seq_tasks = registry.by_track("sequence")
        assert len(seq_tasks) > 0
        for t in seq_tasks:
            assert t.track == "sequence"

    def test_classification_track_tasks(self, registry: TaskRegistry):
        cls_tasks = registry.by_track("classification")
        assert len(cls_tasks) > 0
        for t in cls_tasks:
            assert t.track == "classification"

    def test_get_existing_task(self, registry: TaskRegistry):
        task = registry.get("S1.2_sort")
        assert task.task_id == "S1.2_sort"

    def test_get_missing_task_raises(self, registry: TaskRegistry):
        with pytest.raises(KeyError, match="not found"):
            registry.get("NONEXISTENT")

    def test_contains(self, registry: TaskRegistry):
        assert "S1.2_sort" in registry
        assert "NONEXISTENT" not in registry

    def test_duplicate_registration_raises(self):
        reg = TaskRegistry()
        schema = SequenceInputSchema(element_type=ElementType.INT, min_length=1, max_length=5)
        task = TaskSpec(
            task_id="test", tier="T0", track="sequence",
            description="test", input_schema=schema,
            output_type="sequence", n_classes=None,
            reference_algorithm=lambda x: x,
            input_sampler=lambda s: schema.sample(s),
            verifier=exact_match_verifier,
        )
        reg.register(task)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(task)


# ===================================================================
# 2. Task Spec Completeness
# ===================================================================

class TestTaskSpecCompleteness:
    """Every task has all required fields populated."""

    def test_all_tasks_have_required_fields(self, registry: TaskRegistry):
        for task in registry.all_tasks():
            assert task.task_id, f"Missing task_id"
            assert task.tier, f"Missing tier for {task.task_id}"
            assert task.track in ("sequence", "classification"), f"Bad track for {task.task_id}"
            assert task.description, f"Missing description for {task.task_id}"
            assert task.input_schema is not None, f"Missing input_schema for {task.task_id}"
            assert task.output_type in ("sequence", "scalar", "class"), f"Bad output_type for {task.task_id}"
            assert task.reference_algorithm is not None, f"Missing reference_algorithm for {task.task_id}"
            assert task.input_sampler is not None, f"Missing input_sampler for {task.task_id}"
            assert task.verifier is not None, f"Missing verifier for {task.task_id}"

    def test_classification_tasks_have_n_classes(self, registry: TaskRegistry):
        for task in registry.by_track("classification"):
            assert task.n_classes is not None and task.n_classes >= 2, (
                f"{task.task_id} missing or invalid n_classes: {task.n_classes}"
            )

    def test_sequence_tasks_have_sequence_schema(self, registry: TaskRegistry):
        for task in registry.by_track("sequence"):
            assert isinstance(task.input_schema, SequenceInputSchema), (
                f"{task.task_id} should have SequenceInputSchema"
            )

    def test_classification_tasks_have_tabular_schema(self, registry: TaskRegistry):
        for task in registry.by_track("classification"):
            assert isinstance(task.input_schema, TabularInputSchema), (
                f"{task.task_id} should have TabularInputSchema"
            )


# ===================================================================
# 3. Reference Algorithm Determinism
# ===================================================================

class TestDeterminism:
    """Every task's reference algorithm produces the same output for the same input."""

    N_SAMPLES = 50

    def test_sequence_tasks_deterministic(self, registry: TaskRegistry):
        for task in registry.by_track("sequence"):
            for seed in range(self.N_SAMPLES):
                inp = task.input_sampler(seed)
                out1 = task.reference_algorithm(inp)
                out2 = task.reference_algorithm(inp)
                assert out1 == out2, (
                    f"{task.task_id} non-deterministic at seed {seed}: {out1} != {out2}"
                )

    def test_classification_tasks_deterministic(self, registry: TaskRegistry):
        for task in registry.by_track("classification"):
            for seed in range(self.N_SAMPLES):
                inp = task.input_sampler(seed)
                out1 = task.reference_algorithm(inp)
                out2 = task.reference_algorithm(inp)
                assert out1 == out2, (
                    f"{task.task_id} non-deterministic at seed {seed}: {out1} != {out2}"
                )


# ===================================================================
# 4. Input Sampler Validity
# ===================================================================

class TestInputSamplerValidity:
    """Sampled inputs conform to the task's schema."""

    N_SAMPLES = 100

    def test_sequence_inputs_valid(self, registry: TaskRegistry):
        for task in registry.by_track("sequence"):
            schema = task.input_schema
            assert isinstance(schema, SequenceInputSchema)
            for seed in range(self.N_SAMPLES):
                inp = task.input_sampler(seed)
                assert schema.validate_input(inp), (
                    f"{task.task_id} invalid input at seed {seed}: {inp}"
                )

    def test_classification_inputs_valid(self, registry: TaskRegistry):
        for task in registry.by_track("classification"):
            schema = task.input_schema
            assert isinstance(schema, TabularInputSchema)
            for seed in range(self.N_SAMPLES):
                inp = task.input_sampler(seed)
                assert schema.validate_input(inp), (
                    f"{task.task_id} invalid input at seed {seed}: {inp}"
                )


# ===================================================================
# 5. Verifier Agrees with Reference
# ===================================================================

class TestVerifierAgreement:
    """Verifier(reference_output, reference_output) is always True."""

    N_SAMPLES = 100

    def test_verifier_accepts_correct_output(self, registry: TaskRegistry):
        for task in registry.all_tasks():
            for seed in range(self.N_SAMPLES):
                inp = task.input_sampler(seed)
                expected = task.reference_algorithm(inp)
                assert task.verifier(expected, expected), (
                    f"{task.task_id} verifier rejected correct output at seed {seed}"
                )

    def test_verifier_rejects_wrong_output_sequence(self, registry: TaskRegistry):
        """For sequence tasks, a modified output should fail verification."""
        for task in registry.by_track("sequence"):
            inp = task.input_sampler(42)
            expected = task.reference_algorithm(inp)
            if isinstance(expected, list) and len(expected) > 0:
                wrong = expected.copy()
                # Modify one element
                wrong[0] = wrong[0] + 999 if isinstance(wrong[0], int) else 0
                assert not task.verifier(wrong, expected), (
                    f"{task.task_id} verifier accepted wrong output"
                )

    def test_verifier_rejects_wrong_output_classification(self, registry: TaskRegistry):
        """For classification tasks, a wrong label should fail verification."""
        for task in registry.by_track("classification"):
            inp = task.input_sampler(42)
            expected = task.reference_algorithm(inp)
            wrong = expected + "_WRONG"
            assert not task.verifier(wrong, expected), (
                f"{task.task_id} verifier accepted wrong class"
            )


# ===================================================================
# 6. Reference Output Type Checks
# ===================================================================

class TestOutputTypes:
    """Reference algorithm outputs have correct types."""

    N_SAMPLES = 50

    def test_sequence_output_is_list(self, registry: TaskRegistry):
        for task in registry.by_track("sequence"):
            for seed in range(self.N_SAMPLES):
                inp = task.input_sampler(seed)
                out = task.reference_algorithm(inp)
                assert isinstance(out, list), (
                    f"{task.task_id} output is {type(out).__name__}, expected list"
                )
                for elem in out:
                    assert isinstance(elem, int), (
                        f"{task.task_id} output element is {type(elem).__name__}, expected int"
                    )

    def test_classification_output_is_string(self, registry: TaskRegistry):
        for task in registry.by_track("classification"):
            for seed in range(self.N_SAMPLES):
                inp = task.input_sampler(seed)
                out = task.reference_algorithm(inp)
                assert isinstance(out, str), (
                    f"{task.task_id} output is {type(out).__name__}, expected str"
                )


# ===================================================================
# 7. Specific Task Spot Checks
# ===================================================================

class TestSpotChecks:
    """Verify specific tasks produce known outputs for known inputs."""

    def test_s1_1_reverse(self, registry: TaskRegistry):
        task = registry.get("S1.1_reverse")
        assert task.reference_algorithm([1, 2, 3]) == [3, 2, 1]
        assert task.reference_algorithm([]) == []

    def test_s1_2_sort(self, registry: TaskRegistry):
        task = registry.get("S1.2_sort")
        assert task.reference_algorithm([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]

    def test_s1_5_parity(self, registry: TaskRegistry):
        task = registry.get("S1.5_parity")
        assert task.reference_algorithm([1, 0, 1]) == [0]  # even number of 1s
        assert task.reference_algorithm([1, 1, 1]) == [1]  # odd number of 1s

    def test_s1_6_prefix_sum(self, registry: TaskRegistry):
        task = registry.get("S1.6_prefix_sum")
        assert task.reference_algorithm([1, 2, 3, 4]) == [1, 3, 6, 10]

    def test_s2_1_cumulative_xor(self, registry: TaskRegistry):
        task = registry.get("S2.1_cumulative_xor")
        assert task.reference_algorithm([1, 0, 1, 1]) == [1, 1, 0, 1]

    def test_c1_1_numeric_threshold(self, registry: TaskRegistry):
        task = registry.get("C1.1_numeric_threshold")
        assert task.reference_algorithm({"x1": 75.0}) == "A"
        assert task.reference_algorithm({"x1": 25.0}) == "B"

    def test_c1_3_categorical_match(self, registry: TaskRegistry):
        task = registry.get("C1.3_categorical_match")
        assert task.reference_algorithm({"cat1": "A"}) == "YES"
        assert task.reference_algorithm({"cat1": "B"}) == "NO"

    def test_c2_1_and_rule(self, registry: TaskRegistry):
        task = registry.get("C2.1_and_rule")
        assert task.reference_algorithm({"x1": 75.0, "x2": 0.0, "cat1": "A"}) == "YES"
        assert task.reference_algorithm({"x1": 75.0, "x2": 0.0, "cat1": "B"}) == "NO"
        assert task.reference_algorithm({"x1": 25.0, "x2": 0.0, "cat1": "A"}) == "NO"

    def test_c3_1_xor(self, registry: TaskRegistry):
        task = registry.get("C3.1_xor")
        assert task.reference_algorithm({"x1": 75.0, "x2": 25.0}) == "A"  # T XOR F
        assert task.reference_algorithm({"x1": 75.0, "x2": 75.0}) == "B"  # T XOR T
        assert task.reference_algorithm({"x1": 25.0, "x2": 25.0}) == "B"  # F XOR F
        assert task.reference_algorithm({"x1": 25.0, "x2": 75.0}) == "A"  # F XOR T


# ===================================================================
# 8. Reproducibility
# ===================================================================

class TestReproducibility:
    """Same seed → same sampled input → same reference output."""

    def test_input_sampler_reproducibility(self, registry: TaskRegistry):
        for task in registry.all_tasks():
            s1 = task.input_sampler(42)
            s2 = task.input_sampler(42)
            assert s1 == s2, f"{task.task_id} sampler not reproducible"

    def test_end_to_end_reproducibility(self, registry: TaskRegistry):
        for task in registry.all_tasks():
            inp1 = task.input_sampler(42)
            inp2 = task.input_sampler(42)
            out1 = task.reference_algorithm(inp1)
            out2 = task.reference_algorithm(inp2)
            assert out1 == out2, f"{task.task_id} end-to-end not reproducible"

    def test_stable_hash_helper_is_deterministic(self):
        value = [("x1", "1.5"), ("cat1", "A")]
        assert _stable_hash_int(value) == _stable_hash_int(value)
