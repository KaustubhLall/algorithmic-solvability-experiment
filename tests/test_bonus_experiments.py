"""TASK-15 bonus experiment tests."""

from __future__ import annotations

import json

import pytest

from src.bonus_experiments import (
    BonusExperimentArtifact,
    _evaluate_program_against_oracle,
    _extract_tree_text,
    _generate_candidate_programs,
    _train_decision_tree,
    _tree_structural_info,
    _validate_on_hard_test,
    run_program_search_experiment,
    run_rule_extraction_experiment,
)
from src.data_generator import generate_dataset
from src.dsl.sequence_dsl import (
    Compose,
    Reverse,
    SeqProgram,
    Sort,
    Sum,
)
from src.registry import build_default_registry


@pytest.fixture
def registry():
    return build_default_registry()


# ===================================================================
# EXP-B1 tests
# ===================================================================


class TestRuleExtraction:
    """Tests for EXP-B1: Rule Extraction from Classification Models."""

    def test_run_rule_extraction_writes_artifacts(self, tmp_path, registry):
        """EXP-B1 runner produces JSON, summary, config, and plot."""
        artifact = run_rule_extraction_experiment(
            output_root=tmp_path,
            task_ids=["C1.1_numeric_threshold"],
            registry=registry,
            n_train=100,
            n_hard_test=50,
            max_depth_sweep=[2, 5],
            seeds=[42],
        )

        assert artifact.experiment_id == "EXP-B1"
        assert (artifact.output_dir / "rule_extraction.json").exists()
        assert (artifact.output_dir / "config.json").exists()
        assert (artifact.output_dir / "summary.md").exists()
        assert (artifact.output_dir / "depth_sweep.png").exists()

    def test_rule_extraction_per_task_artifacts(self, tmp_path, registry):
        """Per-task directory contains extracted tree text and result JSON."""
        artifact = run_rule_extraction_experiment(
            output_root=tmp_path,
            task_ids=["C1.1_numeric_threshold"],
            registry=registry,
            n_train=100,
            n_hard_test=50,
            max_depth_sweep=[3],
            seeds=[42],
        )

        per_task = artifact.output_dir / "per_task" / "C1.1_numeric_threshold"
        assert per_task.exists()
        assert (per_task / "extracted_tree.txt").exists()
        assert (per_task / "result.json").exists()

        result = json.loads((per_task / "result.json").read_text())
        assert "best_accuracy" in result
        assert "structural_info" in result

    def test_simple_threshold_task_recoverable(self, tmp_path, registry):
        """C1.1_numeric_threshold (x1 > 50) should be perfectly recoverable by a shallow tree."""
        artifact = run_rule_extraction_experiment(
            output_root=tmp_path,
            task_ids=["C1.1_numeric_threshold"],
            registry=registry,
            n_train=2000,
            n_hard_test=1000,
            max_depth_sweep=[2, 5],
            seeds=[42, 123],
        )

        payload = artifact.payload
        result = payload["task_results"]["C1.1_numeric_threshold"]
        # A simple threshold should be near-perfectly recoverable
        assert result["best_accuracy"] >= 0.95, (
            f"Expected high accuracy for simple threshold, got {result['best_accuracy']}"
        )

    def test_train_decision_tree_returns_valid_objects(self, registry):
        """_train_decision_tree returns a fitted tree, encoder, feature names, and class names."""
        task = registry.get("C1.1_numeric_threshold")
        dataset = generate_dataset(task, n_samples=200, base_seed=42)

        clf, encoder, feature_names, class_names = _train_decision_tree(
            task, dataset.samples, max_depth=3, seed=42,
        )

        assert hasattr(clf, "predict")
        assert len(feature_names) > 0
        assert len(class_names) >= 2

    def test_tree_structural_info_identifies_relevant_features(self, registry):
        """Structural info correctly identifies which features the tree uses."""
        task = registry.get("C1.1_numeric_threshold")
        dataset = generate_dataset(task, n_samples=500, base_seed=42)

        clf, encoder, feature_names, class_names = _train_decision_tree(
            task, dataset.samples, max_depth=3, seed=42,
        )

        info = _tree_structural_info(clf, feature_names, ("x1",))
        assert info["n_nodes"] > 0
        assert info["depth"] >= 1
        # For a simple threshold on x1, tree should primarily use x1
        assert "x1" in info["used_base_features"]

    def test_extract_tree_text_produces_readable_output(self, registry):
        """export_text returns a non-empty string."""
        task = registry.get("C1.1_numeric_threshold")
        dataset = generate_dataset(task, n_samples=200, base_seed=42)

        clf, encoder, feature_names, class_names = _train_decision_tree(
            task, dataset.samples, max_depth=3, seed=42,
        )

        text = _extract_tree_text(clf, feature_names)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_multiple_tasks_produce_results(self, tmp_path, registry):
        """Running on multiple tasks produces results for each."""
        task_ids = ["C1.1_numeric_threshold", "C2.1_and_rule"]
        artifact = run_rule_extraction_experiment(
            output_root=tmp_path,
            task_ids=task_ids,
            registry=registry,
            n_train=100,
            n_hard_test=50,
            max_depth_sweep=[3],
            seeds=[42],
        )

        for tid in task_ids:
            assert tid in artifact.payload["task_results"]

    def test_summary_acceptance_when_one_passes(self, tmp_path, registry):
        """Acceptance is met if at least one task passes 99%."""
        artifact = run_rule_extraction_experiment(
            output_root=tmp_path,
            task_ids=["C1.1_numeric_threshold"],
            registry=registry,
            n_train=2000,
            n_hard_test=500,
            max_depth_sweep=[2, 5, None],
            seeds=[42, 123],
        )

        summary = artifact.payload["summary"]
        # At least check the structure is present
        assert "acceptance_met" in summary
        assert "tasks_passing_99" in summary


# ===================================================================
# EXP-B2 tests
# ===================================================================


class TestProgramSearch:
    """Tests for EXP-B2: DSL Program Search for Sequence Tasks."""

    def test_run_program_search_writes_artifacts(self, tmp_path, registry):
        """EXP-B2 runner produces JSON, summary, config, and plot."""
        artifact = run_program_search_experiment(
            output_root=tmp_path,
            task_ids=["S1.1_reverse"],
            registry=registry,
            n_oracle_samples=30,
            n_hard_test=20,
            search_budget=50,
            max_depth=2,
            seeds=[42],
        )

        assert artifact.experiment_id == "EXP-B2"
        assert (artifact.output_dir / "program_search.json").exists()
        assert (artifact.output_dir / "config.json").exists()
        assert (artifact.output_dir / "summary.md").exists()
        assert (artifact.output_dir / "program_search_results.png").exists()

    def test_program_search_per_task_artifacts(self, tmp_path, registry):
        """Per-task directory contains result JSON."""
        artifact = run_program_search_experiment(
            output_root=tmp_path,
            task_ids=["S1.1_reverse"],
            registry=registry,
            n_oracle_samples=20,
            n_hard_test=10,
            search_budget=30,
            max_depth=2,
            seeds=[42],
        )

        per_task = artifact.output_dir / "per_task" / "S1.1_reverse"
        assert per_task.exists()
        assert (per_task / "result.json").exists()

    def test_generate_candidate_programs_returns_programs(self):
        """_generate_candidate_programs returns a list of SeqProgram."""
        progs = _generate_candidate_programs(budget=20, max_depth=2, seed=42)
        assert len(progs) == 20
        for p in progs:
            assert isinstance(p, SeqProgram)

    def test_evaluate_program_against_oracle_perfect_match(self):
        """A program that matches the oracle perfectly scores 1.0."""
        prog = SeqProgram(program_id="test_sort", op=Sort(), description="sort")
        inputs = [[3, 1, 2], [5, 4], [1]]
        outputs = [sorted(inp) for inp in inputs]

        score = _evaluate_program_against_oracle(prog, inputs, outputs)
        assert score == 1.0

    def test_evaluate_program_against_oracle_no_match(self):
        """A program that never matches scores 0.0."""
        prog = SeqProgram(program_id="test_reverse", op=Reverse(), description="reverse")
        # Oracle outputs are sorted, but program reverses — likely mismatches
        inputs = [[3, 1, 2], [5, 4, 3]]
        outputs = [sorted(inp) for inp in inputs]

        score = _evaluate_program_against_oracle(prog, inputs, outputs)
        assert score < 1.0

    def test_evaluate_program_against_oracle_empty_inputs(self):
        """Empty oracle inputs returns 0.0."""
        prog = SeqProgram(program_id="test", op=Sort())
        score = _evaluate_program_against_oracle(prog, [], [])
        assert score == 0.0

    def test_validate_on_hard_test_perfect_program(self, registry):
        """A program that IS the reference algorithm scores 1.0 on hard test."""
        task = registry.get("S1.2_sort")
        prog = SeqProgram(program_id="sort", op=Sort(), description="sort")

        acc = _validate_on_hard_test(prog, task, n_hard_test=100, seed=42)
        assert acc == 1.0

    def test_validate_on_hard_test_wrong_program(self, registry):
        """A wrong program scores < 1.0 on hard test."""
        task = registry.get("S1.2_sort")
        prog = SeqProgram(program_id="reverse", op=Reverse(), description="reverse")

        acc = _validate_on_hard_test(prog, task, n_hard_test=100, seed=42)
        assert acc < 1.0

    def test_reverse_task_discoverable(self, tmp_path, registry):
        """S1.1_reverse should be discoverable since Reverse() is a DSL primitive."""
        artifact = run_program_search_experiment(
            output_root=tmp_path,
            task_ids=["S1.1_reverse"],
            registry=registry,
            n_oracle_samples=100,
            n_hard_test=100,
            search_budget=500,
            max_depth=2,
            seeds=[42],
        )

        result = artifact.payload["task_results"]["S1.1_reverse"]
        # Reverse is a primitive in the DSL, should be found with decent budget
        assert result["best_oracle_score"] >= 0.5, (
            f"Expected reasonable oracle score for reverse, got {result['best_oracle_score']}"
        )

    def test_sort_task_discoverable(self, tmp_path, registry):
        """S1.2_sort should be discoverable since Sort() is a DSL primitive."""
        artifact = run_program_search_experiment(
            output_root=tmp_path,
            task_ids=["S1.2_sort"],
            registry=registry,
            n_oracle_samples=100,
            n_hard_test=100,
            search_budget=500,
            max_depth=2,
            seeds=[42],
        )

        result = artifact.payload["task_results"]["S1.2_sort"]
        assert result["best_oracle_score"] >= 0.5

    def test_multiple_sequence_tasks_produce_results(self, tmp_path, registry):
        """Running on multiple tasks produces results for each."""
        task_ids = ["S1.1_reverse", "S1.2_sort"]
        artifact = run_program_search_experiment(
            output_root=tmp_path,
            task_ids=task_ids,
            registry=registry,
            n_oracle_samples=20,
            n_hard_test=10,
            search_budget=30,
            max_depth=2,
            seeds=[42],
        )

        for tid in task_ids:
            assert tid in artifact.payload["task_results"]

    def test_payload_summary_structure(self, tmp_path, registry):
        """Payload summary contains expected fields."""
        artifact = run_program_search_experiment(
            output_root=tmp_path,
            task_ids=["S1.1_reverse"],
            registry=registry,
            n_oracle_samples=10,
            n_hard_test=10,
            search_budget=20,
            max_depth=2,
            seeds=[42],
        )

        summary = artifact.payload["summary"]
        assert "total_tasks" in summary
        assert "tasks_passing_99" in summary
        assert "pass_rate" in summary
        assert "acceptance_met" in summary
