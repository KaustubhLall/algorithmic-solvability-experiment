"""TASK-14 diagnostic experiment tests."""

from __future__ import annotations

import json

import pytest

from src.diagnostic_experiments import (
    BaselineTaskRecord,
    DiagnosticExperimentArtifact,
    _calibrated_label,
    _clone_task_with_distractors,
    _curve_auc,
    collect_baseline_task_records,
    run_distractor_robustness_experiment,
    run_feature_importance_alignment_experiment,
    run_noise_robustness_experiment,
    run_sample_efficiency_experiment,
    run_solvability_calibration_experiment,
)
from src.models.harness import ModelConfig, ModelFamily
from src.registry import build_default_registry


def test_collect_baseline_task_records_loads_best_model_config(tmp_path):
    experiment_dir = tmp_path / "EXP-C1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "config.json").write_text(
        json.dumps(
            {
                "model_configs": [
                    {"family": "decision_tree", "name": "decision_tree", "hyperparams": {}},
                ]
            }
        ),
        encoding="utf-8",
    )
    (experiment_dir / "solvability_verdicts.json").write_text(
        json.dumps(
            {
                "C1.1_numeric_threshold": {
                    "task_id": "C1.1_numeric_threshold",
                    "tier": "C1",
                    "track": "classification",
                    "label": "MODERATE",
                    "score": 0.72,
                    "best_model": "decision_tree",
                    "best_iid_accuracy": 1.0,
                    "best_ood_accuracy": 1.0,
                    "evidence": {
                        "criterion_1_high_iid_accuracy": True,
                        "criterion_2_extrapolation_success": True,
                        "criterion_3_baseline_separation": True,
                        "criterion_4_seed_stability": True,
                        "criterion_5_coherent_degradation": True,
                        "criterion_6_counterfactual_sensitivity": False,
                        "criterion_7_distractor_robustness": False,
                        "criterion_8_sample_efficiency": False,
                        "criterion_9_transfer": False,
                    },
                    "notes": [],
                }
            }
        ),
        encoding="utf-8",
    )

    records = collect_baseline_task_records(results_root=tmp_path)

    assert "C1.1_numeric_threshold" in records
    assert records["C1.1_numeric_threshold"].best_model_name == "decision_tree"
    assert records["C1.1_numeric_threshold"].best_model_config is not None
    assert records["C1.1_numeric_threshold"].best_model_config.family.value == "decision_tree"


def test_clone_task_with_distractors_preserves_labels():
    registry = build_default_registry()
    task = registry.get("C2.1_and_rule")
    augmented = _clone_task_with_distractors(task, distractor_count=3)

    assert augmented.input_schema.n_features == task.input_schema.n_features + 3

    sample = augmented.input_sampler(7)
    filtered = {key: value for key, value in sample.items() if key in {"x1", "x2", "cat1"}}
    assert augmented.reference_algorithm(sample) == task.reference_algorithm(filtered)


def test_run_sample_efficiency_experiment_writes_artifacts(tmp_path):
    registry = build_default_registry()
    baseline_root = tmp_path / "baseline"

    experiment_layout = [
        ("EXP-C1", "C1.1_numeric_threshold", "decision_tree", "MODERATE", 0.72),
        ("EXP-C2", "C2.3_nested_if_else", "decision_tree", "MODERATE", 0.74),
        ("EXP-C3", "C3.3_rank_based", "decision_tree", "MODERATE", 0.73),
        ("EXP-S1", "S1.4_count_symbol", "mlp", "WEAK", 0.62),
        ("EXP-S2", "S2.2_balanced_parens", "mlp", "WEAK", 0.64),
        ("EXP-S3", "S3.1_dedup_sort_count", "mlp", "NEGATIVE", 0.37),
        ("EXP-0.3", "S0.1_random_labels", "mlp", "NEGATIVE", 0.20),
        ("EXP-0.3", "C0.1_random_class", "mlp", "NEGATIVE", 0.21),
    ]

    for experiment_id, task_id, best_model, label, score in experiment_layout:
        experiment_dir = baseline_root / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_configs": [
                        {"family": "decision_tree", "name": "decision_tree", "hyperparams": {}},
                        {"family": "mlp", "name": "mlp", "hyperparams": {}},
                    ]
                }
            ),
            encoding="utf-8",
        )
        verdicts_path = experiment_dir / "solvability_verdicts.json"
        existing = json.loads(verdicts_path.read_text(encoding="utf-8")) if verdicts_path.exists() else {}
        existing[task_id] = {
            "task_id": task_id,
            "tier": registry.get(task_id).tier,
            "track": registry.get(task_id).track,
            "label": label,
            "score": score,
            "best_model": best_model,
            "best_iid_accuracy": 1.0 if "C0." not in task_id and "S0." not in task_id else 0.3,
            "best_ood_accuracy": 1.0 if "C0." not in task_id and "S0." not in task_id else 0.2,
            "evidence": {
                "criterion_1_high_iid_accuracy": "0." not in task_id,
                "criterion_2_extrapolation_success": "0." not in task_id,
                "criterion_3_baseline_separation": "0." not in task_id,
                "criterion_4_seed_stability": True,
                "criterion_5_coherent_degradation": True,
                "criterion_6_counterfactual_sensitivity": False,
                "criterion_7_distractor_robustness": False,
                "criterion_8_sample_efficiency": False,
                "criterion_9_transfer": False,
            },
            "notes": [],
        }
        verdicts_path.write_text(json.dumps(existing), encoding="utf-8")

    artifact = run_sample_efficiency_experiment(
        output_root=tmp_path / "out",
        results_root=baseline_root,
        registry=registry,
        sample_sizes=[20, 40],
        test_size=20,
        seeds=[7],
    )

    assert artifact.output_dir.exists()
    assert (artifact.output_dir / "sample_efficiency.json").exists()
    assert "C1.1_numeric_threshold" in artifact.payload["task_curves"]


def test_run_distractor_robustness_experiment_writes_artifacts(tmp_path):
    artifact = run_distractor_robustness_experiment(
        output_root=tmp_path,
        task_ids=["C1.1_numeric_threshold"],
        model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
        distractor_counts=[0, 2],
        seeds=[7],
        n_samples=60,
    )

    assert artifact.output_dir.exists()
    assert (artifact.output_dir / "distractor_robustness.json").exists()
    assert "C1.1_numeric_threshold" in artifact.payload["task_summary"]


def test_run_solvability_calibration_promotes_strong_with_d1_and_d2(tmp_path):
    baseline_records = {
        "S0.1_random_labels": BaselineTaskRecord(
            task_id="S0.1_random_labels",
            track="sequence",
            tier="S0",
            source_experiment="EXP-0.3",
            label="NEGATIVE",
            score=0.2,
            best_model_name="mlp",
            best_model_config=ModelConfig(family=ModelFamily.MLP),
            best_iid_accuracy=0.2,
            best_ood_accuracy=None,
            evidence={
                "criterion_1_high_iid_accuracy": False,
                "criterion_2_extrapolation_success": False,
                "criterion_3_baseline_separation": False,
                "criterion_4_seed_stability": True,
                "criterion_5_coherent_degradation": True,
                "criterion_6_counterfactual_sensitivity": False,
                "criterion_7_distractor_robustness": False,
                "criterion_8_sample_efficiency": False,
                "criterion_9_transfer": False,
            },
            notes=[],
        ),
        "C0.1_random_class": BaselineTaskRecord(
            task_id="C0.1_random_class",
            track="classification",
            tier="C0",
            source_experiment="EXP-0.3",
            label="NEGATIVE",
            score=0.2,
            best_model_name="mlp",
            best_model_config=ModelConfig(family=ModelFamily.MLP),
            best_iid_accuracy=0.3,
            best_ood_accuracy=None,
            evidence={
                "criterion_1_high_iid_accuracy": False,
                "criterion_2_extrapolation_success": False,
                "criterion_3_baseline_separation": False,
                "criterion_4_seed_stability": True,
                "criterion_5_coherent_degradation": True,
                "criterion_6_counterfactual_sensitivity": False,
                "criterion_7_distractor_robustness": False,
                "criterion_8_sample_efficiency": False,
                "criterion_9_transfer": False,
            },
            notes=[],
        ),
        "C1.1_numeric_threshold": BaselineTaskRecord(
            task_id="C1.1_numeric_threshold",
            track="classification",
            tier="C1",
            source_experiment="EXP-C1",
            label="MODERATE",
            score=0.72,
            best_model_name="decision_tree",
            best_model_config=ModelConfig(family=ModelFamily.DECISION_TREE),
            best_iid_accuracy=1.0,
            best_ood_accuracy=1.0,
            evidence={
                "criterion_1_high_iid_accuracy": True,
                "criterion_2_extrapolation_success": True,
                "criterion_3_baseline_separation": True,
                "criterion_4_seed_stability": True,
                "criterion_5_coherent_degradation": True,
                "criterion_6_counterfactual_sensitivity": False,
                "criterion_7_distractor_robustness": False,
                "criterion_8_sample_efficiency": False,
                "criterion_9_transfer": False,
            },
            notes=[],
        ),
    }

    artifact = run_solvability_calibration_experiment(
        output_root=tmp_path,
        baseline_records=baseline_records,
        d1_payload={
            "task_curves": {
                "C1.1_numeric_threshold": {
                    "criterion_8_sample_efficiency": True,
                    "sample_efficiency_score": 1.0,
                    "delta_vs_control_auc": 0.6,
                }
            }
        },
        d2_payload={
            "task_summary": {
                "C1.1_numeric_threshold": {
                    "criterion_7_distractor_robustness": True,
                    "distractor_robustness_score": 0.97,
                    "accuracy_drop": 0.03,
                }
            }
        },
        d3_payload={"task_summary": {}},
        d4_payload={"results": {}},
    )

    assert artifact.output_dir.exists()
    assert artifact.payload["tasks"]["C1.1_numeric_threshold"]["calibrated_label"] == "STRONG"
    assert (artifact.output_dir / "solvability_calibration.json").exists()


# ---------------------------------------------------------------------------
# _curve_auc unit tests
# ---------------------------------------------------------------------------


def test_curve_auc_monotone_perfect():
    sizes = [100, 1000, 10000]
    accs = [1.0, 1.0, 1.0]
    assert _curve_auc(sizes, accs) == 1.0


def test_curve_auc_increasing():
    sizes = [100, 1000]
    accs = [0.5, 0.9]
    auc = _curve_auc(sizes, accs)
    assert 0.5 < auc < 1.0


def test_curve_auc_single_point():
    assert _curve_auc([100], [0.75]) == 0.75


def test_curve_auc_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        _curve_auc([100, 200], [0.5])


# ---------------------------------------------------------------------------
# _calibrated_label unit tests
# ---------------------------------------------------------------------------


def _full_evidence(**overrides: bool) -> dict:
    base = {
        "criterion_1_high_iid_accuracy": True,
        "criterion_2_extrapolation_success": True,
        "criterion_3_baseline_separation": True,
        "criterion_4_seed_stability": True,
        "criterion_5_coherent_degradation": True,
        "criterion_6_counterfactual_sensitivity": False,
        "criterion_7_distractor_robustness": False,
        "criterion_8_sample_efficiency": False,
        "criterion_9_transfer": False,
    }
    base.update(overrides)
    return base


def test_calibrated_label_strong():
    evidence = _full_evidence(
        criterion_7_distractor_robustness=True,
        criterion_8_sample_efficiency=True,
    )
    assert _calibrated_label(evidence, 0.99) == "STRONG"


def test_calibrated_label_moderate():
    evidence = _full_evidence()
    assert _calibrated_label(evidence, 0.99) == "MODERATE"


def test_calibrated_label_weak():
    evidence = _full_evidence(criterion_2_extrapolation_success=False)
    assert _calibrated_label(evidence, 0.95) == "WEAK"


def test_calibrated_label_negative_low_iid():
    evidence = _full_evidence(
        criterion_1_high_iid_accuracy=False,
        criterion_2_extrapolation_success=False,
    )
    assert _calibrated_label(evidence, 0.3) == "NEGATIVE"


def test_calibrated_label_inconclusive():
    evidence = _full_evidence(
        criterion_1_high_iid_accuracy=False,
        criterion_4_seed_stability=True,
    )
    assert _calibrated_label(evidence, 0.7) in {"NEGATIVE", "INCONCLUSIVE"}


# ---------------------------------------------------------------------------
# EXP-D3 noise robustness
# ---------------------------------------------------------------------------


def test_run_noise_robustness_experiment_writes_artifacts(tmp_path):
    artifact = run_noise_robustness_experiment(
        output_root=tmp_path,
        task_ids=["C1.1_numeric_threshold"],
        model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
        noise_levels=[0.0, 0.1],
        seeds=[7],
        n_samples=60,
    )

    assert isinstance(artifact, DiagnosticExperimentArtifact)
    assert artifact.experiment_id == "EXP-D3"
    assert artifact.output_dir.exists()
    assert (artifact.output_dir / "noise_robustness.json").exists()
    assert (artifact.output_dir / "summary.md").exists()
    assert (artifact.output_dir / "config.json").exists()
    assert "C1.1_numeric_threshold" in artifact.payload["task_summary"]

    summary = artifact.payload["task_summary"]["C1.1_numeric_threshold"]
    assert "selected_model" in summary
    assert "clean_accuracy" in summary
    assert "noise_accuracy_drop" in summary
    assert "smooth_degradation" in summary


def test_noise_robustness_per_task_plot_created(tmp_path):
    artifact = run_noise_robustness_experiment(
        output_root=tmp_path,
        task_ids=["C1.1_numeric_threshold"],
        model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
        noise_levels=[0.0, 0.05],
        seeds=[7],
        n_samples=60,
    )
    plot_path = artifact.output_dir / "per_task" / "C1.1_numeric_threshold" / "noise_curve.png"
    assert plot_path.exists()


# ---------------------------------------------------------------------------
# EXP-D4 feature importance alignment
# ---------------------------------------------------------------------------


def test_run_feature_importance_alignment_writes_artifacts(tmp_path):
    artifact = run_feature_importance_alignment_experiment(
        output_root=tmp_path,
        task_ids=["C2.1_and_rule"],
        model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
        seeds=[7],
        n_samples=80,
        distractor_count=3,
    )

    assert isinstance(artifact, DiagnosticExperimentArtifact)
    assert artifact.experiment_id == "EXP-D4"
    assert artifact.output_dir.exists()
    assert (artifact.output_dir / "feature_importance_alignment.json").exists()
    assert (artifact.output_dir / "summary.md").exists()
    assert (artifact.output_dir / "config.json").exists()
    assert "C2.1_and_rule" in artifact.payload["results"]

    task_result = artifact.payload["results"]["C2.1_and_rule"]
    assert "relevant_features" in task_result
    assert task_result["distractor_count"] == 3
    assert "models" in task_result


def test_feature_importance_alignment_per_task_plot_created(tmp_path):
    artifact = run_feature_importance_alignment_experiment(
        output_root=tmp_path,
        task_ids=["C2.1_and_rule"],
        model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
        seeds=[7],
        n_samples=80,
        distractor_count=2,
    )
    plot_path = artifact.output_dir / "per_task" / "C2.1_and_rule" / "feature_alignment.png"
    assert plot_path.exists()


def test_feature_importance_precision_at_k_nonzero(tmp_path):
    artifact = run_feature_importance_alignment_experiment(
        output_root=tmp_path,
        task_ids=["C1.1_numeric_threshold"],
        model_configs=[ModelConfig(family=ModelFamily.DECISION_TREE)],
        seeds=[7],
        n_samples=120,
        distractor_count=5,
    )
    task_result = artifact.payload["results"]["C1.1_numeric_threshold"]
    dt_result = task_result["models"]["decision_tree"]
    assert dt_result["precision_at_k_mean"] > 0.0


# ---------------------------------------------------------------------------
# Distractor clone edge cases
# ---------------------------------------------------------------------------


def test_clone_task_with_zero_distractors_keeps_original_schema():
    registry = build_default_registry()
    task = registry.get("C1.1_numeric_threshold")
    augmented = _clone_task_with_distractors(task, distractor_count=0)
    assert augmented.input_schema.n_features == task.input_schema.n_features


# ---------------------------------------------------------------------------
# Calibration checks within EXP-D5
# ---------------------------------------------------------------------------


def test_solvability_calibration_controls_are_negative(tmp_path):
    baseline_records = {
        "S0.1_random_labels": BaselineTaskRecord(
            task_id="S0.1_random_labels",
            track="sequence", tier="S0",
            source_experiment="EXP-0.3",
            label="NEGATIVE", score=0.2,
            best_model_name="mlp",
            best_model_config=ModelConfig(family=ModelFamily.MLP),
            best_iid_accuracy=0.2, best_ood_accuracy=None,
            evidence={
                "criterion_1_high_iid_accuracy": False,
                "criterion_2_extrapolation_success": False,
                "criterion_3_baseline_separation": False,
                "criterion_4_seed_stability": True,
                "criterion_5_coherent_degradation": True,
            },
            notes=[],
        ),
        "C0.1_random_class": BaselineTaskRecord(
            task_id="C0.1_random_class",
            track="classification", tier="C0",
            source_experiment="EXP-0.3",
            label="NEGATIVE", score=0.2,
            best_model_name="mlp",
            best_model_config=ModelConfig(family=ModelFamily.MLP),
            best_iid_accuracy=0.3, best_ood_accuracy=None,
            evidence={
                "criterion_1_high_iid_accuracy": False,
                "criterion_2_extrapolation_success": False,
                "criterion_3_baseline_separation": False,
                "criterion_4_seed_stability": True,
                "criterion_5_coherent_degradation": True,
            },
            notes=[],
        ),
    }
    artifact = run_solvability_calibration_experiment(
        output_root=tmp_path,
        baseline_records=baseline_records,
        d1_payload={"task_curves": {}},
        d2_payload={"task_summary": {}},
        d3_payload={"task_summary": {}},
        d4_payload={"results": {}},
    )
    checks = artifact.payload["calibration_checks"]
    assert checks["controls_negative_or_weak"] is True
    for task_id in ["S0.1_random_labels", "C0.1_random_class"]:
        assert artifact.payload["tasks"][task_id]["calibrated_label"] in {"NEGATIVE", "WEAK"}
