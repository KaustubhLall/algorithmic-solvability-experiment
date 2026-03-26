"""V-6: Evaluation Engine Validation Tests.

Tests for SR-6 (Evaluation Engine) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. Perfect prediction test: all metrics = 1.0, all errors = 0.
2. Worst prediction test: metrics correct for all-wrong predictions.
3. Known confusion matrix test: hand-crafted confusion reproduced exactly.
4. Metric consistency: accuracy == diagonal sum / total of confusion matrix.
5. Classification vs. sequence dispatch: correct metric set based on task type.
6. Per-class precision/recall/F1 correctness.
7. Macro and weighted F1.
8. Sequence exact match and token accuracy.
9. Error taxonomy for both tracks.
10. Metadata-conditioned metrics.
11. Edge cases (empty predictions, single class, etc.).
12. evaluate_prediction_result convenience function.
13. eval_report_to_dict serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest

from src.evaluation import (
    EvalReport,
    PerClassMetrics,
    evaluate,
    evaluate_prediction_result,
    eval_report_to_dict,
    _compute_accuracy,
    _compute_confusion_matrix,
    _compute_token_accuracy,
    _per_class_from_confusion,
    _macro_f1,
    _weighted_f1,
    _classification_error_taxonomy,
    _sequence_error_taxonomy,
    _metadata_conditioned_accuracy,
    _parse_sequence_str,
)
from src.registry import TaskSpec
from src.schemas import (
    CategoricalFeatureSpec,
    ElementType,
    NumericalFeatureSpec,
    SequenceInputSchema,
    TabularInputSchema,
)


# ===================================================================
# Mock TaskSpec helpers
# ===================================================================

def _make_classification_task(
    task_id: str = "C1.1_test",
    n_classes: int = 2,
) -> TaskSpec:
    """Create a minimal classification TaskSpec for testing."""
    schema = TabularInputSchema(
        numerical_features=(
            NumericalFeatureSpec(name="x1", min_val=0.0, max_val=100.0),
        ),
    )
    return TaskSpec(
        task_id=task_id,
        tier="C1",
        track="classification",
        description="Test classification task",
        input_schema=schema,
        output_type="class",
        n_classes=n_classes,
        reference_algorithm=lambda x: "A" if x.get("x1", 0) > 50 else "B",
        input_sampler=lambda seed: schema.sample(seed),
        verifier=lambda pred, expected: pred == expected,
        complexity_metadata={"depth": 1},
    )


def _make_sequence_task(
    task_id: str = "S1.2_test",
) -> TaskSpec:
    """Create a minimal sequence TaskSpec for testing."""
    schema = SequenceInputSchema(
        element_type=ElementType.INT,
        min_length=2,
        max_length=10,
        value_range=(0, 9),
    )
    return TaskSpec(
        task_id=task_id,
        tier="S1",
        track="sequence",
        description="Test sequence task",
        input_schema=schema,
        output_type="sequence",
        n_classes=None,
        reference_algorithm=lambda x: sorted(x),
        input_sampler=lambda seed: schema.sample(seed),
        verifier=lambda pred, expected: pred == expected,
        complexity_metadata={"depth": 1},
    )


# ===================================================================
# 1. Perfect prediction test
# ===================================================================

class TestPerfectPrediction:

    def test_classification_perfect(self):
        task = _make_classification_task()
        gt = ["A", "B", "A", "B", "A"]
        pred = ["A", "B", "A", "B", "A"]
        report = evaluate(pred, gt, task, "test")

        assert report.accuracy == 1.0
        assert report.track == "classification"
        assert report.n_samples == 5
        assert report.per_class_metrics is not None
        assert report.per_class_metrics["A"].f1 == 1.0
        assert report.per_class_metrics["B"].f1 == 1.0
        assert report.per_class_metrics["A"].precision == 1.0
        assert report.per_class_metrics["A"].recall == 1.0
        assert report.per_class_metrics["B"].precision == 1.0
        assert report.per_class_metrics["B"].recall == 1.0
        assert report.macro_f1 == 1.0
        assert report.weighted_f1 == 1.0
        assert report.error_taxonomy["correct"] == 5
        assert report.error_taxonomy["wrong_class"] == 0

    def test_sequence_perfect(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]", "[4, 5]", "[1]"]
        pred = ["[1, 2, 3]", "[4, 5]", "[1]"]
        report = evaluate(pred, gt, task, "test")

        assert report.accuracy == 1.0
        assert report.track == "sequence"
        assert report.exact_match == 1.0
        assert report.token_accuracy == 1.0
        assert report.error_taxonomy["correct"] == 3
        assert report.error_taxonomy["length_mismatch"] == 0
        assert report.error_taxonomy["content_error"] == 0
        assert report.error_taxonomy["off_by_one"] == 0


# ===================================================================
# 2. Worst prediction test
# ===================================================================

class TestWorstPrediction:

    def test_classification_all_wrong(self):
        task = _make_classification_task()
        gt = ["A", "B", "A", "B", "A"]
        pred = ["B", "A", "B", "A", "B"]
        report = evaluate(pred, gt, task, "test")

        assert report.accuracy == 0.0
        assert report.per_class_metrics is not None
        assert report.per_class_metrics["A"].precision == 0.0
        assert report.per_class_metrics["B"].precision == 0.0
        assert report.macro_f1 == 0.0
        assert report.weighted_f1 == 0.0
        assert report.error_taxonomy["correct"] == 0
        assert report.error_taxonomy["wrong_class"] == 5

    def test_sequence_all_wrong(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]", "[4, 5]"]
        pred = ["[3, 2, 1]", "[5, 4]"]
        report = evaluate(pred, gt, task, "test")

        assert report.accuracy == 0.0
        assert report.exact_match == 0.0
        assert report.error_taxonomy["correct"] == 0


# ===================================================================
# 3. Known confusion matrix test
# ===================================================================

class TestKnownConfusionMatrix:

    def test_2x2_confusion(self):
        task = _make_classification_task()
        gt = ["A", "A", "A", "B", "B", "B"]
        pred = ["A", "A", "B", "B", "B", "A"]
        report = evaluate(pred, gt, task, "test")

        # Expected confusion matrix (rows=true, cols=predicted):
        # A: [2, 1]  (2 correct A, 1 misclassified as B)
        # B: [1, 2]  (1 misclassified as A, 2 correct B)
        assert report.confusion_matrix == [[2, 1], [1, 2]]
        assert report.accuracy == pytest.approx(4 / 6)

    def test_3x3_confusion(self):
        task = _make_classification_task(n_classes=3)
        gt = ["A", "A", "B", "B", "C", "C"]
        pred = ["A", "B", "B", "C", "C", "A"]
        report = evaluate(pred, gt, task, "test")

        # Expected:
        # A: [1, 1, 0]
        # B: [0, 1, 1]
        # C: [1, 0, 1]
        assert report.confusion_matrix == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        assert report.accuracy == pytest.approx(3 / 6)

    def test_single_class_confusion(self):
        task = _make_classification_task(n_classes=1)
        gt = ["A", "A", "A"]
        pred = ["A", "A", "A"]
        report = evaluate(pred, gt, task, "test")

        assert report.confusion_matrix == [[3]]
        assert report.accuracy == 1.0


# ===================================================================
# 4. Metric consistency: accuracy == diagonal / total
# ===================================================================

class TestMetricConsistency:

    @pytest.mark.parametrize("gt,pred", [
        (["A", "B", "A", "B"], ["A", "B", "B", "A"]),
        (["A", "A", "A"], ["A", "A", "A"]),
        (["A", "B", "C"], ["B", "C", "A"]),
        (["A", "B", "A", "B", "A"], ["A", "A", "A", "B", "B"]),
    ])
    def test_accuracy_matches_confusion_diagonal(self, gt, pred):
        task = _make_classification_task()
        report = evaluate(pred, gt, task, "test")

        cm = report.confusion_matrix
        assert cm is not None
        diagonal_sum = sum(cm[i][i] for i in range(len(cm)))
        total = sum(sum(row) for row in cm)
        expected_acc = diagonal_sum / total if total > 0 else 0.0

        assert abs(report.accuracy - expected_acc) < 1e-9


# ===================================================================
# 5. Classification vs. sequence dispatch
# ===================================================================

class TestTrackDispatch:

    def test_classification_has_classification_fields(self):
        task = _make_classification_task()
        report = evaluate(["A", "B"], ["A", "B"], task, "test")

        assert report.track == "classification"
        assert report.per_class_metrics is not None
        assert report.confusion_matrix is not None
        assert report.class_labels is not None
        assert report.macro_f1 is not None
        assert report.weighted_f1 is not None
        assert report.exact_match is None
        assert report.token_accuracy is None

    def test_sequence_has_sequence_fields(self):
        task = _make_sequence_task()
        report = evaluate(["[1, 2]", "[3]"], ["[1, 2]", "[3]"], task, "test")

        assert report.track == "sequence"
        assert report.exact_match is not None
        assert report.token_accuracy is not None
        assert report.per_class_metrics is None
        assert report.confusion_matrix is None
        assert report.class_labels is None
        assert report.macro_f1 is None
        assert report.weighted_f1 is None

    def test_unknown_track_raises(self):
        task = _make_classification_task()
        # Manually override track
        task.track = "unknown"
        with pytest.raises(ValueError, match="Unknown task track"):
            evaluate(["A"], ["A"], task, "test")


# ===================================================================
# 6. Per-class precision/recall/F1
# ===================================================================

class TestPerClassMetrics:

    def test_binary_per_class(self):
        task = _make_classification_task()
        # 3 A's and 2 B's in ground truth
        gt = ["A", "A", "A", "B", "B"]
        # Predict all A: TP_A=3, FP_A=2, FN_A=0, TP_B=0, FP_B=0, FN_B=2
        pred = ["A", "A", "A", "A", "A"]
        report = evaluate(pred, gt, task, "test")

        pcm = report.per_class_metrics
        assert pcm is not None

        # A: precision = 3/(3+2) = 0.6, recall = 3/3 = 1.0
        assert pcm["A"].precision == pytest.approx(0.6)
        assert pcm["A"].recall == pytest.approx(1.0)
        assert pcm["A"].support == 3

        # B: precision = 0/0 = 0.0, recall = 0/2 = 0.0
        assert pcm["B"].precision == 0.0
        assert pcm["B"].recall == 0.0
        assert pcm["B"].support == 2

    def test_multiclass_per_class(self):
        task = _make_classification_task(n_classes=3)
        gt = ["A", "B", "C", "A", "B", "C"]
        pred = ["A", "B", "C", "A", "B", "C"]
        report = evaluate(pred, gt, task, "test")

        for label in ["A", "B", "C"]:
            assert report.per_class_metrics[label].precision == 1.0
            assert report.per_class_metrics[label].recall == 1.0
            assert report.per_class_metrics[label].f1 == 1.0
            assert report.per_class_metrics[label].support == 2


# ===================================================================
# 7. Macro and weighted F1
# ===================================================================

class TestMacroWeightedF1:

    def test_macro_f1_is_unweighted_mean(self):
        task = _make_classification_task()
        gt = ["A", "A", "A", "B"]
        pred = ["A", "A", "B", "B"]
        report = evaluate(pred, gt, task, "test")

        pcm = report.per_class_metrics
        expected_macro = (pcm["A"].f1 + pcm["B"].f1) / 2
        assert report.macro_f1 == pytest.approx(expected_macro)

    def test_weighted_f1_uses_support(self):
        task = _make_classification_task()
        gt = ["A", "A", "A", "B"]
        pred = ["A", "A", "B", "B"]
        report = evaluate(pred, gt, task, "test")

        pcm = report.per_class_metrics
        total = pcm["A"].support + pcm["B"].support
        expected_weighted = (pcm["A"].f1 * pcm["A"].support +
                             pcm["B"].f1 * pcm["B"].support) / total
        assert report.weighted_f1 == pytest.approx(expected_weighted)

    def test_perfect_macro_weighted_equal_one(self):
        task = _make_classification_task()
        gt = ["A", "B", "A", "B"]
        pred = ["A", "B", "A", "B"]
        report = evaluate(pred, gt, task, "test")

        assert report.macro_f1 == 1.0
        assert report.weighted_f1 == 1.0


# ===================================================================
# 8. Sequence exact match and token accuracy
# ===================================================================

class TestSequenceMetrics:

    def test_exact_match_partial(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]", "[4, 5]", "[7, 8, 9]"]
        pred = ["[1, 2, 3]", "[4, 5]", "[9, 8, 7]"]
        report = evaluate(pred, gt, task, "test")

        assert report.exact_match == pytest.approx(2 / 3)

    def test_token_accuracy_partial(self):
        task = _make_sequence_task()
        # 3 tokens correct out of 3 in first, 2/2 in second, 1/3 in third
        gt = ["[1, 2, 3]", "[4, 5]", "[7, 8, 9]"]
        pred = ["[1, 2, 3]", "[4, 5]", "[7, 0, 0]"]
        report = evaluate(pred, gt, task, "test")

        # Total tokens: 3 + 2 + 3 = 8, correct: 3 + 2 + 1 = 6
        assert report.token_accuracy == pytest.approx(6 / 8)

    def test_token_accuracy_length_mismatch(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]"]
        pred = ["[1, 2]"]
        report = evaluate(pred, gt, task, "test")

        # max_len = 3, correct at positions 0,1 = 2
        assert report.token_accuracy == pytest.approx(2 / 3)

    def test_token_accuracy_empty_sequences(self):
        task = _make_sequence_task()
        gt = ["[]"]
        pred = ["[]"]
        report = evaluate(pred, gt, task, "test")

        assert report.exact_match == 1.0
        # Token accuracy is None when there are 0 tokens to compare
        assert report.token_accuracy is None

    def test_non_parseable_sequence(self):
        task = _make_sequence_task()
        gt = ["not_a_list"]
        pred = ["also_not"]
        report = evaluate(pred, gt, task, "test")

        assert report.exact_match == 0.0
        # Token accuracy None because neither is parseable
        assert report.token_accuracy is None


# ===================================================================
# 9. Error taxonomy
# ===================================================================

class TestErrorTaxonomy:

    def test_classification_error_types(self):
        task = _make_classification_task()
        gt = ["A", "A", "B", "B"]
        pred = ["A", "B", "A", "B"]
        report = evaluate(pred, gt, task, "test")

        assert report.error_taxonomy["correct"] == 2
        assert report.error_taxonomy["wrong_class"] == 2
        assert report.error_taxonomy["unknown_class"] == 0

    def test_classification_unknown_class(self):
        task = _make_classification_task()
        gt = ["A", "B"]
        pred = ["A", "UNKNOWN"]
        report = evaluate(pred, gt, task, "test")

        # "UNKNOWN" is not in the label set derived from gt ∪ pred,
        # but since it's in pred, it becomes a known label.
        # It should still count as wrong_class since it's in the label set.
        assert report.error_taxonomy["correct"] == 1

    def test_sequence_off_by_one(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]"]
        pred = ["[1, 2, 4]"]
        report = evaluate(pred, gt, task, "test")

        assert report.error_taxonomy["correct"] == 0
        assert report.error_taxonomy["off_by_one"] == 1

    def test_sequence_length_mismatch(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]"]
        pred = ["[1, 2]"]
        report = evaluate(pred, gt, task, "test")

        assert report.error_taxonomy["length_mismatch"] == 1

    def test_sequence_content_error(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]"]
        pred = ["[3, 2, 1]"]
        report = evaluate(pred, gt, task, "test")

        assert report.error_taxonomy["content_error"] == 1

    def test_sequence_mixed_errors(self):
        task = _make_sequence_task()
        gt = ["[1, 2, 3]", "[4, 5]", "[7, 8, 9]", "[1]"]
        pred = ["[1, 2, 3]", "[4]", "[9, 8, 7]", "[2]"]
        report = evaluate(pred, gt, task, "test")

        assert report.error_taxonomy["correct"] == 1
        assert report.error_taxonomy["length_mismatch"] == 1
        assert report.error_taxonomy["content_error"] == 1
        assert report.error_taxonomy["off_by_one"] == 1


# ===================================================================
# 10. Metadata-conditioned metrics
# ===================================================================

class TestMetadataConditioned:

    def test_conditioned_by_single_key(self):
        task = _make_classification_task()
        gt = ["A", "A", "B", "B"]
        pred = ["A", "B", "B", "A"]
        metadata = [
            {"tier": "easy"},
            {"tier": "easy"},
            {"tier": "hard"},
            {"tier": "hard"},
        ]

        report = evaluate(pred, gt, task, "test",
                          metadata=metadata, condition_keys=["tier"])

        cond = report.metadata_conditioned_metrics
        assert "tier" in cond
        # easy: 1/2 correct, hard: 1/2 correct
        assert cond["tier"]["easy"] == pytest.approx(0.5)
        assert cond["tier"]["hard"] == pytest.approx(0.5)

    def test_conditioned_by_multiple_keys(self):
        task = _make_classification_task()
        gt = ["A", "A", "B", "B"]
        pred = ["A", "A", "B", "A"]
        metadata = [
            {"tier": "easy", "noise": "none"},
            {"tier": "hard", "noise": "none"},
            {"tier": "easy", "noise": "low"},
            {"tier": "hard", "noise": "low"},
        ]

        report = evaluate(pred, gt, task, "test",
                          metadata=metadata, condition_keys=["tier", "noise"])

        cond = report.metadata_conditioned_metrics
        assert "tier" in cond
        assert "noise" in cond
        # tier=easy: 2/2 correct, tier=hard: 1/2 correct
        assert cond["tier"]["easy"] == pytest.approx(1.0)
        assert cond["tier"]["hard"] == pytest.approx(0.5)
        # noise=none: 2/2, noise=low: 1/2
        assert cond["noise"]["none"] == pytest.approx(1.0)
        assert cond["noise"]["low"] == pytest.approx(0.5)

    def test_no_metadata(self):
        task = _make_classification_task()
        report = evaluate(["A"], ["A"], task, "test")
        assert report.metadata_conditioned_metrics == {}


# ===================================================================
# 11. Edge cases
# ===================================================================

class TestEdgeCases:

    def test_length_mismatch_raises(self):
        task = _make_classification_task()
        with pytest.raises(ValueError, match="predictions length"):
            evaluate(["A", "B"], ["A"], task, "test")

    def test_empty_predictions(self):
        task = _make_classification_task()
        report = evaluate([], [], task, "test")
        assert report.accuracy == 0.0
        assert report.n_samples == 0

    def test_single_sample(self):
        task = _make_classification_task()
        report = evaluate(["A"], ["A"], task, "test")
        assert report.accuracy == 1.0
        assert report.n_samples == 1

    def test_all_same_class(self):
        task = _make_classification_task(n_classes=1)
        gt = ["A", "A", "A", "A"]
        pred = ["A", "A", "A", "A"]
        report = evaluate(pred, gt, task, "test")

        assert report.accuracy == 1.0
        assert report.confusion_matrix == [[4]]
        assert report.per_class_metrics["A"].precision == 1.0
        assert report.per_class_metrics["A"].recall == 1.0

    def test_report_preserves_task_id_and_split(self):
        task = _make_classification_task(task_id="C2.1_and_rule")
        report = evaluate(["A"], ["A"], task, "test_extrapolation")

        assert report.task_id == "C2.1_and_rule"
        assert report.split_name == "test_extrapolation"


# ===================================================================
# 12. evaluate_prediction_result convenience
# ===================================================================

class TestEvaluatePredictionResult:

    def test_from_prediction_result(self):
        @dataclass
        class MockPredResult:
            predictions: List[str]
            true_labels: List[str]
            model_name: str = "test_model"
            train_size: int = 100
            test_size: int = 50

        task = _make_classification_task()
        pr = MockPredResult(
            predictions=["A", "B", "A"],
            true_labels=["A", "B", "B"],
        )

        report = evaluate_prediction_result(pr, task, "test_iid")
        assert report.accuracy == pytest.approx(2 / 3)
        assert report.task_id == task.task_id
        assert report.split_name == "test_iid"


# ===================================================================
# 13. eval_report_to_dict serialization
# ===================================================================

class TestReportSerialization:

    def test_classification_to_dict(self):
        task = _make_classification_task()
        report = evaluate(["A", "B"], ["A", "B"], task, "test")
        d = eval_report_to_dict(report)

        assert d["task_id"] == task.task_id
        assert d["accuracy"] == 1.0
        assert "per_class_metrics" in d
        assert "confusion_matrix" in d
        assert "class_labels" in d
        assert "macro_f1" in d
        assert "weighted_f1" in d
        assert "exact_match" not in d
        assert "token_accuracy" not in d

    def test_sequence_to_dict(self):
        task = _make_sequence_task()
        report = evaluate(["[1, 2]"], ["[1, 2]"], task, "test")
        d = eval_report_to_dict(report)

        assert d["task_id"] == task.task_id
        assert d["accuracy"] == 1.0
        assert "exact_match" in d
        assert "token_accuracy" in d
        assert "per_class_metrics" not in d
        assert "confusion_matrix" not in d

    def test_dict_is_json_serializable(self):
        import json
        task = _make_classification_task()
        report = evaluate(["A", "B", "A"], ["A", "B", "B"], task, "test")
        d = eval_report_to_dict(report)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["accuracy"] == d["accuracy"]


# ===================================================================
# 14. Internal helper unit tests
# ===================================================================

class TestInternalHelpers:

    def test_compute_accuracy(self):
        assert _compute_accuracy(["A", "B"], ["A", "B"]) == 1.0
        assert _compute_accuracy(["A", "B"], ["B", "A"]) == 0.0
        assert _compute_accuracy(["A", "B"], ["A", "A"]) == 0.5
        assert _compute_accuracy([], []) == 0.0

    def test_parse_sequence_str(self):
        assert _parse_sequence_str("[1, 2, 3]") == ["1", "2", "3"]
        assert _parse_sequence_str("[]") == []
        assert _parse_sequence_str("[5]") == ["5"]
        assert _parse_sequence_str("not_a_list") is None
        assert _parse_sequence_str("  [1, 2]  ") == ["1", "2"]

    def test_confusion_matrix_basic(self):
        cm = _compute_confusion_matrix(
            ["A", "B", "A"], ["A", "A", "B"], ["A", "B"]
        )
        # True A, Pred A: 1; True A, Pred B: 1
        # True B, Pred A: 1; True B, Pred B: 0
        assert cm == [[1, 1], [1, 0]]

    def test_per_class_from_confusion(self):
        cm = [[2, 1], [1, 2]]
        pcm = _per_class_from_confusion(cm, ["A", "B"])
        assert pcm["A"].precision == pytest.approx(2 / 3)
        assert pcm["A"].recall == pytest.approx(2 / 3)
        assert pcm["B"].precision == pytest.approx(2 / 3)
        assert pcm["B"].recall == pytest.approx(2 / 3)

    def test_macro_f1_helper(self):
        pcm = {
            "A": PerClassMetrics(precision=1.0, recall=1.0, f1=1.0, support=3),
            "B": PerClassMetrics(precision=0.5, recall=0.5, f1=0.5, support=2),
        }
        assert _macro_f1(pcm) == pytest.approx(0.75)

    def test_weighted_f1_helper(self):
        pcm = {
            "A": PerClassMetrics(precision=1.0, recall=1.0, f1=1.0, support=3),
            "B": PerClassMetrics(precision=0.5, recall=0.5, f1=0.5, support=2),
        }
        expected = (1.0 * 3 + 0.5 * 2) / 5
        assert _weighted_f1(pcm) == pytest.approx(expected)

    def test_sequence_error_taxonomy_helper(self):
        tax = _sequence_error_taxonomy(
            ["[1, 2, 3]", "[1]", "[3, 2, 1]", "[1, 2, 4]"],
            ["[1, 2, 3]", "[1, 2]", "[1, 2, 3]", "[1, 2, 3]"],
        )
        assert tax["correct"] == 1
        assert tax["length_mismatch"] == 1
        assert tax["content_error"] == 1
        assert tax["off_by_one"] == 1

    def test_metadata_conditioned_accuracy_helper(self):
        result = _metadata_conditioned_accuracy(
            ["A", "B", "A"], ["A", "A", "A"],
            [{"k": "x"}, {"k": "x"}, {"k": "y"}],
            ["k"],
        )
        assert result["k"]["x"] == pytest.approx(0.5)
        assert result["k"]["y"] == pytest.approx(1.0)


# ===================================================================
# 15. Integration with real registry tasks
# ===================================================================

class TestIntegrationWithRegistry:

    @pytest.fixture(scope="class")
    def registry(self):
        from src.registry import build_default_registry
        return build_default_registry()

    def test_evaluate_classification_task(self, registry):
        """Evaluate a real classification task from the registry."""
        from src.data_generator import generate_dataset
        from src.models.harness import ModelConfig, ModelFamily, ModelHarness
        from src.splits import split_iid

        task = registry.get("C1.1_numeric_threshold")
        ds = generate_dataset(task, n_samples=200, base_seed=42)
        split = split_iid(ds, train_fraction=0.8, seed=42)

        train_samples = split.train
        test_samples = split.test
        train_inputs = [s.input_data for s in train_samples]
        train_outputs = [s.output_data for s in train_samples]
        test_inputs = [s.input_data for s in test_samples]
        test_outputs = [s.output_data for s in test_samples]

        harness = ModelHarness(ModelConfig(family=ModelFamily.DECISION_TREE))
        pred_result = harness.run(train_inputs, train_outputs, test_inputs, test_outputs)

        report = evaluate_prediction_result(pred_result, task, "test_iid")

        assert report.track == "classification"
        assert report.accuracy >= 0.9  # Decision tree on threshold should be near-perfect
        assert report.confusion_matrix is not None
        assert report.per_class_metrics is not None
        assert report.macro_f1 is not None

    def test_evaluate_sequence_task(self, registry):
        """Evaluate a real sequence task from the registry."""
        from src.data_generator import generate_dataset
        from src.models.harness import ModelConfig, ModelFamily, ModelHarness
        from src.splits import split_iid

        task = registry.get("S1.2_sort")
        ds = generate_dataset(task, n_samples=200, base_seed=42)
        split = split_iid(ds, train_fraction=0.8, seed=42)

        train_samples = split.train
        test_samples = split.test
        train_inputs = [s.input_data for s in train_samples]
        train_outputs = [s.output_data for s in train_samples]
        test_inputs = [s.input_data for s in test_samples]
        test_outputs = [s.output_data for s in test_samples]

        harness = ModelHarness(ModelConfig(family=ModelFamily.SEQUENCE_BASELINE))
        pred_result = harness.run(train_inputs, train_outputs, test_inputs, test_outputs)

        report = evaluate_prediction_result(pred_result, task, "test_iid")

        assert report.track == "sequence"
        assert report.exact_match is not None
        assert report.error_taxonomy is not None
        assert 0.0 <= report.accuracy <= 1.0
