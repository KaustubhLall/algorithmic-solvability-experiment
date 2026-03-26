"""SR-6: Evaluation Engine.

Takes predictions, ground-truth outputs, and a task spec, and produces
a structured evaluation report with metrics appropriate for the task type.

Classification metrics: accuracy, per-class precision/recall/F1, macro F1,
weighted F1, confusion matrix, error taxonomy.

Sequence metrics: exact match accuracy, token-level accuracy, error taxonomy.

Both tracks: metadata-conditioned metric breakdowns.

Used by: SR-7 (Experiment Runner), SR-8 (Report Generator).
Validated by: V-6 (Evaluation Engine Validation).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.registry import TaskSpec


# ===================================================================
# EvalReport dataclass
# ===================================================================

@dataclass
class PerClassMetrics:
    """Precision, recall, and F1 for a single class.

    Attributes:
        precision: TP / (TP + FP), or 0.0 if no predictions for this class.
        recall: TP / (TP + FN), or 0.0 if no true samples for this class.
        f1: Harmonic mean of precision and recall, or 0.0 if both are 0.
        support: Number of true samples for this class.
    """
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvalReport:
    """Structured evaluation report produced by the evaluation engine.

    Attributes:
        task_id: ID of the evaluated task.
        split_name: Name of the split (e.g., "test_iid").
        track: "sequence" or "classification".
        n_samples: Number of evaluated samples.
        accuracy: Overall accuracy (exact match for both tracks).
        per_class_metrics: Per-class precision/recall/F1 (classification only).
        macro_f1: Macro-averaged F1 across classes (classification only).
        weighted_f1: Support-weighted F1 (classification only).
        confusion_matrix: N×N confusion matrix as list of lists (classification only).
        class_labels: Ordered class labels for confusion matrix axes.
        exact_match: Exact match accuracy (sequence only).
        token_accuracy: Per-position token accuracy (sequence only).
        error_taxonomy: Counts of error types.
        metadata_conditioned_metrics: Metric breakdowns by metadata key.
    """
    task_id: str
    split_name: str
    track: str
    n_samples: int
    accuracy: float
    per_class_metrics: Optional[Dict[str, PerClassMetrics]] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    class_labels: Optional[List[str]] = None
    exact_match: Optional[float] = None
    token_accuracy: Optional[float] = None
    error_taxonomy: Dict[str, int] = field(default_factory=dict)
    metadata_conditioned_metrics: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
# Core metric computation
# ===================================================================

def _compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """Compute exact-match accuracy."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(predictions)


def _compute_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str],
) -> List[List[int]]:
    """Compute NxN confusion matrix.

    Rows = true class, Columns = predicted class.
    """
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = [[0] * n for _ in range(n)]
    for pred, true in zip(predictions, ground_truth):
        true_idx = label_to_idx.get(true)
        pred_idx = label_to_idx.get(pred)
        if true_idx is not None and pred_idx is not None:
            cm[true_idx][pred_idx] += 1
    return cm


def _per_class_from_confusion(
    cm: List[List[int]],
    labels: List[str],
) -> Dict[str, PerClassMetrics]:
    """Compute per-class precision, recall, F1 from a confusion matrix."""
    n = len(labels)
    result: Dict[str, PerClassMetrics] = {}

    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        fn = sum(cm[i][j] for j in range(n)) - tp
        support = sum(cm[i][j] for j in range(n))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        result[label] = PerClassMetrics(
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )

    return result


def _macro_f1(per_class: Dict[str, PerClassMetrics]) -> float:
    """Compute macro-averaged F1 (unweighted mean of per-class F1)."""
    if not per_class:
        return 0.0
    return sum(m.f1 for m in per_class.values()) / len(per_class)


def _weighted_f1(per_class: Dict[str, PerClassMetrics]) -> float:
    """Compute support-weighted F1."""
    if not per_class:
        return 0.0
    total_support = sum(m.support for m in per_class.values())
    if total_support == 0:
        return 0.0
    return sum(m.f1 * m.support for m in per_class.values()) / total_support


# ===================================================================
# Error taxonomy
# ===================================================================

def _classification_error_taxonomy(
    predictions: List[str],
    ground_truth: List[str],
    labels: List[str],
) -> Dict[str, int]:
    """Categorize classification errors.

    Error types:
    - "correct": prediction matches ground truth
    - "wrong_class": prediction is a valid class but wrong
    - "unknown_class": prediction is not in the known label set
    """
    taxonomy: Dict[str, int] = {
        "correct": 0,
        "wrong_class": 0,
        "unknown_class": 0,
    }
    label_set = set(labels)

    for pred, true in zip(predictions, ground_truth):
        if pred == true:
            taxonomy["correct"] += 1
        elif pred in label_set:
            taxonomy["wrong_class"] += 1
        else:
            taxonomy["unknown_class"] += 1

    return taxonomy


def _sequence_error_taxonomy(
    predictions: List[str],
    ground_truth: List[str],
) -> Dict[str, int]:
    """Categorize sequence prediction errors.

    Error types:
    - "correct": exact match
    - "length_mismatch": output is wrong length
    - "content_error": correct length but wrong content
    - "off_by_one": correct length, differs in exactly one position
    """
    taxonomy: Dict[str, int] = {
        "correct": 0,
        "length_mismatch": 0,
        "content_error": 0,
        "off_by_one": 0,
    }

    for pred_str, true_str in zip(predictions, ground_truth):
        if pred_str == true_str:
            taxonomy["correct"] += 1
            continue

        # Parse stringified lists for detailed analysis
        pred_tokens = _parse_sequence_str(pred_str)
        true_tokens = _parse_sequence_str(true_str)

        if pred_tokens is None or true_tokens is None:
            taxonomy["content_error"] += 1
            continue

        if len(pred_tokens) != len(true_tokens):
            taxonomy["length_mismatch"] += 1
        else:
            diff_count = sum(1 for a, b in zip(pred_tokens, true_tokens) if a != b)
            if diff_count == 1:
                taxonomy["off_by_one"] += 1
            else:
                taxonomy["content_error"] += 1

    return taxonomy


def _parse_sequence_str(s: str) -> Optional[List[str]]:
    """Try to parse a stringified list like '[1, 2, 3]' into tokens."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [t.strip() for t in inner.split(",")]
    return None


# ===================================================================
# Sequence-specific metrics
# ===================================================================

def _compute_token_accuracy(
    predictions: List[str],
    ground_truth: List[str],
) -> Optional[float]:
    """Compute per-position token accuracy for sequence outputs.

    Only computable when both prediction and ground truth are parseable
    as lists. Positions beyond the shorter sequence are counted as wrong.
    """
    total_tokens = 0
    correct_tokens = 0

    for pred_str, true_str in zip(predictions, ground_truth):
        pred_tokens = _parse_sequence_str(pred_str)
        true_tokens = _parse_sequence_str(true_str)

        if pred_tokens is None or true_tokens is None:
            continue

        max_len = max(len(pred_tokens), len(true_tokens))
        if max_len == 0:
            continue

        total_tokens += max_len
        for i in range(min(len(pred_tokens), len(true_tokens))):
            if pred_tokens[i] == true_tokens[i]:
                correct_tokens += 1

    if total_tokens == 0:
        return None
    return correct_tokens / total_tokens


# ===================================================================
# Metadata-conditioned metrics
# ===================================================================

def _metadata_conditioned_accuracy(
    predictions: List[str],
    ground_truth: List[str],
    metadata: List[Dict[str, Any]],
    condition_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute accuracy broken down by metadata keys.

    For each key in condition_keys, groups samples by their metadata value
    and computes accuracy per group.

    Returns:
        Dict mapping condition_key → {group_value → accuracy}.
    """
    result: Dict[str, Dict[str, float]] = {}

    for key in condition_keys:
        groups: Dict[str, List[Tuple[str, str]]] = {}
        for pred, true, meta in zip(predictions, ground_truth, metadata):
            val = meta.get(key)
            if val is None:
                continue
            group_key = str(val)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append((pred, true))

        if groups:
            group_acc: Dict[str, float] = {}
            for group_val, pairs in sorted(groups.items()):
                correct = sum(1 for p, t in pairs if p == t)
                group_acc[group_val] = correct / len(pairs) if pairs else 0.0
            result[key] = group_acc

    return result


# ===================================================================
# Main evaluate function
# ===================================================================

def evaluate(
    predictions: List[str],
    ground_truth: List[str],
    task: TaskSpec,
    split_name: str,
    metadata: Optional[List[Dict[str, Any]]] = None,
    condition_keys: Optional[List[str]] = None,
) -> EvalReport:
    """Evaluate predictions against ground truth for a given task.

    Dispatches to classification or sequence metrics based on task.track.

    Args:
        predictions: Predicted labels/outputs as strings.
        ground_truth: Ground truth labels/outputs as strings.
        task: The TaskSpec for this evaluation.
        split_name: Name of the split being evaluated (e.g., "test_iid").
        metadata: Optional per-sample metadata dicts for conditioned metrics.
        condition_keys: Optional list of metadata keys to condition on.

    Returns:
        An EvalReport with all computed metrics.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"predictions length ({len(predictions)}) != "
            f"ground_truth length ({len(ground_truth)})"
        )
    if metadata is not None and len(metadata) != len(predictions):
        raise ValueError(
            f"metadata length ({len(metadata)}) != "
            f"predictions length ({len(predictions)})"
        )

    n_samples = len(predictions)
    accuracy = _compute_accuracy(predictions, ground_truth)

    # Metadata-conditioned metrics
    meta_conditioned: Dict[str, Any] = {}
    if metadata and condition_keys:
        meta_conditioned = _metadata_conditioned_accuracy(
            predictions, ground_truth, metadata, condition_keys
        )

    if task.track == "classification":
        return _evaluate_classification(
            predictions, ground_truth, task, split_name,
            n_samples, accuracy, meta_conditioned,
        )
    elif task.track == "sequence":
        return _evaluate_sequence(
            predictions, ground_truth, task, split_name,
            n_samples, accuracy, meta_conditioned,
        )
    else:
        raise ValueError(f"Unknown task track: {task.track}")


def _evaluate_classification(
    predictions: List[str],
    ground_truth: List[str],
    task: TaskSpec,
    split_name: str,
    n_samples: int,
    accuracy: float,
    meta_conditioned: Dict[str, Any],
) -> EvalReport:
    """Compute classification-specific metrics."""
    known_labels = sorted(set(ground_truth))

    cm = _compute_confusion_matrix(predictions, ground_truth, known_labels)
    per_class = _per_class_from_confusion(cm, known_labels)
    macro = _macro_f1(per_class)
    weighted = _weighted_f1(per_class)
    error_tax = _classification_error_taxonomy(predictions, ground_truth, known_labels)

    return EvalReport(
        task_id=task.task_id,
        split_name=split_name,
        track="classification",
        n_samples=n_samples,
        accuracy=accuracy,
        per_class_metrics=per_class,
        macro_f1=macro,
        weighted_f1=weighted,
        confusion_matrix=cm,
        class_labels=known_labels,
        exact_match=None,
        token_accuracy=None,
        error_taxonomy=error_tax,
        metadata_conditioned_metrics=meta_conditioned,
    )


def _evaluate_sequence(
    predictions: List[str],
    ground_truth: List[str],
    task: TaskSpec,
    split_name: str,
    n_samples: int,
    accuracy: float,
    meta_conditioned: Dict[str, Any],
) -> EvalReport:
    """Compute sequence-specific metrics."""
    token_acc = _compute_token_accuracy(predictions, ground_truth)
    error_tax = _sequence_error_taxonomy(predictions, ground_truth)

    return EvalReport(
        task_id=task.task_id,
        split_name=split_name,
        track="sequence",
        n_samples=n_samples,
        accuracy=accuracy,
        per_class_metrics=None,
        macro_f1=None,
        weighted_f1=None,
        confusion_matrix=None,
        class_labels=None,
        exact_match=accuracy,
        token_accuracy=token_acc,
        error_taxonomy=error_tax,
        metadata_conditioned_metrics=meta_conditioned,
    )


# ===================================================================
# Convenience: evaluate a PredictionResult directly
# ===================================================================

def evaluate_prediction_result(
    prediction_result: Any,
    task: TaskSpec,
    split_name: str,
    metadata: Optional[List[Dict[str, Any]]] = None,
    condition_keys: Optional[List[str]] = None,
) -> EvalReport:
    """Evaluate a PredictionResult from the Model Harness.

    Args:
        prediction_result: A PredictionResult object with .predictions and .true_labels.
        task: The TaskSpec.
        split_name: Name of the split.
        metadata: Optional per-sample metadata.
        condition_keys: Optional metadata keys to condition on.

    Returns:
        An EvalReport.
    """
    return evaluate(
        predictions=prediction_result.predictions,
        ground_truth=prediction_result.true_labels,
        task=task,
        split_name=split_name,
        metadata=metadata,
        condition_keys=condition_keys,
    )


# ===================================================================
# Utility: report to dict (for JSON serialization)
# ===================================================================

def eval_report_to_dict(report: EvalReport) -> Dict[str, Any]:
    """Convert an EvalReport to a JSON-serializable dictionary."""
    d: Dict[str, Any] = {
        "task_id": report.task_id,
        "split_name": report.split_name,
        "track": report.track,
        "n_samples": report.n_samples,
        "accuracy": report.accuracy,
        "error_taxonomy": report.error_taxonomy,
        "metadata_conditioned_metrics": report.metadata_conditioned_metrics,
    }

    if report.per_class_metrics is not None:
        d["per_class_metrics"] = {
            label: {
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "support": m.support,
            }
            for label, m in report.per_class_metrics.items()
        }

    if report.macro_f1 is not None:
        d["macro_f1"] = report.macro_f1
    if report.weighted_f1 is not None:
        d["weighted_f1"] = report.weighted_f1
    if report.confusion_matrix is not None:
        d["confusion_matrix"] = report.confusion_matrix
    if report.class_labels is not None:
        d["class_labels"] = report.class_labels
    if report.exact_match is not None:
        d["exact_match"] = report.exact_match
    if report.token_accuracy is not None:
        d["token_accuracy"] = report.token_accuracy

    return d
