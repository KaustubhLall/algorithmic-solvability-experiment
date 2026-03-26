"""SR-8: Report Generator.

Produces structured output artifacts (JSON, Markdown, and plots) for an
ExperimentReport produced by SR-7.

Used by: All experiments.
Validated by: V-8 (Report Generator Validation).
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.registry import TaskRegistry, TaskSpec, build_default_registry
from src.runner import (
    AggregatedResult,
    ExperimentReport,
    SingleRunResult,
    aggregated_result_to_dict,
    experiment_report_to_dict,
    single_result_to_dict,
)


BASELINE_MODELS = {
    "majority_class",
    "logistic_regression",
    "decision_tree",
    "knn",
    "sequence_baseline",
}

SPLIT_DIFFICULTY_RANK = {
    "iid": 0,
    "noise": 1,
    "distractor": 1,
    "value_extrapolation": 2,
    "length_extrapolation": 2,
    "feature_subset_shift": 2,
    "category_combination": 2,
    "composition": 3,
    "template_holdout": 3,
    "rule_complexity_increase": 3,
    "adversarial": 4,
    "adversarial_boundary": 4,
}

SOLVABILITY_WEIGHTS = {
    "iid_accuracy": 0.15,
    "extrapolation_accuracy": 0.25,
    "baseline_gap": 0.15,
    "seed_stability": 0.10,
    "distractor_robustness": 0.10,
    "sample_efficiency": 0.10,
    "degradation_coherence": 0.15,
}

LABEL_PRIORITY = {
    "NEGATIVE": 0,
    "INCONCLUSIVE": 1,
    "WEAK": 2,
    "MODERATE": 3,
    "STRONG": 4,
}


@dataclass
class ModelSolvabilityAssessment:
    """Solvability evidence for one model on one task."""

    task_id: str
    model_name: str
    label: str
    score: float
    normalized_score: Optional[float]
    criteria: Dict[str, Optional[bool]]
    evidence: Dict[str, Any]


@dataclass
class TaskSolvabilityVerdict:
    """Final per-task solvability verdict."""

    task_id: str
    label: str
    chosen_model: Optional[str]
    score: float
    normalized_score: Optional[float]
    criteria: Dict[str, Optional[bool]]
    evidence: Dict[str, Any]
    per_model: List[ModelSolvabilityAssessment]


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _clamp(value: Optional[float], lower: float = 0.0, upper: float = 1.0) -> Optional[float]:
    if value is None:
        return None
    return max(lower, min(upper, value))


def _split_rank(split_name: str) -> Tuple[int, str]:
    return (SPLIT_DIFFICULTY_RANK.get(split_name, 99), split_name)


def _group_aggregated_by_task(
    aggregated_results: Iterable[AggregatedResult],
) -> Dict[str, List[AggregatedResult]]:
    grouped: Dict[str, List[AggregatedResult]] = defaultdict(list)
    for result in aggregated_results:
        grouped[result.task_id].append(result)
    return dict(grouped)


def _group_single_by_task(
    single_results: Iterable[SingleRunResult],
) -> Dict[str, List[SingleRunResult]]:
    grouped: Dict[str, List[SingleRunResult]] = defaultdict(list)
    for result in single_results:
        grouped[result.task_id].append(result)
    return dict(grouped)


def _group_aggregated_by_model(
    aggregated_results: Iterable[AggregatedResult],
) -> Dict[str, Dict[str, AggregatedResult]]:
    grouped: Dict[str, Dict[str, AggregatedResult]] = defaultdict(dict)
    for result in aggregated_results:
        grouped[result.model_name][result.split_strategy] = result
    return dict(grouped)


def _group_single_by_model_and_split(
    single_results: Iterable[SingleRunResult],
) -> Dict[str, Dict[str, List[SingleRunResult]]]:
    grouped: Dict[str, Dict[str, List[SingleRunResult]]] = defaultdict(lambda: defaultdict(list))
    for result in single_results:
        grouped[result.model_name][result.split_strategy].append(result)
    return {
        model_name: dict(split_map)
        for model_name, split_map in grouped.items()
    }


def _is_baseline_model(model_name: str) -> bool:
    return model_name in BASELINE_MODELS


def _score_components(
    iid_accuracy: Optional[float],
    extrapolation_accuracy: Optional[float],
    baseline_gap: Optional[float],
    max_accuracy_std: Optional[float],
    distractor_drop: Optional[float],
    degradation_coherent: Optional[bool],
) -> Tuple[float, Optional[float], Dict[str, Optional[float]]]:
    components: Dict[str, Optional[float]] = {
        "iid_accuracy": _clamp(iid_accuracy),
        "extrapolation_accuracy": _clamp(extrapolation_accuracy),
        "baseline_gap": _clamp(None if baseline_gap is None else baseline_gap / 0.25),
        "seed_stability": _clamp(None if max_accuracy_std is None else 1.0 - (max_accuracy_std / 0.10)),
        "distractor_robustness": _clamp(
            None if distractor_drop is None else 1.0 - (max(distractor_drop, 0.0) / 0.20)
        ),
        "sample_efficiency": None,
        "degradation_coherence": None if degradation_coherent is None else float(degradation_coherent),
    }

    total = 0.0
    available_weight = 0.0
    for name, weight in SOLVABILITY_WEIGHTS.items():
        value = components.get(name)
        if value is None:
            continue
        total += weight * value
        available_weight += weight

    normalized = None if available_weight == 0.0 else total / available_weight
    return total, normalized, components


def _degradation_is_coherent(results_by_split: Dict[str, AggregatedResult]) -> Optional[bool]:
    ranked_values: Dict[int, List[float]] = defaultdict(list)
    for split_name, result in results_by_split.items():
        rank = SPLIT_DIFFICULTY_RANK.get(split_name)
        if rank is None:
            continue
        ranked_values[rank].append(result.accuracy_mean)

    if 0 not in ranked_values:
        return None

    averaged = [
        (rank, sum(values) / len(values))
        for rank, values in sorted(ranked_values.items())
    ]

    if len(averaged) < 2:
        return None

    for (_, prev_acc), (_, next_acc) in zip(averaged, averaged[1:]):
        if next_acc > prev_acc + 0.05:
            return False
    return True


def _build_model_assessment(
    task: TaskSpec,
    model_name: str,
    results_by_split: Dict[str, AggregatedResult],
    baseline_by_split: Dict[str, AggregatedResult],
) -> ModelSolvabilityAssessment:
    iid_result = results_by_split.get("iid")
    iid_accuracy = None if iid_result is None else iid_result.accuracy_mean

    non_iid_results = [
        result
        for split_name, result in results_by_split.items()
        if split_name != "iid"
    ]
    non_iid_results.sort(key=lambda result: _split_rank(result.split_strategy))
    non_iid_accuracies = [result.accuracy_mean for result in non_iid_results]
    extrapolation_accuracy = _mean(non_iid_accuracies)
    min_ood_accuracy = min(non_iid_accuracies) if non_iid_accuracies else None

    baseline_gap_values: List[float] = []
    if not _is_baseline_model(model_name):
        for result in non_iid_results:
            baseline_result = baseline_by_split.get(result.split_strategy)
            if baseline_result is not None:
                baseline_gap_values.append(
                    result.accuracy_mean - baseline_result.accuracy_mean
                )
        if not baseline_gap_values and iid_result is not None:
            baseline_iid = baseline_by_split.get("iid")
            if baseline_iid is not None:
                baseline_gap_values.append(iid_result.accuracy_mean - baseline_iid.accuracy_mean)
    baseline_gap = _mean(baseline_gap_values)

    n_seeds = min((result.n_seeds for result in results_by_split.values()), default=0)
    max_accuracy_std = max(
        (result.accuracy_std for result in results_by_split.values()),
        default=None,
    )
    degradation_coherent = _degradation_is_coherent(results_by_split)

    distractor_accuracy = None
    distractor_drop = None
    distractor_result = results_by_split.get("distractor")
    if iid_accuracy is not None and distractor_result is not None:
        distractor_accuracy = distractor_result.accuracy_mean
        distractor_drop = iid_accuracy - distractor_accuracy

    transfer_splits = {"composition", "template_holdout", "rule_complexity_increase"}
    transfer_accuracy = _mean(
        [
            result.accuracy_mean
            for split_name, result in results_by_split.items()
            if split_name in transfer_splits
        ]
    )

    criteria: Dict[str, Optional[bool]] = {
        "criterion_1_high_iid_accuracy": iid_accuracy is not None and iid_accuracy >= 0.95,
        "criterion_2_extrapolation_success": (
            iid_accuracy is not None
            and extrapolation_accuracy is not None
            and min_ood_accuracy is not None
            and min_ood_accuracy >= 0.85
            and (iid_accuracy - extrapolation_accuracy) <= 0.10
        ),
        "criterion_3_baseline_separation": (
            not _is_baseline_model(model_name)
            and baseline_gap is not None
            and baseline_gap >= 0.10
        ),
        "criterion_4_seed_stability": (
            n_seeds >= 5
            and max_accuracy_std is not None
            and max_accuracy_std <= 0.05
        ),
        "criterion_5_coherent_degradation": degradation_coherent,
        "criterion_6_counterfactual_sensitivity": None,
        "criterion_7_distractor_robustness": (
            iid_accuracy is not None
            and distractor_accuracy is not None
            and (iid_accuracy - distractor_accuracy) <= 0.05
        ),
        "criterion_8_sample_efficiency": None,
        "criterion_9_transfer": (
            transfer_accuracy is not None and transfer_accuracy >= 0.75
        ),
    }

    required_keys = [
        "criterion_1_high_iid_accuracy",
        "criterion_2_extrapolation_success",
        "criterion_3_baseline_separation",
        "criterion_4_seed_stability",
        "criterion_5_coherent_degradation",
    ]
    optional_keys = [
        "criterion_6_counterfactual_sensitivity",
        "criterion_7_distractor_robustness",
        "criterion_8_sample_efficiency",
        "criterion_9_transfer",
    ]

    required_met = all(criteria[key] is True for key in required_keys)
    optional_true_count = sum(criteria[key] is True for key in optional_keys)

    if required_met:
        label = "STRONG" if optional_true_count >= 2 else "MODERATE"
    elif criteria["criterion_1_high_iid_accuracy"]:
        label = "WEAK"
    elif iid_accuracy is not None and iid_accuracy < 0.60 and (
        extrapolation_accuracy is None or extrapolation_accuracy < 0.60
    ):
        label = "NEGATIVE"
    else:
        label = "INCONCLUSIVE"

    score, normalized_score, score_components = _score_components(
        iid_accuracy=iid_accuracy,
        extrapolation_accuracy=extrapolation_accuracy,
        baseline_gap=baseline_gap,
        max_accuracy_std=max_accuracy_std,
        distractor_drop=distractor_drop,
        degradation_coherent=degradation_coherent,
    )

    evidence: Dict[str, Any] = {
        "track": task.track,
        "tier": task.tier,
        "iid_accuracy": iid_accuracy,
        "extrapolation_accuracy": extrapolation_accuracy,
        "min_ood_accuracy": min_ood_accuracy,
        "baseline_gap": baseline_gap,
        "n_seeds": n_seeds,
        "max_accuracy_std": max_accuracy_std,
        "distractor_accuracy": distractor_accuracy,
        "distractor_drop": distractor_drop,
        "transfer_accuracy": transfer_accuracy,
        "available_splits": sorted(results_by_split.keys(), key=_split_rank),
        "score_components": score_components,
    }

    return ModelSolvabilityAssessment(
        task_id=task.task_id,
        model_name=model_name,
        label=label,
        score=score,
        normalized_score=normalized_score,
        criteria=criteria,
        evidence=evidence,
    )


def assess_task_solvability(
    task: TaskSpec,
    aggregated_results: Sequence[AggregatedResult],
) -> TaskSolvabilityVerdict:
    """Assess solvability for one task from aggregated experiment results."""

    results_by_model = _group_aggregated_by_model(aggregated_results)
    baseline_by_split: Dict[str, AggregatedResult] = {}
    for result in aggregated_results:
        if not _is_baseline_model(result.model_name):
            continue
        incumbent = baseline_by_split.get(result.split_strategy)
        if incumbent is None or result.accuracy_mean > incumbent.accuracy_mean:
            baseline_by_split[result.split_strategy] = result

    assessments = [
        _build_model_assessment(task, model_name, split_map, baseline_by_split)
        for model_name, split_map in sorted(results_by_model.items())
    ]

    positive = [
        assessment
        for assessment in assessments
        if assessment.label in {"STRONG", "MODERATE", "WEAK"}
    ]
    if positive:
        winner = max(
            positive,
            key=lambda assessment: (
                LABEL_PRIORITY[assessment.label],
                assessment.normalized_score if assessment.normalized_score is not None else -1.0,
                assessment.score,
            ),
        )
    elif assessments and all(assessment.label == "NEGATIVE" for assessment in assessments):
        winner = max(assessments, key=lambda assessment: assessment.score)
    elif assessments:
        winner = max(
            assessments,
            key=lambda assessment: (
                LABEL_PRIORITY[assessment.label],
                assessment.normalized_score if assessment.normalized_score is not None else -1.0,
                assessment.score,
            ),
        )
        winner = ModelSolvabilityAssessment(
            task_id=winner.task_id,
            model_name=winner.model_name,
            label="INCONCLUSIVE",
            score=winner.score,
            normalized_score=winner.normalized_score,
            criteria=winner.criteria,
            evidence=winner.evidence,
        )
    else:
        winner = ModelSolvabilityAssessment(
            task_id=task.task_id,
            model_name="",
            label="INCONCLUSIVE",
            score=0.0,
            normalized_score=None,
            criteria={},
            evidence={"reason": "no_results"},
        )

    return TaskSolvabilityVerdict(
        task_id=task.task_id,
        label=winner.label,
        chosen_model=winner.model_name or None,
        score=winner.score,
        normalized_score=winner.normalized_score,
        criteria=winner.criteria,
        evidence=winner.evidence,
        per_model=assessments,
    )


def task_verdict_to_dict(verdict: TaskSolvabilityVerdict) -> Dict[str, Any]:
    """Convert a task verdict into a JSON-serializable dict."""

    return {
        "task_id": verdict.task_id,
        "label": verdict.label,
        "chosen_model": verdict.chosen_model,
        "score": verdict.score,
        "normalized_score": verdict.normalized_score,
        "criteria": verdict.criteria,
        "evidence": verdict.evidence,
        "per_model": [asdict(assessment) for assessment in verdict.per_model],
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_experiment_dir(output_root: str | Path, experiment_id: str) -> Path:
    """Validate experiment output paths before removing or writing directories."""
    output_root_path = Path(output_root)
    output_root_resolved = output_root_path.resolve()

    if not experiment_id or experiment_id in {".", ".."}:
        raise ValueError(f"Invalid experiment_id: {experiment_id!r}")
    if "/" in experiment_id or "\\" in experiment_id:
        raise ValueError(
            f"experiment_id must not contain path separators: {experiment_id!r}"
        )

    experiment_dir = output_root_path / experiment_id
    experiment_dir_resolved = experiment_dir.resolve()
    try:
        experiment_dir_resolved.relative_to(output_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Resolved experiment directory is outside output_root: {experiment_dir_resolved}"
        ) from exc

    return experiment_dir


def _has_plottable_confusion_matrix(result: SingleRunResult) -> bool:
    report = result.eval_report
    matrix = report.confusion_matrix
    labels = report.class_labels
    if matrix is None or labels is None:
        return False
    if len(matrix) == 0 or len(labels) == 0:
        return False
    if len(matrix) != len(labels):
        return False
    return all(len(row) == len(labels) for row in matrix)


def _pick_confusion_source(single_results: Sequence[SingleRunResult]) -> Optional[SingleRunResult]:
    candidates = [
        result
        for result in single_results
        if result.eval_report.track == "classification"
        and _has_plottable_confusion_matrix(result)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda result: (
            result.split_strategy == "iid",
            result.eval_report.accuracy,
            -_split_rank(result.split_strategy)[0],
        ),
    )


def _plot_confusion_matrix(path: Path, result: SingleRunResult) -> None:
    report = result.eval_report
    if not _has_plottable_confusion_matrix(result):
        raise ValueError("Result does not contain a plottable confusion matrix")
    assert report.confusion_matrix is not None
    assert report.class_labels is not None

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    image = ax.imshow(report.confusion_matrix, cmap="Blues")
    ax.set_title(f"{result.task_id} - {result.model_name} ({result.split_strategy})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(report.class_labels)))
    ax.set_yticks(range(len(report.class_labels)))
    ax.set_xticklabels(report.class_labels, rotation=45, ha="right")
    ax.set_yticklabels(report.class_labels)

    for row_idx, row in enumerate(report.confusion_matrix):
        for col_idx, value in enumerate(row):
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_extrapolation_curve(path: Path, aggregated_results: Sequence[AggregatedResult]) -> None:
    split_order = sorted(
        {result.split_strategy for result in aggregated_results},
        key=_split_rank,
    )

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if not split_order:
        ax.text(0.5, 0.5, "No aggregated results", ha="center", va="center")
        ax.set_axis_off()
    else:
        grouped: Dict[str, List[AggregatedResult]] = defaultdict(list)
        for result in aggregated_results:
            grouped[result.model_name].append(result)

        x_positions = list(range(len(split_order)))
        for model_name, results in sorted(grouped.items()):
            by_split = {result.split_strategy: result.accuracy_mean for result in results}
            y_values = [by_split.get(split_name) for split_name in split_order]
            ax.plot(
                x_positions,
                y_values,
                marker="o",
                linewidth=1.8,
                label=model_name,
            )

        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Split")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(split_order, rotation=30, ha="right")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_metrics_payload(
    task: TaskSpec,
    aggregated_results: Sequence[AggregatedResult],
    single_results: Sequence[SingleRunResult],
) -> Dict[str, Any]:
    grouped_aggregated = _group_aggregated_by_model(aggregated_results)
    grouped_single = _group_single_by_model_and_split(single_results)

    models: Dict[str, Any] = {}
    for model_name in sorted(set(grouped_aggregated) | set(grouped_single)):
        split_payloads: Dict[str, Any] = {}
        split_names = sorted(
            set(grouped_aggregated.get(model_name, {})) | set(grouped_single.get(model_name, {})),
            key=_split_rank,
        )
        for split_name in split_names:
            aggregated_result = grouped_aggregated.get(model_name, {}).get(split_name)
            single_split_results = grouped_single.get(model_name, {}).get(split_name, [])
            split_payloads[split_name] = {
                "aggregated": None if aggregated_result is None else aggregated_result_to_dict(aggregated_result),
                "runs": [single_result_to_dict(result) for result in single_split_results],
            }
        models[model_name] = {"splits": split_payloads}

    return {
        "task_id": task.task_id,
        "tier": task.tier,
        "track": task.track,
        "description": task.description,
        "complexity_metadata": task.complexity_metadata,
        "aggregated_results": [
            aggregated_result_to_dict(result)
            for result in sorted(
                aggregated_results,
                key=lambda result: (result.model_name, _split_rank(result.split_strategy)),
            )
        ],
        "single_results": [
            single_result_to_dict(result)
            for result in sorted(
                single_results,
                key=lambda result: (result.model_name, _split_rank(result.split_strategy), result.seed),
            )
        ],
        "models": models,
    }


def _build_errors_payload(
    task: TaskSpec,
    single_results: Sequence[SingleRunResult],
) -> Dict[str, Any]:
    grouped = _group_single_by_model_and_split(single_results)
    by_model_and_split: Dict[str, Any] = {}

    for model_name, split_map in sorted(grouped.items()):
        model_payload: Dict[str, Any] = {}
        for split_name, results in sorted(split_map.items(), key=lambda item: _split_rank(item[0])):
            totals: Dict[str, int] = defaultdict(int)
            for result in results:
                for key, value in result.eval_report.error_taxonomy.items():
                    totals[key] += int(value)

            per_seed = [
                {
                    "seed": result.seed,
                    "error_taxonomy": dict(result.eval_report.error_taxonomy),
                }
                for result in sorted(results, key=lambda result: result.seed)
            ]

            mean_error_rates: Dict[str, float] = {}
            for error_key in sorted(totals.keys()):
                rates: List[float] = []
                for result in results:
                    total_errors = sum(result.eval_report.error_taxonomy.values())
                    if total_errors == 0:
                        continue
                    rates.append(result.eval_report.error_taxonomy.get(error_key, 0) / total_errors)
                mean_error_rates[error_key] = 0.0 if not rates else float(sum(rates) / len(rates))

            model_payload[split_name] = {
                "total_counts": dict(sorted(totals.items())),
                "mean_error_rates": mean_error_rates,
                "per_seed": per_seed,
            }
        by_model_and_split[model_name] = model_payload

    return {
        "task_id": task.task_id,
        "track": task.track,
        "errors": by_model_and_split,
    }


def _best_accuracy_by_split(
    aggregated_results: Sequence[AggregatedResult],
    split_name: str,
) -> Tuple[Optional[str], Optional[float]]:
    candidates = [
        result
        for result in aggregated_results
        if result.split_strategy == split_name
    ]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda result: result.accuracy_mean)
    return best.model_name, best.accuracy_mean


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _build_summary_markdown(
    report: ExperimentReport,
    tasks: Sequence[TaskSpec],
    verdicts: Dict[str, TaskSolvabilityVerdict],
    aggregated_by_task: Dict[str, List[AggregatedResult]],
) -> str:
    verdict_counts: Dict[str, int] = defaultdict(int)
    for verdict in verdicts.values():
        verdict_counts[verdict.label] += 1

    lines = [
        f"# Experiment Summary: {report.experiment_id}",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Tasks: {len(tasks)}",
        f"- Models: {len(report.spec.model_configs)}",
        f"- Split strategies: {', '.join(split.value for split in report.spec.split_strategies)}",
        f"- Seeds used: {', '.join(str(seed) for seed in report.seeds_used)}",
        f"- Total runtime (s): {report.total_time_seconds:.3f}",
        "",
        "## Solvability Counts",
        "",
        "| Label | Count |",
        "|---|---|",
    ]

    for label in ["STRONG", "MODERATE", "WEAK", "INCONCLUSIVE", "NEGATIVE"]:
        lines.append(f"| {label} | {verdict_counts.get(label, 0)} |")

    lines.extend(
        [
            "",
            "## Task Outcomes",
            "",
            "| Task | Track | Best IID model | Best IID acc | Best OOD model | Best OOD acc | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for task in sorted(tasks, key=lambda task: task.task_id):
        task_results = aggregated_by_task.get(task.task_id, [])
        best_iid_model, best_iid_accuracy = _best_accuracy_by_split(task_results, "iid")

        non_iid_candidates = [
            result for result in task_results if result.split_strategy != "iid"
        ]
        if non_iid_candidates:
            best_ood = max(non_iid_candidates, key=lambda result: result.accuracy_mean)
            best_ood_model = best_ood.model_name
            best_ood_accuracy = best_ood.accuracy_mean
        else:
            best_ood_model = None
            best_ood_accuracy = None

        verdict = verdicts[task.task_id]
        lines.append(
            "| "
            + " | ".join(
                [
                    task.task_id,
                    task.track,
                    best_iid_model or "n/a",
                    _format_metric(best_iid_accuracy),
                    best_ood_model or "n/a",
                    _format_metric(best_ood_accuracy),
                    verdict.label,
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def _build_comparison_markdown(
    tasks: Sequence[TaskSpec],
    verdicts: Dict[str, TaskSolvabilityVerdict],
    aggregated_by_task: Dict[str, List[AggregatedResult]],
) -> str:
    lines = [
        "# Cross-Task Comparison",
        "",
        "| Task | Tier | Chosen model | Verdict | Normalized score | Best IID acc | Best OOD acc |",
        "|---|---|---|---|---|---|---|",
    ]

    for task in sorted(tasks, key=lambda task: task.task_id):
        verdict = verdicts[task.task_id]
        task_results = aggregated_by_task.get(task.task_id, [])
        _, best_iid_accuracy = _best_accuracy_by_split(task_results, "iid")
        non_iid = [result.accuracy_mean for result in task_results if result.split_strategy != "iid"]
        best_ood_accuracy = max(non_iid) if non_iid else None

        lines.append(
            "| "
            + " | ".join(
                [
                    task.task_id,
                    task.tier,
                    verdict.chosen_model or "n/a",
                    verdict.label,
                    _format_metric(verdict.normalized_score),
                    _format_metric(best_iid_accuracy),
                    _format_metric(best_ood_accuracy),
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def generate_report_artifacts(
    report: ExperimentReport,
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> Path:
    """Write SR-8 experiment artifacts to disk.

    Args:
        report: Experiment report produced by SR-7.
        output_root: Root directory under which to create the experiment folder.
        registry: Optional registry to resolve task metadata.

    Returns:
        Path to the generated experiment directory.
    """

    if registry is None:
        registry = build_default_registry()

    experiment_dir = _resolve_experiment_dir(output_root, str(report.experiment_id))
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    per_task_dir = experiment_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=True)

    tasks = [registry.get(task_id) for task_id in report.spec.task_ids]
    aggregated_by_task = _group_aggregated_by_task(report.aggregated_results)
    single_by_task = _group_single_by_task(report.single_results)

    verdicts = {
        task.task_id: assess_task_solvability(task, aggregated_by_task.get(task.task_id, []))
        for task in tasks
    }

    config_payload = {
        "experiment_id": report.experiment_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": experiment_report_to_dict(report)["spec"],
        "seeds_used": report.seeds_used,
        "total_time_seconds": report.total_time_seconds,
    }
    _write_json(experiment_dir / "config.json", config_payload)

    for task in tasks:
        task_dir = per_task_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        task_aggregated = aggregated_by_task.get(task.task_id, [])
        task_single = single_by_task.get(task.task_id, [])
        metrics_payload = _build_metrics_payload(task, task_aggregated, task_single)
        errors_payload = _build_errors_payload(task, task_single)

        _write_json(task_dir / "metrics.json", metrics_payload)
        _write_json(task_dir / "errors.json", errors_payload)
        _plot_extrapolation_curve(task_dir / "extrap_curve.png", task_aggregated)

        confusion_source = _pick_confusion_source(task_single)
        if confusion_source is not None:
            _plot_confusion_matrix(task_dir / "confusion.png", confusion_source)

    summary_md = _build_summary_markdown(report, tasks, verdicts, aggregated_by_task)
    comparison_md = _build_comparison_markdown(tasks, verdicts, aggregated_by_task)
    (experiment_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    (experiment_dir / "comparison.md").write_text(comparison_md, encoding="utf-8")
    _write_json(
        experiment_dir / "solvability_verdicts.json",
        {
            "experiment_id": report.experiment_id,
            "verdicts": {
                task_id: task_verdict_to_dict(verdict)
                for task_id, verdict in sorted(verdicts.items())
            },
        },
    )

    return experiment_dir
