"""SR-8: Report Generator.

Produces structured output artifacts for experiment runs:
- config.json
- summary.md
- per-task metrics/errors JSON
- confusion matrix and extrapolation plots
- comparison markdown
- solvability verdicts JSON

Used by: All experiments (via SR-7 reports).
Validated by: V-8 (Report Generator Validation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.registry import TaskRegistry, TaskSpec, build_default_registry
from src.runner import (
    AggregatedResult,
    ExperimentReport,
    SingleRunResult,
    aggregated_result_to_dict,
    experiment_report_to_dict,
    single_result_to_dict,
)


BASELINE_MODEL_NAMES = {"majority_class", "sequence_baseline"}
IID_SPLIT = "iid"


@dataclass(frozen=True)
class SolvabilityVerdict:
    """Operationalized solvability verdict for a single task."""

    task_id: str
    label: str
    score: float
    best_model: Optional[str]
    best_iid_accuracy: Optional[float]
    best_ood_accuracy: Optional[float]
    evidence: Dict[str, bool]
    notes: List[str]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_experiment_dir(output_root: str | Path, experiment_id: str) -> Path:
    """Validate experiment output paths before writing artifacts."""
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _group_by_task(items: Iterable[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for item in items:
        grouped.setdefault(item.task_id, []).append(item)
    return grouped


def _split_result_groups(
    task_results: List[AggregatedResult],
) -> Tuple[List[AggregatedResult], List[AggregatedResult]]:
    iid = [result for result in task_results if result.split_strategy == IID_SPLIT]
    ood = [result for result in task_results if result.split_strategy != IID_SPLIT]
    return iid, ood


def _best_result(results: List[AggregatedResult]) -> Optional[AggregatedResult]:
    if not results:
        return None
    return max(results, key=lambda result: (result.accuracy_mean, -result.accuracy_std))


def _difficulty_rank(split_name: str) -> int:
    ranks = {
        "iid": 0,
        "noise": 1,
        "value_extrapolation": 2,
        "length_extrapolation": 2,
        "distractor": 2,
    }
    return ranks.get(split_name, 1)


def _compute_task_score(
    best_iid: Optional[AggregatedResult],
    best_ood: Optional[AggregatedResult],
    baseline_iid: Optional[AggregatedResult],
    distractor_result: Optional[AggregatedResult],
    n_seeds: int,
) -> float:
    iid_acc = best_iid.accuracy_mean if best_iid else 0.0
    ood_acc = best_ood.accuracy_mean if best_ood else 0.0
    baseline_gap = 0.0
    if best_iid and baseline_iid:
        baseline_gap = max(0.0, best_iid.accuracy_mean - baseline_iid.accuracy_mean)

    stability = 0.0
    if best_iid and n_seeds > 0:
        stability = max(0.0, 1.0 - min(best_iid.accuracy_std, 1.0))

    distractor_robustness = 0.0
    if best_iid and distractor_result:
        distractor_robustness = max(
            0.0,
            1.0 - abs(best_iid.accuracy_mean - distractor_result.accuracy_mean),
        )

    degradation_coherence = 0.5
    if best_iid and best_ood:
        degradation_coherence = max(
            0.0,
            1.0 - max(0.0, best_iid.accuracy_mean - best_ood.accuracy_mean),
        )

    weighted = (
        0.15 * iid_acc
        + 0.25 * ood_acc
        + 0.15 * min(1.0, baseline_gap)
        + 0.10 * stability
        + 0.10 * distractor_robustness
        + 0.10 * 0.0  # sample efficiency not yet available in SR-7/SR-8
        + 0.15 * degradation_coherence
    )
    return round(weighted, 4)


def compute_solvability_verdict(
    task: TaskSpec,
    task_results: List[AggregatedResult],
) -> SolvabilityVerdict:
    """Map aggregated task results to a Section 9.4 verdict label.

    The design document defines the labels in terms of evidence criteria.
    SR-8 operationalizes those criteria using the metrics currently available
    from SR-7: IID accuracy, extrapolation behavior, baseline gap, seed
    stability, and split-wise robustness signals when present.
    """

    iid_results, ood_results = _split_result_groups(task_results)
    non_baseline_results = [
        result for result in task_results
        if result.model_name not in BASELINE_MODEL_NAMES
    ]
    non_baseline_iid = [
        result for result in iid_results
        if result.model_name not in BASELINE_MODEL_NAMES
    ]
    non_baseline_ood = [
        result for result in ood_results
        if result.model_name not in BASELINE_MODEL_NAMES
    ]
    baseline_iid = _best_result([
        result for result in iid_results
        if result.model_name in BASELINE_MODEL_NAMES
    ])
    best_iid = _best_result(non_baseline_iid or iid_results)
    best_ood = _best_result(non_baseline_ood)
    best_overall = _best_result(non_baseline_results or task_results)

    distractor_result = _best_result([
        result for result in ood_results if result.split_strategy == "distractor"
    ])
    composition_like_result = _best_result([
        result for result in ood_results
        if result.split_strategy in {"composition", "template_holdout", "category_combination"}
    ])

    notes: List[str] = []

    criterion_1 = best_iid is not None and best_iid.accuracy_mean >= 0.95
    if not criterion_1:
        notes.append("No model reached high IID accuracy.")

    criterion_2 = best_ood is not None and best_ood.accuracy_mean >= 0.85
    if not criterion_2:
        notes.append("Out-of-distribution generalization evidence is weak or absent.")

    criterion_3 = (
        best_iid is not None
        and baseline_iid is not None
        and (best_iid.accuracy_mean - baseline_iid.accuracy_mean) >= 0.15
    )
    if baseline_iid is None:
        notes.append("Baseline separation could not be measured because no floor baseline ran.")
    elif not criterion_3:
        notes.append("Best IID model did not clearly separate from the floor baseline.")

    criterion_4 = (
        best_iid is not None
        and best_iid.n_seeds >= 2
        and best_iid.accuracy_std <= 0.05
    )
    if best_iid is None or best_iid.n_seeds < 2:
        notes.append("Seed stability is insufficiently measured (<2 seeds).")
    elif not criterion_4:
        notes.append("Seed-to-seed variation is too high for a stable verdict.")

    coherent_degradation = True
    if best_iid is not None and ood_results:
        sorted_ood = sorted(
            ood_results,
            key=lambda result: _difficulty_rank(result.split_strategy),
        )
        coherent_degradation = all(
            result.accuracy_mean <= best_iid.accuracy_mean + 0.05
            for result in sorted_ood
        )
    criterion_5 = bool(ood_results) and coherent_degradation
    if not ood_results:
        notes.append("Coherent degradation cannot be assessed without harder splits.")
    elif not criterion_5:
        notes.append("Split-wise degradation pattern looks inconsistent.")

    criterion_6 = False
    criterion_7 = (
        distractor_result is not None
        and best_iid is not None
        and abs(best_iid.accuracy_mean - distractor_result.accuracy_mean) <= 0.05
    )
    criterion_8 = False
    criterion_9 = (
        composition_like_result is not None
        and composition_like_result.accuracy_mean >= 0.85
    )

    evidence = {
        "criterion_1_high_iid_accuracy": criterion_1,
        "criterion_2_extrapolation_success": criterion_2,
        "criterion_3_baseline_separation": criterion_3,
        "criterion_4_seed_stability": criterion_4,
        "criterion_5_coherent_degradation": criterion_5,
        "criterion_6_counterfactual_sensitivity": criterion_6,
        "criterion_7_distractor_robustness": criterion_7,
        "criterion_8_sample_efficiency": criterion_8,
        "criterion_9_transfer": criterion_9,
    }

    optional_keys = [
        "criterion_6_counterfactual_sensitivity",
        "criterion_7_distractor_robustness",
        "criterion_8_sample_efficiency",
        "criterion_9_transfer",
    ]
    optional_count = sum(1 for key in optional_keys if evidence[key])
    all_minimum = all(evidence[f"criterion_{idx}_{name}"] for idx, name in [
        (1, "high_iid_accuracy"),
        (2, "extrapolation_success"),
        (3, "baseline_separation"),
        (4, "seed_stability"),
        (5, "coherent_degradation"),
    ])

    if all_minimum and optional_count >= 2:
        label = "STRONG"
    elif all_minimum:
        label = "MODERATE"
    elif criterion_1 and any(not evidence[key] for key in [
        "criterion_2_extrapolation_success",
        "criterion_3_baseline_separation",
        "criterion_4_seed_stability",
        "criterion_5_coherent_degradation",
    ]):
        label = "WEAK"
    elif best_overall is not None and best_overall.accuracy_mean < 0.60:
        label = "NEGATIVE"
    else:
        label = "INCONCLUSIVE"

    score = _compute_task_score(
        best_iid=best_iid,
        best_ood=best_ood,
        baseline_iid=baseline_iid,
        distractor_result=distractor_result,
        n_seeds=best_iid.n_seeds if best_iid else 0,
    )

    return SolvabilityVerdict(
        task_id=task.task_id,
        label=label,
        score=score,
        best_model=best_overall.model_name if best_overall else None,
        best_iid_accuracy=best_iid.accuracy_mean if best_iid else None,
        best_ood_accuracy=best_ood.accuracy_mean if best_ood else None,
        evidence=evidence,
        notes=notes,
    )


def compute_solvability_verdicts(
    report: ExperimentReport,
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-task solvability verdict payloads."""

    if registry is None:
        registry = build_default_registry()

    grouped = _group_by_task(report.aggregated_results)
    verdicts: Dict[str, Dict[str, Any]] = {}

    for task_id, task_results in sorted(grouped.items()):
        task = registry.get(task_id)
        verdict = compute_solvability_verdict(task, task_results)
        verdicts[task_id] = {
            "task_id": verdict.task_id,
            "tier": task.tier,
            "track": task.track,
            "label": verdict.label,
            "score": verdict.score,
            "best_model": verdict.best_model,
            "best_iid_accuracy": verdict.best_iid_accuracy,
            "best_ood_accuracy": verdict.best_ood_accuracy,
            "evidence": verdict.evidence,
            "notes": verdict.notes,
        }

    return verdicts


def _aggregate_error_taxonomies(task_single_results: List[SingleRunResult]) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for result in task_single_results:
        key = f"{result.model_name}/{result.split_strategy}"
        bucket = grouped.setdefault(
            key,
            {
                "model_name": result.model_name,
                "split_strategy": result.split_strategy,
                "n_runs": 0,
                "total_counts": {},
            },
        )
        bucket["n_runs"] += 1
        for error_name, count in result.eval_report.error_taxonomy.items():
            bucket["total_counts"][error_name] = (
                bucket["total_counts"].get(error_name, 0) + count
            )

    for bucket in grouped.values():
        bucket["mean_counts"] = {
            name: round(total / bucket["n_runs"], 4)
            for name, total in sorted(bucket["total_counts"].items())
        }

    return dict(sorted(grouped.items()))


def _mean_confusion_matrix(results: List[SingleRunResult]) -> Optional[np.ndarray]:
    label_order: List[str] = []
    aligned_matrices: List[np.ndarray] = []
    for result in results:
        cm = result.eval_report.confusion_matrix
        labels = result.eval_report.class_labels
        if cm is None:
            continue
        if labels is None or len(labels) == 0:
            continue
        arr = np.array(cm, dtype=float)
        if arr.size == 0 or arr.ndim != 2:
            continue
        if arr.shape != (len(labels), len(labels)):
            continue

        for label in labels:
            if label not in label_order:
                label_order.append(label)

        aligned = np.zeros((len(label_order), len(label_order)), dtype=float)
        label_to_index = {label: idx for idx, label in enumerate(label_order)}
        for row_idx, true_label in enumerate(labels):
            for col_idx, pred_label in enumerate(labels):
                aligned[label_to_index[true_label]][label_to_index[pred_label]] = arr[row_idx][col_idx]

        if aligned_matrices:
            expanded: List[np.ndarray] = []
            for matrix in aligned_matrices:
                expanded_matrix = np.zeros((len(label_order), len(label_order)), dtype=float)
                expanded_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
                expanded.append(expanded_matrix)
            aligned_matrices = expanded

        aligned_matrices.append(aligned)

    if not aligned_matrices:
        return None
    return np.mean(aligned_matrices, axis=0)


def _save_confusion_plot(task_dir: Path, task_results: List[SingleRunResult]) -> Optional[str]:
    classification_runs = []
    for result in task_results:
        cm = result.eval_report.confusion_matrix
        labels = result.eval_report.class_labels
        if cm is None or labels is None:
            continue
        if len(cm) == 0 or len(labels) == 0:
            continue
        if len(cm) != len(labels) or any(len(row) != len(labels) for row in cm):
            continue
        classification_runs.append(result)
    if not classification_runs:
        return None

    preferred_runs = [result for result in classification_runs if result.split_strategy == IID_SPLIT]
    selected_runs = preferred_runs or classification_runs
    mean_cm = _mean_confusion_matrix(selected_runs)
    if mean_cm is None or mean_cm.size == 0 or mean_cm.ndim != 2:
        return None

    labels = list(selected_runs[0].eval_report.class_labels or [])
    for result in selected_runs[1:]:
        for label in result.eval_report.class_labels or []:
            if label not in labels:
                labels.append(label)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(mean_cm, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Mean Confusion Matrix")
    if labels:
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels, rotation=0)
    fig.tight_layout()

    out_path = task_dir / "confusion.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path.name


def _save_extrapolation_plot(task_dir: Path, task_results: List[AggregatedResult]) -> str:
    ordered = sorted(
        task_results,
        key=lambda result: (
            _difficulty_rank(result.split_strategy),
            result.split_strategy,
            result.model_name,
        ),
    )
    split_names = list(dict.fromkeys(result.split_strategy for result in ordered))
    x_positions = {split_name: idx for idx, split_name in enumerate(split_names)}

    fig, ax = plt.subplots(figsize=(7, 4))
    models = sorted({result.model_name for result in ordered})
    for model_name in models:
        model_results = [result for result in ordered if result.model_name == model_name]
        x = [x_positions[result.split_strategy] for result in model_results]
        y = [result.accuracy_mean for result in model_results]
        ax.plot(x, y, marker="o", label=model_name)

    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(
        [x_positions[name] for name in split_names],
        split_names,
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Split")
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = task_dir / "extrap_curve.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path.name


def _task_metrics_payload(
    task: TaskSpec,
    aggregated_results: List[AggregatedResult],
    single_results: List[SingleRunResult],
    verdict: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task": {
            "task_id": task.task_id,
            "tier": task.tier,
            "track": task.track,
            "description": task.description,
            "complexity_metadata": task.complexity_metadata,
        },
        "verdict": verdict,
        "aggregated_results": [
            aggregated_result_to_dict(result)
            for result in sorted(
                aggregated_results,
                key=lambda result: (result.model_name, result.split_strategy),
            )
        ],
        "single_results": [
            single_result_to_dict(result)
            for result in sorted(
                single_results,
                key=lambda result: (result.model_name, result.split_strategy, result.seed),
            )
        ],
    }


def _task_summary_row(
    task: TaskSpec,
    task_results: List[AggregatedResult],
    verdict: Dict[str, Any],
) -> Dict[str, Any]:
    iid_results, ood_results = _split_result_groups(task_results)
    best_iid = _best_result(iid_results)
    best_ood = _best_result(ood_results)

    return {
        "task_id": task.task_id,
        "tier": task.tier,
        "track": task.track,
        "best_iid": None if best_iid is None else round(best_iid.accuracy_mean, 4),
        "best_iid_model": None if best_iid is None else best_iid.model_name,
        "best_ood": None if best_ood is None else round(best_ood.accuracy_mean, 4),
        "best_ood_model": None if best_ood is None else best_ood.model_name,
        "verdict": verdict["label"],
        "score": verdict["score"],
    }


def _render_summary_markdown(
    report: ExperimentReport,
    task_rows: List[Dict[str, Any]],
    verdicts: Dict[str, Dict[str, Any]],
) -> str:
    lines = [
        f"# Experiment Summary: {report.experiment_id}",
        "",
        "## Overview",
        "",
        f"- Tasks: {len(report.spec.task_ids)}",
        f"- Models: {len(report.spec.model_configs)}",
        f"- Split strategies: {', '.join(split.value for split in report.spec.split_strategies)}",
        f"- Seeds used: {', '.join(str(seed) for seed in report.seeds_used)}",
        f"- Total single runs: {len(report.single_results)}",
        f"- Total aggregated groups: {len(report.aggregated_results)}",
        f"- Total runtime (s): {report.total_time_seconds:.4f}",
        "",
        "## Task Summary",
        "",
        "| Task | Tier | Track | Best IID | Best OOD | Verdict | Score |",
        "|---|---|---|---|---|---|---|",
    ]

    for row in task_rows:
        best_iid = "N/A" if row["best_iid"] is None else f'{row["best_iid"]:.4f} ({row["best_iid_model"]})'
        best_ood = "N/A" if row["best_ood"] is None else f'{row["best_ood"]:.4f} ({row["best_ood_model"]})'
        lines.append(
            f'| {row["task_id"]} | {row["tier"]} | {row["track"]} | '
            f"{best_iid} | {best_ood} | {row['verdict']} | {row['score']:.4f} |"
        )

    lines.extend(["", "## Verdict Notes", ""])
    for task_id in sorted(verdicts):
        verdict = verdicts[task_id]
        lines.append(f"### {task_id}: {verdict['label']}")
        lines.append("")
        lines.append(f"- Score: {verdict['score']:.4f}")
        if verdict["best_model"] is not None:
            lines.append(f"- Best model: {verdict['best_model']}")
        if verdict["best_iid_accuracy"] is not None:
            lines.append(f"- Best IID accuracy: {verdict['best_iid_accuracy']:.4f}")
        if verdict["best_ood_accuracy"] is not None:
            lines.append(f"- Best OOD accuracy: {verdict['best_ood_accuracy']:.4f}")
        lines.append(f"- Evidence flags: {json.dumps(verdict['evidence'], sort_keys=True)}")
        for note in verdict["notes"]:
            lines.append(f"- Note: {note}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_comparison_markdown(task_rows: List[Dict[str, Any]]) -> str:
    lines = [
        f"# Cross-Task Comparison",
        "",
        "| Task | Tier | Track | Best IID | Best OOD | Verdict | Score |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in task_rows:
        best_iid = "N/A" if row["best_iid"] is None else f"{row['best_iid']:.4f}"
        best_ood = "N/A" if row["best_ood"] is None else f"{row['best_ood']:.4f}"
        lines.append(
            f'| {row["task_id"]} | {row["tier"]} | {row["track"]} | '
            f"{best_iid} | {best_ood} | {row['verdict']} | {row['score']:.4f} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def generate_report(
    report: ExperimentReport,
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> Path:
    """Write experiment artifacts to disk and return the experiment directory."""

    if registry is None:
        registry = build_default_registry()

    base_dir = _ensure_dir(
        _resolve_experiment_dir(output_root, str(report.experiment_id))
    )
    per_task_dir = _ensure_dir(base_dir / "per_task")

    report_dict = experiment_report_to_dict(report)
    _write_json(base_dir / "config.json", report_dict["spec"])

    verdicts = compute_solvability_verdicts(report, registry=registry)
    _write_json(base_dir / "solvability_verdicts.json", verdicts)

    aggregated_by_task = _group_by_task(report.aggregated_results)
    single_by_task = _group_by_task(report.single_results)
    task_rows: List[Dict[str, Any]] = []

    for task_id in sorted(aggregated_by_task):
        task = registry.get(task_id)
        task_dir = _ensure_dir(per_task_dir / task_id)
        task_aggregated = aggregated_by_task.get(task_id, [])
        task_single = single_by_task.get(task_id, [])
        verdict = verdicts[task_id]

        _write_json(
            task_dir / "metrics.json",
            _task_metrics_payload(task, task_aggregated, task_single, verdict),
        )
        _write_json(task_dir / "errors.json", _aggregate_error_taxonomies(task_single))
        _save_extrapolation_plot(task_dir, task_aggregated)
        _save_confusion_plot(task_dir, task_single)

        task_rows.append(_task_summary_row(task, task_aggregated, verdict))

    task_rows.sort(key=lambda row: row["task_id"])
    (base_dir / "summary.md").write_text(
        _render_summary_markdown(report, task_rows, verdicts),
        encoding="utf-8",
    )
    (base_dir / "comparison.md").write_text(
        _render_comparison_markdown(task_rows),
        encoding="utf-8",
    )

    return base_dir


generate_report_artifacts = generate_report
