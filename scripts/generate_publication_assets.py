from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    }
)


RESULT_EXPERIMENTS = [
    "EXP-0.1",
    "EXP-0.2",
    "EXP-0.3",
    "EXP-S1",
    "EXP-S2",
    "EXP-S3",
    "EXP-C1",
    "EXP-C2",
    "EXP-C3",
    "EXP-D1",
    "EXP-D2",
    "EXP-D3",
    "EXP-D4",
    "EXP-D5",
    "EXP-B1",
    "EXP-B2",
]

BASELINE_EXPERIMENTS = ["EXP-0.3", "EXP-S1", "EXP-S2", "EXP-S3", "EXP-C1", "EXP-C2", "EXP-C3"]
LABEL_ORDER = ["NEGATIVE", "WEAK", "INCONCLUSIVE", "MODERATE", "STRONG"]
TRACK_ORDER = ["sequence", "classification"]
TRACK_DISPLAY = {"sequence": "Sequence", "classification": "Classification"}
TRACK_COLORS = {"sequence": "#4C78A8", "classification": "#F58518"}
LABEL_RANK = {label: idx for idx, label in enumerate(LABEL_ORDER)}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _mean_or_none(values: List[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _parse_runtime_seconds(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"Total runtime \(s\): ([0-9.]+)",
        r"Runtime \(s\): ([0-9.]+)",
        r"Wall-clock runtime \(s\): ([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return float(match.group(1))
    return None


def _baseline_rows(results_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for experiment_id in BASELINE_EXPERIMENTS:
        verdicts = _read_json(results_root / experiment_id / "solvability_verdicts.json")
        for task_id, payload in sorted(verdicts.items()):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "task_id": task_id,
                    "tier": payload["tier"],
                    "track": payload["track"],
                    "label": payload["label"],
                    "score": payload["score"],
                    "best_model": payload.get("best_model"),
                    "best_iid_model": payload.get("best_iid_model"),
                    "best_ood_model": payload.get("best_ood_model"),
                    "best_iid_accuracy": payload.get("best_iid_accuracy"),
                    "best_ood_accuracy": payload.get("best_ood_accuracy"),
                }
            )
    return rows


def _task_inventory_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    inventory: Counter[tuple[str, str]] = Counter()
    for row in rows:
        inventory[(row["track"], row["tier"])] += 1
    inventory_rows = []
    for track in TRACK_ORDER:
        tiers = sorted({tier for row_track, tier in inventory if row_track == track})
        for tier in tiers:
            inventory_rows.append(
                {
                    "track": track,
                    "tier": tier,
                    "task_count": inventory[(track, tier)],
                }
            )
    return inventory_rows


def _baseline_track_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary_rows = []
    for track in TRACK_ORDER:
        track_rows = [row for row in rows if row["track"] == track]
        iid_values = [float(row["best_iid_accuracy"]) for row in track_rows if row["best_iid_accuracy"] is not None]
        ood_values = [float(row["best_ood_accuracy"]) for row in track_rows if row["best_ood_accuracy"] is not None]
        label_counts = Counter(row["label"] for row in track_rows)
        summary_rows.append(
            {
                "track": track,
                "task_count": len(track_rows),
                "mean_score": _round_or_none(_mean_or_none([float(row["score"]) for row in track_rows])),
                "mean_best_iid_accuracy": _round_or_none(_mean_or_none(iid_values)),
                "mean_best_ood_accuracy": _round_or_none(_mean_or_none(ood_values)),
                "negative": label_counts.get("NEGATIVE", 0),
                "weak": label_counts.get("WEAK", 0),
                "inconclusive": label_counts.get("INCONCLUSIVE", 0),
                "moderate": label_counts.get("MODERATE", 0),
                "strong": label_counts.get("STRONG", 0),
            }
        )
    return summary_rows


def _diagnostic_rows(results_root: Path) -> Dict[str, Any]:
    d1 = _read_json(results_root / "EXP-D1" / "sample_efficiency.json")
    d2 = _read_json(results_root / "EXP-D2" / "distractor_robustness.json")
    d3 = _read_json(results_root / "EXP-D3" / "noise_robustness.json")
    d4 = _read_json(results_root / "EXP-D4" / "feature_importance_alignment.json")
    d5 = _read_json(results_root / "EXP-D5" / "solvability_calibration.json")

    d1_rows = []
    for task_id, payload in sorted(d1["task_curves"].items()):
        d1_rows.append(
            {
                "task_id": task_id,
                "track": payload["track"],
                "model_name": payload["model_name"],
                "auc": payload["auc"],
                "delta_vs_control_auc": payload.get("delta_vs_control_auc"),
                "sample_efficiency_score": payload.get("sample_efficiency_score"),
                "criterion_8_sample_efficiency": payload.get("criterion_8_sample_efficiency"),
            }
        )

    d2_rows = []
    for task_id, payload in sorted(d2["task_summary"].items()):
        d2_rows.append(
            {
                "task_id": task_id,
                "selected_model": payload["selected_model"],
                "baseline_accuracy": payload["baseline_accuracy"],
                "max_distractor_accuracy": payload["max_distractor_accuracy"],
                "accuracy_drop": payload["accuracy_drop"],
                "distractor_robustness_score": payload["distractor_robustness_score"],
                "criterion_7_distractor_robustness": payload["criterion_7_distractor_robustness"],
            }
        )

    d3_rows = []
    for task_id, payload in sorted(d3["task_summary"].items()):
        d3_rows.append(
            {
                "task_id": task_id,
                "selected_model": payload["selected_model"],
                "clean_accuracy": payload["clean_accuracy"],
                "max_noise_accuracy": payload["max_noise_accuracy"],
                "noise_accuracy_drop": payload["noise_accuracy_drop"],
                "smooth_degradation": payload["smooth_degradation"],
            }
        )

    d4_rows = []
    for task_id, payload in sorted(d4["results"].items()):
        for model_name, model_payload in sorted(payload["models"].items()):
            d4_rows.append(
                {
                    "task_id": task_id,
                    "model_name": model_name,
                    "precision_at_k_mean": model_payload["precision_at_k_mean"],
                    "jaccard_at_k_mean": model_payload["jaccard_at_k_mean"],
                    "mean_relevant_rank": model_payload["mean_relevant_rank"],
                    "accuracy_mean": model_payload["accuracy_mean"],
                }
            )

    d5_rows = []
    for task_id, payload in sorted(d5["tasks"].items()):
        d5_rows.append(
            {
                "task_id": task_id,
                "track": payload["track"],
                "tier": payload["tier"],
                "baseline_label": payload["baseline_label"],
                "calibrated_label": payload["calibrated_label"],
                "baseline_score": payload["baseline_score"],
                "calibrated_score": payload["calibrated_score"],
                "best_model": payload["best_model"],
                "best_iid_accuracy": payload["best_iid_accuracy"],
                "best_ood_accuracy": payload["best_ood_accuracy"],
                "sample_efficiency_score": payload["diagnostic_support"]["sample_efficiency_score"],
                "distractor_robustness_score": payload["diagnostic_support"]["distractor_robustness_score"],
            }
        )

    return {
        "d1": d1_rows,
        "d2": d2_rows,
        "d3": d3_rows,
        "d4": d4_rows,
        "d5": d5_rows,
        "d5_checks": d5["calibration_checks"],
    }


def _diagnostic_overview_rows(diagnostic: Dict[str, Any]) -> List[Dict[str, Any]]:
    d1_rows = diagnostic["d1"]
    d2_rows = diagnostic["d2"]
    d3_rows = diagnostic["d3"]
    d4_rows = diagnostic["d4"]
    d5_rows = diagnostic["d5"]

    return [
        {
            "diagnostic_id": "EXP-D1",
            "unit": "tasks",
            "evaluated": len(d1_rows),
            "positive_outcomes": sum(1 for row in d1_rows if row["criterion_8_sample_efficiency"]),
            "rate": _round_or_none(
                sum(1 for row in d1_rows if row["criterion_8_sample_efficiency"]) / len(d1_rows)
                if d1_rows
                else None
            ),
            "note": "Criterion 8 pass rate across algorithmic tasks and controls.",
        },
        {
            "diagnostic_id": "EXP-D2",
            "unit": "tasks",
            "evaluated": len(d2_rows),
            "positive_outcomes": sum(1 for row in d2_rows if row["criterion_7_distractor_robustness"]),
            "rate": _round_or_none(
                sum(1 for row in d2_rows if row["criterion_7_distractor_robustness"]) / len(d2_rows)
                if d2_rows
                else None
            ),
            "note": "Distractor-robust tasks among the D2 subset.",
        },
        {
            "diagnostic_id": "EXP-D3",
            "unit": "tasks",
            "evaluated": len(d3_rows),
            "positive_outcomes": sum(1 for row in d3_rows if row["smooth_degradation"]),
            "rate": _round_or_none(
                sum(1 for row in d3_rows if row["smooth_degradation"]) / len(d3_rows)
                if d3_rows
                else None
            ),
            "note": "Tasks with smooth accuracy degradation under noise.",
        },
        {
            "diagnostic_id": "EXP-D4",
            "unit": "task_model_pairs",
            "evaluated": len(d4_rows),
            "positive_outcomes": sum(1 for row in d4_rows if float(row["precision_at_k_mean"]) >= 0.9999),
            "rate": _round_or_none(
                sum(1 for row in d4_rows if float(row["precision_at_k_mean"]) >= 0.9999) / len(d4_rows)
                if d4_rows
                else None
            ),
            "note": "Feature-alignment evaluations with precision@k effectively equal to 1.0.",
        },
        {
            "diagnostic_id": "EXP-D5",
            "unit": "tasks",
            "evaluated": len(d5_rows),
            "positive_outcomes": sum(
                1 for row in d5_rows if row["baseline_label"] != row["calibrated_label"]
            ),
            "rate": _round_or_none(
                sum(1 for row in d5_rows if row["baseline_label"] != row["calibrated_label"]) / len(d5_rows)
                if d5_rows
                else None
            ),
            "note": "Share of tasks whose label changed after diagnostic calibration.",
        },
    ]


def _calibration_overview_rows(d5_rows: List[Dict[str, Any]], checks: Dict[str, bool]) -> List[Dict[str, Any]]:
    rows = []
    for track in ["all", *TRACK_ORDER]:
        track_rows = d5_rows if track == "all" else [row for row in d5_rows if row["track"] == track]
        upgrades = 0
        downgrades = 0
        unchanged = 0
        for row in track_rows:
            baseline_rank = LABEL_RANK[row["baseline_label"]]
            calibrated_rank = LABEL_RANK[row["calibrated_label"]]
            if calibrated_rank > baseline_rank:
                upgrades += 1
            elif calibrated_rank < baseline_rank:
                downgrades += 1
            else:
                unchanged += 1
        rows.append(
            {
                "track": track,
                "task_count": len(track_rows),
                "upgrades": upgrades,
                "downgrades": downgrades,
                "unchanged": unchanged,
                "controls_negative_or_weak": checks["controls_negative_or_weak"] if track == "all" else "",
                "trivial_tasks_strong": checks["trivial_tasks_strong"] if track == "all" else "",
            }
        )
    return rows


def _calibration_transition_rows(d5_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    transition_counts: Counter[tuple[str, str]] = Counter()
    for row in d5_rows:
        transition_counts[(row["baseline_label"], row["calibrated_label"])] += 1
    rows = []
    for baseline_label in LABEL_ORDER:
        for calibrated_label in LABEL_ORDER:
            rows.append(
                {
                    "baseline_label": baseline_label,
                    "calibrated_label": calibrated_label,
                    "task_count": transition_counts[(baseline_label, calibrated_label)],
                }
            )
    return rows


def _bonus_rows(results_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    b1 = _read_json(results_root / "EXP-B1" / "rule_extraction.json")
    b2 = _read_json(results_root / "EXP-B2" / "program_search.json")

    b1_rows = []
    for task_id, payload in sorted(b1["task_results"].items()):
        b1_rows.append(
            {
                "task_id": task_id,
                "best_depth": payload["best_depth"],
                "best_accuracy": payload["best_accuracy"],
                "passes_99_threshold": payload["passes_99_threshold"],
                "uses_only_relevant": payload["structural_info"]["uses_only_relevant"],
                "n_nodes": payload["structural_info"]["n_nodes"],
                "depth": payload["structural_info"]["depth"],
            }
        )

    b2_rows = []
    for task_id, payload in sorted(b2["task_results"].items()):
        b2_rows.append(
            {
                "task_id": task_id,
                "best_program": payload["best_program"],
                "best_oracle_score": payload["best_oracle_score"],
                "best_hard_test_accuracy": payload["best_hard_test_accuracy"],
                "passes_99_threshold": payload["passes_99_threshold"],
            }
        )

    return {"b1": b1_rows, "b2": b2_rows}


def _bonus_summary_rows(bonus: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows = []
    for experiment_id, key in [("EXP-B1", "b1"), ("EXP-B2", "b2")]:
        task_rows = bonus[key]
        pass_count = sum(1 for row in task_rows if row["passes_99_threshold"])
        rows.append(
            {
                "experiment_id": experiment_id,
                "tasks_evaluated": len(task_rows),
                "tasks_passing": pass_count,
                "pass_rate": _round_or_none(pass_count / len(task_rows) if task_rows else None),
            }
        )
    return rows


def _runtime_rows(results_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for experiment_id in RESULT_EXPERIMENTS:
        runtime = _parse_runtime_seconds(results_root / experiment_id / "summary.md")
        rows.append(
            {
                "experiment_id": experiment_id,
                "runtime_seconds": runtime,
                "runtime_available": runtime is not None,
            }
        )
    return rows


def _artifact_manifest(
    results_root: Path,
    output_root: Path,
    data_files: List[Path],
    figure_files: List[Path],
    runtime_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    required_results = {
        "baseline_experiments": BASELINE_EXPERIMENTS,
        "diagnostic_experiments": ["EXP-D1", "EXP-D2", "EXP-D3", "EXP-D4", "EXP-D5"],
        "bonus_experiments": ["EXP-B1", "EXP-B2"],
    }
    result_status: Dict[str, Dict[str, bool]] = {}
    for category, experiment_ids in required_results.items():
        result_status[category] = {
            experiment_id: (results_root / experiment_id).exists()
            for experiment_id in experiment_ids
        }

    runtime_missing = [row["experiment_id"] for row in runtime_rows if row["runtime_seconds"] is None]
    return {
        "results_root": str(results_root),
        "output_root": str(output_root),
        "results_status": result_status,
        "all_required_results_present": all(
            present
            for category in result_status.values()
            for present in category.values()
        ),
        "derived_data_files": {path.name: path.exists() for path in data_files},
        "derived_figure_files": {path.name: path.exists() for path in figure_files},
        "runtime_coverage": {
            "available": sum(1 for row in runtime_rows if row["runtime_seconds"] is not None),
            "total": len(runtime_rows),
            "missing_experiments": runtime_missing,
        },
    }


def _write_checklist_markdown(path: Path, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Publication Asset Checklist",
        "",
        "## Result prerequisites",
    ]
    for category, status in manifest["results_status"].items():
        lines.append(f"- {category}:")
        for experiment_id, present in status.items():
            mark = "x" if present else " "
            lines.append(f"  - [{mark}] {experiment_id}")

    lines.extend(
        [
            "",
            "## Derived data files",
        ]
    )
    for filename, present in manifest["derived_data_files"].items():
        mark = "x" if present else " "
        lines.append(f"- [{mark}] {filename}")

    lines.extend(
        [
            "",
            "## Derived figure files",
        ]
    )
    for filename, present in manifest["derived_figure_files"].items():
        mark = "x" if present else " "
        lines.append(f"- [{mark}] {filename}")

    coverage = manifest["runtime_coverage"]
    lines.extend(
        [
            "",
            "## Runtime coverage",
            f"- Parsed runtimes available for {coverage['available']} of {coverage['total']} experiments.",
        ]
    )
    if coverage["missing_experiments"]:
        missing = ", ".join(coverage["missing_experiments"])
        lines.append(f"- Runtime not recoverable from summary artifacts for: {missing}.")

    _write_markdown(path, "\n".join(lines))


def _plot_baseline_label_distribution(rows: List[Dict[str, Any]], output_path: Path) -> None:
    counts_by_track = {
        track: Counter(row["label"] for row in rows if row["track"] == track)
        for track in TRACK_ORDER
    }
    x = np.arange(len(LABEL_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for offset, track in [(-width / 2, "sequence"), (width / 2, "classification")]:
        ax.bar(
            x + offset,
            [counts_by_track[track].get(label, 0) for label in LABEL_ORDER],
            width,
            label=TRACK_DISPLAY[track],
            color=TRACK_COLORS[track],
        )
    ax.set_xticks(x, LABEL_ORDER)
    ax.set_ylabel("Task count")
    ax.set_title("Baseline Solvability Labels by Track")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_accuracy_boxplots(rows: List[Dict[str, Any]], output_path: Path) -> None:
    groups = [
        [row["best_iid_accuracy"] for row in rows if row["track"] == "sequence" and row["best_iid_accuracy"] is not None],
        [row["best_ood_accuracy"] for row in rows if row["track"] == "sequence" and row["best_ood_accuracy"] is not None],
        [row["best_iid_accuracy"] for row in rows if row["track"] == "classification" and row["best_iid_accuracy"] is not None],
        [row["best_ood_accuracy"] for row in rows if row["track"] == "classification" and row["best_ood_accuracy"] is not None],
    ]
    labels = ["Seq IID", "Seq OOD", "Cls IID", "Cls OOD"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bp = ax.boxplot(groups, patch_artist=True, tick_labels=labels)
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#ECA82C"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Best IID and OOD Accuracy by Track")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_iid_vs_ood_scatter(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for track in TRACK_ORDER:
        track_rows = [
            row
            for row in rows
            if row["track"] == track
            and row["best_iid_accuracy"] is not None
            and row["best_ood_accuracy"] is not None
        ]
        ax.scatter(
            [row["best_iid_accuracy"] for row in track_rows],
            [row["best_ood_accuracy"] for row in track_rows],
            s=70,
            alpha=0.85,
            color=TRACK_COLORS[track],
            label=TRACK_DISPLAY[track],
        )

    annotation_offsets = {
        "C1.6_modular_class": (6, 5),
        "C3.1_xor": (6, -12),
        "S1.4_count_symbol": (6, 5),
        "S2.2_balanced_parens": (-48, 6),
    }
    for row in rows:
        if row["task_id"] not in annotation_offsets:
            continue
        if row["best_iid_accuracy"] is None or row["best_ood_accuracy"] is None:
            continue
        ax.annotate(
            row["task_id"],
            (row["best_iid_accuracy"], row["best_ood_accuracy"]),
            textcoords="offset points",
            xytext=annotation_offsets[row["task_id"]],
            fontsize=8,
        )

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#777777", linewidth=1.2)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Best IID accuracy")
    ax.set_ylabel("Best OOD accuracy")
    ax.set_title("Task-Level IID vs OOD Accuracy")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_calibrated_label_shift(d5_rows: List[Dict[str, Any]], output_path: Path) -> None:
    baseline = Counter(row["baseline_label"] for row in d5_rows)
    calibrated = Counter(row["calibrated_label"] for row in d5_rows)
    x = np.arange(len(LABEL_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        x - width / 2,
        [baseline.get(label, 0) for label in LABEL_ORDER],
        width,
        label="Baseline",
        color="#9D755D",
    )
    ax.bar(
        x + width / 2,
        [calibrated.get(label, 0) for label in LABEL_ORDER],
        width,
        label="Calibrated",
        color="#59A14F",
    )
    ax.set_xticks(x, LABEL_ORDER)
    ax.set_ylabel("Task count")
    ax.set_title("Label Distribution Before and After Diagnostics")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_top_calibrated_scores(d5_rows: List[Dict[str, Any]], output_path: Path, top_n: int = 10) -> None:
    top_rows = sorted(d5_rows, key=lambda row: row["calibrated_score"], reverse=True)[:top_n]
    labels = [row["task_id"] for row in top_rows]
    scores = [row["calibrated_score"] for row in top_rows]
    colors = [TRACK_COLORS[row["track"]] for row in top_rows]

    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1], alpha=0.92)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Calibrated solvability score")
    ax.set_title("Top Tasks by Calibrated Score")
    ax.legend(
        handles=[
            Patch(facecolor=TRACK_COLORS["sequence"], label="Sequence"),
            Patch(facecolor=TRACK_COLORS["classification"], label="Classification"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_bonus_success_rates(
    bonus_summary_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    labels = ["Rule Extraction (B1)", "Program Search (B2)"]
    rates = [row["pass_rate"] for row in bonus_summary_rows]
    colors = ["#B279A2", "#E45756"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.bar(labels, rates, color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success rate")
    ax.set_title("Bonus Recovery Success Rates")
    for idx, rate in enumerate(rates):
        ax.text(idx, rate + 0.03, f"{rate:.1%}", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_diagnostic_heatmap(
    d1_rows: List[Dict[str, Any]],
    d2_rows: List[Dict[str, Any]],
    d3_rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    task_ids = sorted(
        {
            *[row["task_id"] for row in d1_rows],
            *[row["task_id"] for row in d2_rows],
            *[row["task_id"] for row in d3_rows],
        }
    )
    metric_names = ["D1 sample", "D2 distractor", "D3 noise"]
    matrix = np.full((len(task_ids), len(metric_names)), np.nan, dtype=float)

    d1_map = {row["task_id"]: 1.0 if row["criterion_8_sample_efficiency"] else 0.0 for row in d1_rows}
    d2_map = {row["task_id"]: 1.0 if row["criterion_7_distractor_robustness"] else 0.0 for row in d2_rows}
    d3_map = {row["task_id"]: 1.0 if row["smooth_degradation"] else 0.0 for row in d3_rows}

    for row_idx, task_id in enumerate(task_ids):
        if task_id in d1_map:
            matrix[row_idx, 0] = d1_map[task_id]
        if task_id in d2_map:
            matrix[row_idx, 1] = d2_map[task_id]
        if task_id in d3_map:
            matrix[row_idx, 2] = d3_map[task_id]

    cmap = ListedColormap(["#B2182B", "#1A9850"])
    cmap.set_bad("#D9D9D9")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    masked = np.ma.masked_invalid(matrix)

    fig, ax = plt.subplots(figsize=(7.6, max(4.4, len(task_ids) * 0.33)))
    ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(metric_names)), metric_names)
    ax.set_yticks(range(len(task_ids)), task_ids)
    ax.set_title("Diagnostic Pass/Fail Matrix")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                text = "NA"
                color = "#333333"
            elif value >= 0.5:
                text = "P"
                color = "white"
            else:
                text = "F"
                color = "white"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color=color, fontsize=9)

    ax.legend(
        handles=[
            Patch(facecolor="#1A9850", label="Pass"),
            Patch(facecolor="#B2182B", label="Fail"),
            Patch(facecolor="#D9D9D9", label="Not evaluated"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results"
    output_root = _ensure_dir(repo_root / "output" / "publication_assets")
    data_dir = _ensure_dir(output_root / "data")
    figures_dir = _ensure_dir(output_root / "figures")

    baseline_rows = _baseline_rows(results_root)
    diagnostic = _diagnostic_rows(results_root)
    bonus = _bonus_rows(results_root)
    runtime_rows = _runtime_rows(results_root)

    task_inventory_rows = _task_inventory_rows(baseline_rows)
    baseline_track_summary_rows = _baseline_track_summary_rows(baseline_rows)
    diagnostic_overview_rows = _diagnostic_overview_rows(diagnostic)
    calibration_overview_rows = _calibration_overview_rows(diagnostic["d5"], diagnostic["d5_checks"])
    calibration_transition_rows = _calibration_transition_rows(diagnostic["d5"])
    bonus_summary_rows = _bonus_summary_rows(bonus)
    top_calibrated_rows = sorted(
        diagnostic["d5"],
        key=lambda row: row["calibrated_score"],
        reverse=True,
    )[:10]

    data_files = [
        data_dir / "baseline_task_results.csv",
        data_dir / "task_inventory.csv",
        data_dir / "baseline_track_summary.csv",
        data_dir / "diagnostic_d1_sample_efficiency.csv",
        data_dir / "diagnostic_d2_distractor_robustness.csv",
        data_dir / "diagnostic_d3_noise_robustness.csv",
        data_dir / "diagnostic_d4_feature_alignment.csv",
        data_dir / "diagnostic_d5_calibrated_labels.csv",
        data_dir / "diagnostic_overview.csv",
        data_dir / "calibration_overview.csv",
        data_dir / "calibration_transitions.csv",
        data_dir / "bonus_b1_rule_extraction.csv",
        data_dir / "bonus_b2_program_search.csv",
        data_dir / "bonus_summary.csv",
        data_dir / "top_calibrated_tasks.csv",
        data_dir / "experiment_runtimes.csv",
        data_dir / "publication_summary.json",
        data_dir / "publication_checklist.md",
    ]

    figure_files = [
        figures_dir / "baseline_verdict_distribution.png",
        figures_dir / "accuracy_by_track.png",
        figures_dir / "iid_vs_ood_accuracy.png",
        figures_dir / "calibrated_label_shift.png",
        figures_dir / "top_calibrated_scores.png",
        figures_dir / "bonus_success_rates.png",
        figures_dir / "diagnostic_matrix.png",
    ]

    _write_csv(data_dir / "baseline_task_results.csv", baseline_rows)
    _write_csv(data_dir / "task_inventory.csv", task_inventory_rows)
    _write_csv(data_dir / "baseline_track_summary.csv", baseline_track_summary_rows)
    _write_csv(data_dir / "diagnostic_d1_sample_efficiency.csv", diagnostic["d1"])
    _write_csv(data_dir / "diagnostic_d2_distractor_robustness.csv", diagnostic["d2"])
    _write_csv(data_dir / "diagnostic_d3_noise_robustness.csv", diagnostic["d3"])
    _write_csv(data_dir / "diagnostic_d4_feature_alignment.csv", diagnostic["d4"])
    _write_csv(data_dir / "diagnostic_d5_calibrated_labels.csv", diagnostic["d5"])
    _write_csv(data_dir / "diagnostic_overview.csv", diagnostic_overview_rows)
    _write_csv(data_dir / "calibration_overview.csv", calibration_overview_rows)
    _write_csv(data_dir / "calibration_transitions.csv", calibration_transition_rows)
    _write_csv(data_dir / "bonus_b1_rule_extraction.csv", bonus["b1"])
    _write_csv(data_dir / "bonus_b2_program_search.csv", bonus["b2"])
    _write_csv(data_dir / "bonus_summary.csv", bonus_summary_rows)
    _write_csv(data_dir / "top_calibrated_tasks.csv", top_calibrated_rows)
    _write_csv(data_dir / "experiment_runtimes.csv", runtime_rows)

    _plot_baseline_label_distribution(baseline_rows, figures_dir / "baseline_verdict_distribution.png")
    _plot_accuracy_boxplots(baseline_rows, figures_dir / "accuracy_by_track.png")
    _plot_iid_vs_ood_scatter(baseline_rows, figures_dir / "iid_vs_ood_accuracy.png")
    _plot_calibrated_label_shift(diagnostic["d5"], figures_dir / "calibrated_label_shift.png")
    _plot_top_calibrated_scores(diagnostic["d5"], figures_dir / "top_calibrated_scores.png")
    _plot_bonus_success_rates(bonus_summary_rows, figures_dir / "bonus_success_rates.png")
    _plot_diagnostic_heatmap(
        diagnostic["d1"],
        diagnostic["d2"],
        diagnostic["d3"],
        figures_dir / "diagnostic_matrix.png",
    )

    manifest = _artifact_manifest(results_root, output_root, data_files, figure_files, runtime_rows)
    _write_checklist_markdown(data_dir / "publication_checklist.md", manifest)
    manifest = _artifact_manifest(results_root, output_root, data_files, figure_files, runtime_rows)
    _write_checklist_markdown(data_dir / "publication_checklist.md", manifest)

    summary_payload = {
        "artifact_manifest": manifest,
        "task_inventory_rows": task_inventory_rows,
        "baseline_track_summary_rows": baseline_track_summary_rows,
        "diagnostic_overview_rows": diagnostic_overview_rows,
        "calibration_overview_rows": calibration_overview_rows,
        "bonus_summary_rows": bonus_summary_rows,
        "top_calibrated_tasks": top_calibrated_rows,
        "runtime_rows": runtime_rows,
    }
    _write_json(data_dir / "publication_summary.json", summary_payload)


if __name__ == "__main__":
    main()
