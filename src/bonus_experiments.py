"""TASK-15 bonus experiment runners: algorithm discovery.

EXP-B1: Rule Extraction from Classification Models
EXP-B2: DSL Program Search for Sequence Tasks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

from src.data_generator import Sample, generate_dataset
from src.dsl.sequence_dsl import SeqProgram, sample_programs_batch
from src.models.harness import InputEncoder
from src.registry import TaskRegistry, TaskSpec, build_default_registry

# ---------------------------------------------------------------------------
# Shared artifact dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BonusExperimentArtifact:
    experiment_id: str
    output_dir: Path
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EXP-B1: Classification tasks to attempt rule extraction on
# Select tasks with MODERATE or STRONG verdicts from TASK-13/14
TASK15_B1_TASK_IDS: List[str] = [
    "C1.1_numeric_threshold",
    "C1.2_range_binning",
    "C1.3_categorical_match",
    "C1.5_numeric_comparison",
    "C2.1_and_rule",
    "C2.2_or_rule",
    "C2.3_nested_if_else",
    "C2.5_k_of_n",
    "C2.6_categorical_gate",
    "C3.1_xor",
    "C3.3_rank_based",
    "C3.5_interaction_poly",
]

TASK15_B1_N_TRAIN = 2000
TASK15_B1_N_HARD_TEST = 5000
TASK15_B1_SEEDS = [42, 123, 456]
TASK15_B1_MAX_DEPTH_SWEEP = [2, 3, 5, 8, None]  # None = unlimited

# EXP-B2: Sequence tasks to attempt program search on
TASK15_B2_TASK_IDS: List[str] = [
    "S1.1_reverse",
    "S1.2_sort",
    "S1.4_count_symbol",
    "S1.5_parity",
    "S1.6_prefix_sum",
    "S1.7_deduplicate",
    "S1.8_extrema",
    "S3.1_dedup_sort_count",
    "S3.2_filter_sort_sum",
]

TASK15_B2_N_ORACLE_SAMPLES = 500
TASK15_B2_N_HARD_TEST = 1000
TASK15_B2_SEARCH_BUDGET = 5000  # candidate programs to evaluate
TASK15_B2_MAX_DEPTH = 3
TASK15_B2_SEEDS = [42, 123, 456]

# Known relevant features per classification task (for structural comparison)
TASK15_RELEVANT_FEATURES: Dict[str, Tuple[str, ...]] = {
    "C1.1_numeric_threshold": ("x1",),
    "C1.2_range_binning": ("x1",),
    "C1.3_categorical_match": ("cat1",),
    "C1.5_numeric_comparison": ("x1", "x2"),
    "C2.1_and_rule": ("x1", "cat1"),
    "C2.2_or_rule": ("x1", "cat1"),
    "C2.3_nested_if_else": ("x1", "cat1", "x2"),
    "C2.5_k_of_n": ("x1", "x2", "cat1"),
    "C2.6_categorical_gate": ("x1", "cat1"),
    "C3.1_xor": ("x1", "x2"),
    "C3.3_rank_based": ("x1", "x2", "x3"),
    "C3.5_interaction_poly": ("x1", "x2"),
}


# ===================================================================
# EXP-B1: Rule Extraction from Classification Models
# ===================================================================


def _train_decision_tree(
    task: TaskSpec,
    train_samples: Sequence[Sample],
    max_depth: Optional[int],
    seed: int,
) -> Tuple[DecisionTreeClassifier, InputEncoder, list, list]:
    """Train a sklearn DecisionTreeClassifier on task data.

    Returns (fitted_tree, encoder, feature_names, class_names).
    """
    encoder = InputEncoder()
    X_train = encoder.fit_transform([s.input_data for s in train_samples])
    y_train = [str(s.output_data) for s in train_samples]

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    feature_names = encoder.feature_names
    class_names = list(clf.classes_)
    return clf, encoder, feature_names, class_names


def _evaluate_tree_on_hard_test(
    clf: DecisionTreeClassifier,
    encoder: InputEncoder,
    task: TaskSpec,
    hard_test_samples: Sequence[Sample],
) -> float:
    """Evaluate the extracted tree against reference labels on hard test samples."""
    X_test = encoder.transform([s.input_data for s in hard_test_samples])
    y_true = [str(s.output_data) for s in hard_test_samples]
    y_pred = clf.predict(X_test).tolist()
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / max(len(y_true), 1)


def _tree_structural_info(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    relevant_features: Tuple[str, ...],
) -> Dict[str, Any]:
    """Extract structural info from a fitted decision tree."""
    tree = clf.tree_
    n_nodes = tree.node_count
    depth = clf.get_depth()
    n_leaves = clf.get_n_leaves()

    # Which features are actually used in splits?
    used_feature_indices = set()
    for i in range(n_nodes):
        if tree.feature[i] >= 0:  # -2 means leaf
            used_feature_indices.add(tree.feature[i])

    used_features = sorted(
        feature_names[idx] for idx in used_feature_indices if idx < len(feature_names)
    )

    # Check if only relevant features are used
    relevant_set = set(relevant_features)
    used_set = set(used_features)
    # A label-encoded feature keeps the base name (e.g. "cat1"); match it
    used_base_features = set()
    for f in used_set:
        base = f.split("=")[0] if "=" in f else f
        used_base_features.add(base)

    uses_only_relevant = used_base_features.issubset(relevant_set)
    extra_features = used_base_features - relevant_set
    missing_features = relevant_set - used_base_features

    return {
        "n_nodes": n_nodes,
        "depth": depth,
        "n_leaves": n_leaves,
        "used_features": used_features,
        "used_base_features": sorted(used_base_features),
        "relevant_features": sorted(relevant_set),
        "uses_only_relevant": uses_only_relevant,
        "extra_features_used": sorted(extra_features),
        "missing_relevant_features": sorted(missing_features),
    }


def _extract_tree_text(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
) -> str:
    """Export the tree as a human-readable text rule."""
    try:
        return export_text(clf, feature_names=feature_names, max_depth=10)
    except Exception:
        return "(could not export tree text)"


def run_rule_extraction_experiment(
    output_root: Path | str = "results",
    task_ids: Optional[List[str]] = None,
    registry: Optional[TaskRegistry] = None,
    n_train: int = TASK15_B1_N_TRAIN,
    n_hard_test: int = TASK15_B1_N_HARD_TEST,
    max_depth_sweep: Optional[List[Optional[int]]] = None,
    seeds: Optional[List[int]] = None,
) -> BonusExperimentArtifact:
    """EXP-B1: Extract decision tree rules and compare to reference algorithms.

    For each classification task, trains decision trees at various depth limits,
    evaluates functional equivalence on a hard test set, and reports which tasks
    have recoverable rules.
    """
    output_root = Path(output_root)
    output_dir = output_root / "EXP-B1"
    output_dir.mkdir(parents=True, exist_ok=True)

    if registry is None:
        registry = build_default_registry()
    if task_ids is None:
        task_ids = TASK15_B1_TASK_IDS
    if max_depth_sweep is None:
        max_depth_sweep = TASK15_B1_MAX_DEPTH_SWEEP
    if seeds is None:
        seeds = TASK15_B1_SEEDS

    task_results: Dict[str, Any] = {}
    pass_count = 0
    total_count = 0

    for task_id in task_ids:
        task = registry.get(task_id)
        relevant = TASK15_RELEVANT_FEATURES.get(task_id, ())

        best_accuracy = 0.0
        best_depth: Optional[int] = None
        best_tree_text = ""
        best_structural_info: Dict[str, Any] = {}
        depth_results: List[Dict[str, Any]] = []

        for max_depth in max_depth_sweep:
            seed_accuracies: List[float] = []

            for seed in seeds:
                # Generate training data
                dataset = generate_dataset(task, n_samples=n_train + n_hard_test, base_seed=seed)
                train_samples = dataset.samples[:n_train]
                hard_test_samples = dataset.samples[n_train:]

                clf, encoder, feature_names, class_names = _train_decision_tree(
                    task, train_samples, max_depth, seed,
                )
                acc = _evaluate_tree_on_hard_test(clf, encoder, task, hard_test_samples)
                seed_accuracies.append(acc)

            mean_acc = float(np.mean(seed_accuracies))
            depth_results.append({
                "max_depth": max_depth,
                "mean_accuracy": round(mean_acc, 4),
                "seed_accuracies": [round(a, 4) for a in seed_accuracies],
            })

            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_depth = max_depth
                # Re-train one final model to get the tree text and structural info
                final_dataset = generate_dataset(task, n_samples=n_train + n_hard_test, base_seed=seeds[0])
                clf_final, enc_final, fn_final, cn_final = _train_decision_tree(
                    task, final_dataset.samples[:n_train], max_depth, seeds[0],
                )
                best_tree_text = _extract_tree_text(clf_final, fn_final)
                best_structural_info = _tree_structural_info(clf_final, fn_final, relevant)

        passes_threshold = best_accuracy >= 0.99
        if passes_threshold:
            pass_count += 1
        total_count += 1

        task_results[task_id] = {
            "best_depth": best_depth,
            "best_accuracy": round(best_accuracy, 4),
            "passes_99_threshold": passes_threshold,
            "depth_sweep": depth_results,
            "extracted_rule": best_tree_text,
            "structural_info": best_structural_info,
            "relevant_features": list(relevant),
        }

        # Write per-task artifacts
        per_task_dir = output_dir / "per_task" / task_id
        per_task_dir.mkdir(parents=True, exist_ok=True)
        (per_task_dir / "extracted_tree.txt").write_text(best_tree_text, encoding="utf-8")
        (per_task_dir / "result.json").write_text(
            json.dumps(task_results[task_id], indent=2, default=str), encoding="utf-8"
        )

    # Plot: accuracy by depth for each task
    _plot_b1_depth_sweep(task_results, output_dir)

    payload = {
        "experiment_id": "EXP-B1",
        "task_results": task_results,
        "summary": {
            "total_tasks": total_count,
            "tasks_passing_99": pass_count,
            "pass_rate": round(pass_count / max(total_count, 1), 4),
            "acceptance_met": pass_count >= 1,
        },
    }

    (output_dir / "rule_extraction.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps({
            "task_ids": task_ids,
            "n_train": n_train,
            "n_hard_test": n_hard_test,
            "max_depth_sweep": [d if d is not None else "unlimited" for d in max_depth_sweep],
            "seeds": seeds,
        }, indent=2), encoding="utf-8"
    )

    # Write markdown summary
    _write_b1_summary(payload, output_dir)

    return BonusExperimentArtifact(
        experiment_id="EXP-B1",
        output_dir=output_dir,
        payload=payload,
    )


def _plot_b1_depth_sweep(
    task_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot accuracy vs tree depth for each task."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for task_id, result in task_results.items():
        depths = []
        accs = []
        for dr in result["depth_sweep"]:
            d = dr["max_depth"] if dr["max_depth"] is not None else 20
            depths.append(d)
            accs.append(dr["mean_accuracy"])
        ax.plot(depths, accs, marker="o", label=task_id, linewidth=1.5)

    ax.set_xlabel("Max Tree Depth")
    ax.set_ylabel("Mean Accuracy on Hard Test")
    ax.set_title("EXP-B1: Rule Extraction Accuracy by Tree Depth")
    ax.axhline(y=0.99, color="red", linestyle="--", linewidth=0.8, label="99% threshold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "depth_sweep.png", dpi=150)
    plt.close(fig)


def _write_b1_summary(payload: Dict[str, Any], output_dir: Path) -> None:
    lines = ["# EXP-B1: Rule Extraction from Classification Models\n"]
    summary = payload["summary"]
    lines.append(f"**Tasks evaluated:** {summary['total_tasks']}")
    lines.append(f"**Tasks passing >99% on hard test:** {summary['tasks_passing_99']}")
    lines.append(f"**Pass rate:** {summary['pass_rate']:.1%}")
    lines.append(f"**Acceptance met (≥1 pass):** {summary['acceptance_met']}\n")

    lines.append("## Per-Task Results\n")
    lines.append("| Task | Best Depth | Best Accuracy | >99%? | Uses Only Relevant Features |")
    lines.append("|---|---|---|---|---|")

    for task_id, result in payload["task_results"].items():
        depth_str = str(result["best_depth"]) if result["best_depth"] is not None else "unlimited"
        acc_str = f"{result['best_accuracy']:.2%}"
        pass_str = "✅" if result["passes_99_threshold"] else "❌"
        struct = result.get("structural_info", {})
        relevant_str = "✅" if struct.get("uses_only_relevant", False) else "❌"
        lines.append(f"| {task_id} | {depth_str} | {acc_str} | {pass_str} | {relevant_str} |")

    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ===================================================================
# EXP-B2: DSL Program Search for Sequence Tasks
# ===================================================================


def _generate_candidate_programs(
    budget: int,
    max_depth: int,
    seed: int,
) -> List[SeqProgram]:
    """Generate a pool of candidate DSL programs via random sampling."""
    return sample_programs_batch(n=budget, seed=seed, max_depth=max_depth)


def _evaluate_program_against_oracle(
    program: SeqProgram,
    oracle_inputs: List[List[int]],
    oracle_outputs: List[Any],
) -> float:
    """Score a candidate program by agreement with the oracle (reference algorithm)."""
    if not oracle_inputs:
        return 0.0
    correct = 0
    for inp, expected_out in zip(oracle_inputs, oracle_outputs):
        try:
            candidate_out = program.evaluate(list(inp))
            if candidate_out == expected_out:
                correct += 1
        except Exception:
            pass
    return correct / len(oracle_inputs)


def _validate_on_hard_test(
    program: SeqProgram,
    task: TaskSpec,
    n_hard_test: int,
    seed: int,
) -> float:
    """Validate a candidate program against the reference algorithm on additional test inputs.

    Samples new inputs via the task's input sampler using different seeds.
    """
    correct = 0
    total = 0

    for i in range(n_hard_test):
        inp = task.input_sampler(seed + 100000 + i)
        try:
            expected = task.reference_algorithm(inp)
            candidate = program.evaluate(list(inp))
            if candidate == expected:
                correct += 1
        except Exception:
            pass
        total += 1

    return correct / max(total, 1)


def run_program_search_experiment(
    output_root: Path | str = "results",
    task_ids: Optional[List[str]] = None,
    registry: Optional[TaskRegistry] = None,
    n_oracle_samples: int = TASK15_B2_N_ORACLE_SAMPLES,
    n_hard_test: int = TASK15_B2_N_HARD_TEST,
    search_budget: int = TASK15_B2_SEARCH_BUDGET,
    max_depth: int = TASK15_B2_MAX_DEPTH,
    seeds: Optional[List[int]] = None,
) -> BonusExperimentArtifact:
    """EXP-B2: Search over DSL programs for sequence tasks.

    For each sequence task, generates oracle I/O pairs from the reference algorithm,
    searches over randomly sampled DSL programs for the best match, and validates
    the best candidate on a hard test set.
    """
    output_root = Path(output_root)
    output_dir = output_root / "EXP-B2"
    output_dir.mkdir(parents=True, exist_ok=True)

    if registry is None:
        registry = build_default_registry()
    if task_ids is None:
        task_ids = TASK15_B2_TASK_IDS
    if seeds is None:
        seeds = TASK15_B2_SEEDS

    task_results: Dict[str, Any] = {}
    pass_count = 0
    total_count = 0

    for task_id in task_ids:
        task = registry.get(task_id)

        best_program_name: Optional[str] = None
        best_oracle_score = 0.0
        best_hard_test_score = 0.0
        best_program_id: Optional[str] = None
        seed_results: List[Dict[str, Any]] = []

        for seed in seeds:
            # Generate oracle I/O pairs from the reference algorithm
            oracle_inputs: List[List[int]] = []
            oracle_outputs: List[Any] = []
            for i in range(n_oracle_samples):
                inp = task.input_sampler(seed + i)
                out = task.reference_algorithm(inp)
                oracle_inputs.append(list(inp) if isinstance(inp, (list, tuple)) else inp)
                oracle_outputs.append(out)

            # Generate candidate programs
            candidates = _generate_candidate_programs(search_budget, max_depth, seed)

            # Score each candidate against the oracle, tracking best incrementally
            top_score = 0.0
            top_prog: Optional[SeqProgram] = None
            n_perfect = 0
            for prog in candidates:
                score = _evaluate_program_against_oracle(prog, oracle_inputs, oracle_outputs)
                if score >= 1.0:
                    n_perfect += 1
                if score > top_score:
                    top_score = score
                    top_prog = prog
                    if score >= 1.0:
                        break  # perfect match, no need to search further

            # Validate top candidate on hard test
            hard_test_acc = 0.0
            if top_prog is not None and top_score > 0.0:
                hard_test_acc = _validate_on_hard_test(
                    top_prog, task, n_hard_test, seed + 999999,
                )

            seed_results.append({
                "seed": seed,
                "top_oracle_score": round(top_score, 4),
                "top_program": top_prog.name() if top_prog else None,
                "top_program_id": top_prog.program_id if top_prog else None,
                "hard_test_accuracy": round(hard_test_acc, 4),
                "n_candidates_evaluated": len(candidates),
                "n_perfect_oracle_matches": n_perfect,
            })

            if hard_test_acc > best_hard_test_score:
                best_hard_test_score = hard_test_acc
                best_oracle_score = top_score
                best_program_name = top_prog.name() if top_prog else None
                best_program_id = top_prog.program_id if top_prog else None

        passes_threshold = best_hard_test_score >= 0.99
        if passes_threshold:
            pass_count += 1
        total_count += 1

        task_results[task_id] = {
            "best_program": best_program_name,
            "best_program_id": best_program_id,
            "best_oracle_score": round(best_oracle_score, 4),
            "best_hard_test_accuracy": round(best_hard_test_score, 4),
            "passes_99_threshold": passes_threshold,
            "seed_results": seed_results,
        }

        # Per-task artifacts
        per_task_dir = output_dir / "per_task" / task_id
        per_task_dir.mkdir(parents=True, exist_ok=True)
        (per_task_dir / "result.json").write_text(
            json.dumps(task_results[task_id], indent=2, default=str), encoding="utf-8"
        )

    # Summary plot
    _plot_b2_results(task_results, output_dir)

    payload = {
        "experiment_id": "EXP-B2",
        "task_results": task_results,
        "summary": {
            "total_tasks": total_count,
            "tasks_passing_99": pass_count,
            "pass_rate": round(pass_count / max(total_count, 1), 4),
            "acceptance_met": pass_count >= 1,
        },
    }

    (output_dir / "program_search.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps({
            "task_ids": task_ids,
            "n_oracle_samples": n_oracle_samples,
            "n_hard_test": n_hard_test,
            "search_budget": search_budget,
            "max_depth": max_depth,
            "seeds": seeds,
        }, indent=2), encoding="utf-8"
    )

    _write_b2_summary(payload, output_dir)

    return BonusExperimentArtifact(
        experiment_id="EXP-B2",
        output_dir=output_dir,
        payload=payload,
    )


def _plot_b2_results(
    task_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Plot best hard-test accuracy per task."""
    tasks = list(task_results.keys())
    accs = [task_results[t]["best_hard_test_accuracy"] for t in tasks]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(tasks))
    bars = ax.bar(x, accs, color=["green" if a >= 0.99 else "steelblue" for a in accs])
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Best Hard-Test Accuracy")
    ax.set_title("EXP-B2: Program Search — Best Recovered Program Accuracy")
    ax.axhline(y=0.99, color="red", linestyle="--", linewidth=0.8, label="99% threshold")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "program_search_results.png", dpi=150)
    plt.close(fig)


def _write_b2_summary(payload: Dict[str, Any], output_dir: Path) -> None:
    lines = ["# EXP-B2: DSL Program Search for Sequence Tasks\n"]
    summary = payload["summary"]
    lines.append(f"**Tasks evaluated:** {summary['total_tasks']}")
    lines.append(f"**Tasks with recovered program (>99%):** {summary['tasks_passing_99']}")
    lines.append(f"**Pass rate:** {summary['pass_rate']:.1%}")
    lines.append(f"**Acceptance met (≥1 pass):** {summary['acceptance_met']}\n")

    lines.append("## Per-Task Results\n")
    lines.append("| Task | Best Program | Oracle Score | Hard-Test Acc | >99%? |")
    lines.append("|---|---|---|---|---|")

    for task_id, result in payload["task_results"].items():
        prog = result["best_program"] or "—"
        oracle = f"{result['best_oracle_score']:.2%}"
        hard = f"{result['best_hard_test_accuracy']:.2%}"
        pass_str = "✅" if result["passes_99_threshold"] else "❌"
        lines.append(f"| {task_id} | `{prog}` | {oracle} | {hard} | {pass_str} |")

    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ===================================================================
# Top-level runner
# ===================================================================


def run_all_bonus_experiments(
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, BonusExperimentArtifact]:
    """Run all TASK-15 bonus experiments (EXP-B1 and EXP-B2)."""
    output_root = Path(output_root)

    if registry is None:
        registry = build_default_registry()

    b1 = run_rule_extraction_experiment(
        output_root=output_root,
        registry=registry,
    )
    b2 = run_program_search_experiment(
        output_root=output_root,
        registry=registry,
    )

    return {"EXP-B1": b1, "EXP-B2": b2}
