"""TASK-14 diagnostic experiment runners and calibration helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

from src.data_generator import Sample, generate_dataset
from src.evaluation import evaluate
from src.models.harness import (
    InputEncoder,
    LabelEncoder,
    ModelConfig,
    ModelFamily,
    ModelHarness,
    SklearnModelWrapper,
    build_model,
)
from src.registry import TaskRegistry, TaskSpec, build_default_registry
from src.schemas import (
    CategoricalFeatureSpec,
    Distribution,
    FeatureSpec,
    NumericalFeatureSpec,
    TabularInputSchema,
)
from src.splits import split_iid, split_noise


TASK14_TRAIN_FRACTION = 0.7
TASK14_SEEDS = [42, 123, 456, 789, 1024]

TASK14_D1_SAMPLE_SIZES = [100, 250, 500, 1000, 2000]
TASK14_D1_TEST_SIZE = 1000
TASK14_D1_TASK_IDS = [
    "S1.4_count_symbol",
    "S2.2_balanced_parens",
    "S3.1_dedup_sort_count",
    "C1.1_numeric_threshold",
    "C2.3_nested_if_else",
    "C3.3_rank_based",
]
TASK14_D1_CONTROL_TASK_IDS = ["S0.1_random_labels", "C0.1_random_class"]

TASK14_D2_TASK_IDS = [
    "C1.1_numeric_threshold",
    "C2.1_and_rule",
    "C2.6_categorical_gate",
    "C3.1_xor",
    "C3.5_interaction_poly",
]
TASK14_D2_DISTRACTOR_COUNTS = [0, 2, 5, 10, 20]
TASK14_D2_N_SAMPLES = 900

TASK14_D3_TASK_IDS = [
    "C1.1_numeric_threshold",
    "C2.1_and_rule",
    "C3.1_xor",
    "C3.5_interaction_poly",
]
TASK14_D3_NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]
TASK14_D3_N_SAMPLES = 900

TASK14_D4_TASK_IDS = [
    "C2.1_and_rule",
    "C2.6_categorical_gate",
    "C3.1_xor",
]
TASK14_D4_DISTRACTOR_COUNT = 5
TASK14_D4_N_SAMPLES = 900

TASK14_BASELINE_EXPERIMENT_IDS = [
    "EXP-0.2",
    "EXP-0.3",
    "EXP-S1",
    "EXP-S2",
    "EXP-S3",
    "EXP-C1",
    "EXP-C2",
    "EXP-C3",
]

TASK14_RELEVANT_FEATURES: Dict[str, Tuple[str, ...]] = {
    "C1.1_numeric_threshold": ("x1",),
    "C2.1_and_rule": ("x1", "cat1"),
    "C2.3_nested_if_else": ("x1", "cat1", "x2"),
    "C2.6_categorical_gate": ("x1", "cat1"),
    "C3.1_xor": ("x1", "x2"),
    "C3.3_rank_based": ("x1", "x2", "x3"),
    "C3.5_interaction_poly": ("x1", "x2"),
}

TASK14_TRIVIAL_TASK_IDS = ["C1.1_numeric_threshold"]
TASK14_CONTROL_TASK_IDS = ["S0.1_random_labels", "C0.1_random_class"]


@dataclass(frozen=True)
class DiagnosticExperimentArtifact:
    """Payload and artifact directory for one TASK-14 diagnostic experiment."""

    experiment_id: str
    output_dir: Path
    payload: Dict[str, Any]


@dataclass(frozen=True)
class BaselineTaskRecord:
    """Baseline verdict and model selection metadata for a task."""

    task_id: str
    track: str
    tier: str
    source_experiment: str
    label: str
    score: float
    best_model_name: Optional[str]
    best_model_config: Optional[ModelConfig]
    best_iid_accuracy: Optional[float]
    best_ood_accuracy: Optional[float]
    evidence: Dict[str, bool]
    notes: List[str]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonicalize_hyperparams(hyperparams: Mapping[str, Any]) -> Dict[str, Any]:
    canonical: Dict[str, Any] = {}
    for key, value in hyperparams.items():
        if key == "hidden_layer_sizes" and isinstance(value, list):
            canonical[key] = tuple(value)
        else:
            canonical[key] = value
    return canonical


def _default_diagnostic_model_configs() -> List[ModelConfig]:
    return [
        ModelConfig(family=ModelFamily.DECISION_TREE),
        ModelConfig(
            family=ModelFamily.GRADIENT_BOOSTED_TREES,
            hyperparams={"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05},
        ),
        ModelConfig(
            family=ModelFamily.MLP,
            hyperparams={"hidden_layer_sizes": (128, 64), "max_iter": 400},
        ),
    ]


def _load_model_lookup(config_path: Path) -> Dict[str, ModelConfig]:
    config_payload = _load_json(config_path)
    lookup: Dict[str, ModelConfig] = {}
    for item in config_payload.get("model_configs", []):
        config = ModelConfig(
            family=item["family"],
            name=item.get("name"),
            hyperparams=_canonicalize_hyperparams(item.get("hyperparams", {})),
        )
        lookup[config.name or config.family.value] = config
    return lookup


def collect_baseline_task_records(
    results_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, BaselineTaskRecord]:
    """Load baseline verdicts and winning model configs from existing artifacts."""

    active_registry = build_default_registry() if registry is None else registry
    results_root_path = Path(results_root)
    records: Dict[str, BaselineTaskRecord] = {}

    for experiment_id in TASK14_BASELINE_EXPERIMENT_IDS:
        experiment_dir = results_root_path / experiment_id
        config_path = experiment_dir / "config.json"
        verdicts_path = experiment_dir / "solvability_verdicts.json"
        if not config_path.exists() or not verdicts_path.exists():
            continue

        model_lookup = _load_model_lookup(config_path)
        verdicts = _load_json(verdicts_path)

        for task_id, payload in verdicts.items():
            task = active_registry.get(task_id)
            best_model_name = payload.get("best_model")
            record = BaselineTaskRecord(
                task_id=task_id,
                track=task.track,
                tier=task.tier,
                source_experiment=experiment_id,
                label=payload["label"],
                score=float(payload["score"]),
                best_model_name=best_model_name,
                best_model_config=model_lookup.get(best_model_name) if best_model_name else None,
                best_iid_accuracy=payload.get("best_iid_accuracy"),
                best_ood_accuracy=payload.get("best_ood_accuracy"),
                evidence=dict(payload.get("evidence", {})),
                notes=list(payload.get("notes", [])),
            )
            existing = records.get(task_id)
            if existing is None or record.score > existing.score:
                records[task_id] = record

    return records


def _curve_auc(sample_sizes: Sequence[int], accuracies: Sequence[float]) -> float:
    if len(sample_sizes) != len(accuracies):
        raise ValueError("sample_sizes and accuracies must have the same length")
    if len(sample_sizes) < 2:
        return float(accuracies[0]) if accuracies else 0.0

    x = np.log10(np.asarray(sample_sizes, dtype=float))
    y = np.asarray(accuracies, dtype=float)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    area = _trapz(y, x=x)
    normalized = area / max(x[-1] - x[0], 1e-8)
    return round(float(normalized), 4)


def _run_accuracy_from_samples(
    task: TaskSpec,
    model_config: ModelConfig,
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
) -> float:
    harness = ModelHarness(model_config)
    prediction = harness.run(
        train_inputs=[sample.input_data for sample in train_samples],
        train_outputs=[sample.output_data for sample in train_samples],
        test_inputs=[sample.input_data for sample in test_samples],
        test_outputs=[sample.output_data for sample in test_samples],
    )
    report = evaluate(
        predictions=prediction.predictions,
        ground_truth=prediction.true_labels,
        task=task,
        split_name="diagnostic",
    )
    return float(report.accuracy)


def _build_iid_split_cache(
    task: TaskSpec,
    seeds: Sequence[int],
    n_samples: int,
    train_fraction: float,
) -> Dict[int, Any]:
    cache: Dict[int, Any] = {}
    for seed in seeds:
        dataset = generate_dataset(task, n_samples=n_samples, base_seed=seed)
        cache[seed] = split_iid(dataset, train_fraction=train_fraction, seed=seed)
    return cache


def _build_noise_split_cache(
    task: TaskSpec,
    seeds: Sequence[int],
    n_samples: int,
    train_fraction: float,
    noise_levels: Sequence[float],
) -> Dict[float, Dict[int, Any]]:
    dataset_by_seed = {
        seed: generate_dataset(task, n_samples=n_samples, base_seed=seed)
        for seed in seeds
    }
    schema = task.input_schema if isinstance(task.input_schema, TabularInputSchema) else None
    return {
        level: {
            seed: split_noise(
                dataset_by_seed[seed],
                train_fraction=train_fraction,
                test_noise_level=level,
                seed=seed,
                schema=schema,
            )
            for seed in seeds
        }
        for level in noise_levels
    }


def _build_d1_selection(
    records: Mapping[str, BaselineTaskRecord],
) -> List[BaselineTaskRecord]:
    selected: List[BaselineTaskRecord] = []
    for task_id in TASK14_D1_TASK_IDS + TASK14_D1_CONTROL_TASK_IDS:
        if task_id not in records:
            raise ValueError(
                f"Missing baseline artifacts for TASK-14 sample-efficiency task {task_id}"
            )
        record = records[task_id]
        if record.best_model_config is None:
            raise ValueError(f"No best-model config found for baseline task {task_id}")
        selected.append(record)
    return selected


def _track_control_auc(
    curves: Mapping[str, Dict[str, Any]],
    track: str,
) -> Optional[float]:
    control_task_id = "S0.1_random_labels" if track == "sequence" else "C0.1_random_class"
    payload = curves.get(control_task_id)
    if payload is None:
        return None
    return float(payload["auc"])


def _save_learning_curve_plot(
    output_path: Path,
    curves: Mapping[str, Dict[str, Any]],
    track: str,
    sample_sizes: Sequence[int],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for task_id, payload in sorted(curves.items()):
        if payload["track"] != track:
            continue
        y_values = [
            payload["accuracy_by_sample_size"][str(size)]["mean"]
            for size in sample_sizes
        ]
        ax.plot(sample_sizes, y_values, marker="o", label=task_id)

    ax.set_xscale("log")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Learning Curves ({track})")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _render_d1_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# EXP-D1 Sample Efficiency Comparison",
        "",
        "## Curves",
        "",
        "| Task | Track | Model | AUC | Delta vs Control | Criterion 8 |",
        "|---|---|---|---|---|---|",
    ]
    for task_id, task_payload in sorted(payload["task_curves"].items()):
        delta = task_payload.get("delta_vs_control_auc")
        delta_text = "N/A" if delta is None else f"{delta:.4f}"
        lines.append(
            f"| {task_id} | {task_payload['track']} | {task_payload['model_name']} | "
            f"{task_payload['auc']:.4f} | {delta_text} | "
            f"{'PASS' if task_payload['criterion_8_sample_efficiency'] else 'FAIL'} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def run_sample_efficiency_experiment(
    output_root: str | Path = "results",
    results_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
    sample_sizes: Optional[Sequence[int]] = None,
    test_size: int = TASK14_D1_TEST_SIZE,
    seeds: Optional[Sequence[int]] = None,
) -> DiagnosticExperimentArtifact:
    """Run EXP-D1 sample-efficiency curves from baseline-selected tasks/models."""

    active_registry = build_default_registry() if registry is None else registry
    active_sample_sizes = list(sample_sizes or TASK14_D1_SAMPLE_SIZES)
    active_seeds = list(seeds or TASK14_SEEDS)
    max_train_size = max(active_sample_sizes)
    total_samples = max_train_size + test_size
    train_fraction = max_train_size / total_samples

    records = collect_baseline_task_records(results_root=results_root, registry=active_registry)
    selected = _build_d1_selection(records)
    curves: Dict[str, Dict[str, Any]] = {}

    for record in selected:
        task = active_registry.get(record.task_id)
        split_cache = _build_iid_split_cache(
            task,
            active_seeds,
            total_samples,
            train_fraction,
        )
        per_size: Dict[str, Dict[str, Any]] = {}
        for train_size in active_sample_sizes:
            accuracies: List[float] = []
            for seed in active_seeds:
                split = split_cache[seed]
                if split.train_size < train_size:
                    raise ValueError(
                        f"Train pool for {task.task_id} is smaller than requested size {train_size}"
                    )
                accuracies.append(
                    _run_accuracy_from_samples(
                        task,
                        record.best_model_config,
                        split.train[:train_size],
                        split.test,
                    )
                )

            per_size[str(train_size)] = {
                "mean": round(float(np.mean(accuracies)), 4),
                "std": round(float(np.std(accuracies)), 4),
                "per_seed_accuracy": [round(float(value), 4) for value in accuracies],
            }

        mean_curve = [per_size[str(size)]["mean"] for size in active_sample_sizes]
        curves[task.task_id] = {
            "task_id": task.task_id,
            "track": task.track,
            "tier": task.tier,
            "model_name": record.best_model_name,
            "source_experiment": record.source_experiment,
            "accuracy_by_sample_size": per_size,
            "auc": _curve_auc(active_sample_sizes, mean_curve),
        }

    for task_id, task_payload in curves.items():
        task = active_registry.get(task_id)
        control_auc = _track_control_auc(curves, task.track)
        if task.complexity_metadata.get("is_control"):
            task_payload["delta_vs_control_auc"] = 0.0
            task_payload["sample_efficiency_score"] = 0.0
            task_payload["criterion_8_sample_efficiency"] = False
            continue

        delta = None if control_auc is None else round(task_payload["auc"] - control_auc, 4)
        score = 0.0 if delta is None else round(float(np.clip(delta / 0.5, 0.0, 1.0)), 4)
        task_payload["delta_vs_control_auc"] = delta
        task_payload["sample_efficiency_score"] = score
        task_payload["criterion_8_sample_efficiency"] = bool(delta is not None and delta >= 0.15)

    payload = {
        "experiment_id": "EXP-D1",
        "sample_sizes": active_sample_sizes,
        "seeds": active_seeds,
        "test_size": test_size,
        "task_curves": curves,
    }

    output_dir = _ensure_dir(Path(output_root) / "EXP-D1")
    _write_json(output_dir / "config.json", {
        "sample_sizes": active_sample_sizes,
        "seeds": active_seeds,
        "test_size": test_size,
        "task_ids": [record.task_id for record in selected],
        "control_task_ids": list(TASK14_D1_CONTROL_TASK_IDS),
    })
    _write_json(output_dir / "sample_efficiency.json", payload)
    (output_dir / "summary.md").write_text(_render_d1_summary(payload), encoding="utf-8")
    _save_learning_curve_plot(
        output_dir / "sequence_learning_curves.png",
        curves,
        "sequence",
        active_sample_sizes,
    )
    _save_learning_curve_plot(
        output_dir / "classification_learning_curves.png",
        curves,
        "classification",
        active_sample_sizes,
    )
    return DiagnosticExperimentArtifact(
        experiment_id="EXP-D1",
        output_dir=output_dir,
        payload=payload,
    )


def _clone_task_with_distractors(task: TaskSpec, distractor_count: int) -> TaskSpec:
    schema = task.input_schema
    if not isinstance(schema, TabularInputSchema):
        raise ValueError(f"Task {task.task_id} does not support distractor augmentation")

    def numerical_template(index: int) -> NumericalFeatureSpec:
        source = (
            schema.numerical_features[index % len(schema.numerical_features)]
            if schema.numerical_features
            else NumericalFeatureSpec(name="template_num", min_val=0.0, max_val=100.0)
        )
        return NumericalFeatureSpec(
            name=f"distractor_num_{index + 1}",
            min_val=source.min_val,
            max_val=source.max_val,
            distribution=source.distribution,
        )

    def categorical_template(index: int) -> CategoricalFeatureSpec:
        source = (
            schema.categorical_features[index % len(schema.categorical_features)]
            if schema.categorical_features
            else CategoricalFeatureSpec(
                name="template_cat",
                values=("A", "B", "C"),
                distribution=Distribution.UNIFORM,
            )
        )
        return CategoricalFeatureSpec(
            name=f"distractor_cat_{index + 1}",
            values=source.values,
            distribution=source.distribution,
            weights=source.weights,
        )

    extra_specs: List[FeatureSpec] = []
    for index in range(distractor_count):
        use_numeric = bool(schema.numerical_features) and (
            not schema.categorical_features or index % 2 == 0
        )
        extra_specs.append(
            numerical_template(index) if use_numeric else categorical_template(index)
        )

    augmented_schema = schema.with_extra_irrelevant(tuple(extra_specs))
    relevant_names = {spec.name for spec in schema.relevant_feature_specs}

    def augmented_reference(row: Mapping[str, Any]) -> Any:
        filtered = {name: row[name] for name in relevant_names}
        return task.reference_algorithm(filtered)

    return TaskSpec(
        task_id=task.task_id,
        tier=task.tier,
        track=task.track,
        description=task.description,
        input_schema=augmented_schema,
        output_type=task.output_type,
        n_classes=task.n_classes,
        reference_algorithm=augmented_reference,
        input_sampler=lambda seed: augmented_schema.sample(seed),
        verifier=task.verifier,
        complexity_metadata={
            **task.complexity_metadata,
            "distractor_count": distractor_count,
        },
    )


def _aggregate_curve_runs(
    values_by_level: Mapping[int | float, List[float]],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(level): {
            "mean": round(float(np.mean(values)), 4),
            "std": round(float(np.std(values)), 4),
            "per_seed_accuracy": [round(float(value), 4) for value in values],
        }
        for level, values in values_by_level.items()
    }


def _save_metric_curve_plot(
    output_path: Path,
    title: str,
    x_values: Sequence[int | float],
    by_model: Mapping[str, Dict[str, Dict[str, Any]]],
    x_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ordered_x = list(x_values)
    for model_name, values in sorted(by_model.items()):
        y_values = [values[str(level)]["mean"] for level in ordered_x]
        ax.plot(ordered_x, y_values, marker="o", label=model_name)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _render_d2_d3_summary(
    title: str,
    summary: Mapping[str, Dict[str, Any]],
    selected_key: str,
    score_key: str,
    pass_key: str,
) -> str:
    lines = [
        f"# {title}",
        "",
        "| Task | Selected Model | Score | Pass |",
        "|---|---|---|---|",
    ]
    for task_id, payload in sorted(summary.items()):
        lines.append(
            f"| {task_id} | {payload[selected_key]} | {payload[score_key]:.4f} | "
            f"{'PASS' if payload[pass_key] else 'FAIL'} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def run_distractor_robustness_experiment(
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
    task_ids: Optional[Sequence[str]] = None,
    model_configs: Optional[Sequence[ModelConfig]] = None,
    distractor_counts: Optional[Sequence[int]] = None,
    seeds: Optional[Sequence[int]] = None,
    n_samples: int = TASK14_D2_N_SAMPLES,
    train_fraction: float = TASK14_TRAIN_FRACTION,
) -> DiagnosticExperimentArtifact:
    """Run EXP-D2 distractor robustness using task-level feature augmentation."""

    active_registry = build_default_registry() if registry is None else registry
    active_task_ids = list(task_ids or TASK14_D2_TASK_IDS)
    active_models = list(model_configs or _default_diagnostic_model_configs())
    active_counts = list(distractor_counts or TASK14_D2_DISTRACTOR_COUNTS)
    active_seeds = list(seeds or TASK14_SEEDS)

    results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}
    output_dir = _ensure_dir(Path(output_root) / "EXP-D2")
    per_task_dir = _ensure_dir(output_dir / "per_task")

    for task_id in active_task_ids:
        task = active_registry.get(task_id)
        split_cache_by_count = {
            count: _build_iid_split_cache(
                _clone_task_with_distractors(task, count),
                active_seeds,
                n_samples,
                train_fraction,
            )
            for count in active_counts
        }
        by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for model_config in active_models:
            per_count: Dict[int, List[float]] = {count: [] for count in active_counts}
            for count in active_counts:
                augmented_task = _clone_task_with_distractors(task, count)
                for seed in active_seeds:
                    split = split_cache_by_count[count][seed]
                    per_count[count].append(
                        _run_accuracy_from_samples(
                            augmented_task,
                            model_config,
                            split.train,
                            split.test,
                        )
                    )
            by_model[model_config.name or model_config.family.value] = _aggregate_curve_runs(per_count)

        results[task_id] = by_model
        baseline_model = max(
            by_model,
            key=lambda model_name: by_model[model_name][str(active_counts[0])]["mean"],
        )
        baseline_acc = by_model[baseline_model][str(active_counts[0])]["mean"]
        max_distractor_acc = by_model[baseline_model][str(active_counts[-1])]["mean"]
        accuracy_drop = round(float(max(0.0, baseline_acc - max_distractor_acc)), 4)
        robustness_score = round(float(np.clip(1.0 - accuracy_drop, 0.0, 1.0)), 4)
        summary[task_id] = {
            "selected_model": baseline_model,
            "baseline_accuracy": baseline_acc,
            "max_distractor_accuracy": max_distractor_acc,
            "accuracy_drop": accuracy_drop,
            "distractor_robustness_score": robustness_score,
            "criterion_7_distractor_robustness": accuracy_drop <= 0.05,
        }

        task_output_dir = _ensure_dir(per_task_dir / task_id)
        _save_metric_curve_plot(
            task_output_dir / "distractor_curve.png",
            f"Distractor Robustness: {task_id}",
            active_counts,
            by_model,
            "Distractor count",
        )

    payload = {
        "experiment_id": "EXP-D2",
        "distractor_counts": active_counts,
        "seeds": active_seeds,
        "n_samples": n_samples,
        "results": results,
        "task_summary": summary,
    }

    _write_json(output_dir / "config.json", {
        "task_ids": active_task_ids,
        "distractor_counts": active_counts,
        "seeds": active_seeds,
        "n_samples": n_samples,
        "model_configs": [
            {
                "family": config.family.value,
                "name": config.name,
                "hyperparams": config.hyperparams,
            }
            for config in active_models
        ],
    })
    _write_json(output_dir / "distractor_robustness.json", payload)
    (output_dir / "summary.md").write_text(
        _render_d2_d3_summary(
            "EXP-D2 Distractor Feature Robustness",
            summary,
            "selected_model",
            "distractor_robustness_score",
            "criterion_7_distractor_robustness",
        ),
        encoding="utf-8",
    )
    return DiagnosticExperimentArtifact(
        experiment_id="EXP-D2",
        output_dir=output_dir,
        payload=payload,
    )


def run_noise_robustness_experiment(
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
    task_ids: Optional[Sequence[str]] = None,
    model_configs: Optional[Sequence[ModelConfig]] = None,
    noise_levels: Optional[Sequence[float]] = None,
    seeds: Optional[Sequence[int]] = None,
    n_samples: int = TASK14_D3_N_SAMPLES,
    train_fraction: float = TASK14_TRAIN_FRACTION,
) -> DiagnosticExperimentArtifact:
    """Run EXP-D3 noise robustness across supported classification diagnostics."""

    active_registry = build_default_registry() if registry is None else registry
    active_task_ids = list(task_ids or TASK14_D3_TASK_IDS)
    active_models = list(model_configs or _default_diagnostic_model_configs())
    active_noise_levels = list(noise_levels or TASK14_D3_NOISE_LEVELS)
    active_seeds = list(seeds or TASK14_SEEDS)

    results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    summary: Dict[str, Dict[str, Any]] = {}
    output_dir = _ensure_dir(Path(output_root) / "EXP-D3")
    per_task_dir = _ensure_dir(output_dir / "per_task")

    for task_id in active_task_ids:
        task = active_registry.get(task_id)
        split_cache = _build_noise_split_cache(
            task,
            active_seeds,
            n_samples,
            train_fraction,
            active_noise_levels,
        )
        by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for model_config in active_models:
            per_level: Dict[float, List[float]] = {level: [] for level in active_noise_levels}
            for level in active_noise_levels:
                for seed in active_seeds:
                    split = split_cache[level][seed]
                    per_level[level].append(
                        _run_accuracy_from_samples(task, model_config, split.train, split.test)
                    )
            by_model[model_config.name or model_config.family.value] = _aggregate_curve_runs(per_level)

        results[task_id] = by_model
        selected_model = max(
            by_model,
            key=lambda model_name: by_model[model_name][str(active_noise_levels[0])]["mean"],
        )
        clean_acc = by_model[selected_model][str(active_noise_levels[0])]["mean"]
        max_noise_acc = by_model[selected_model][str(active_noise_levels[-1])]["mean"]
        accuracy_drop = round(float(max(0.0, clean_acc - max_noise_acc)), 4)
        summary[task_id] = {
            "selected_model": selected_model,
            "clean_accuracy": clean_acc,
            "max_noise_accuracy": max_noise_acc,
            "noise_accuracy_drop": accuracy_drop,
            "smooth_degradation": all(
                by_model[selected_model][str(active_noise_levels[idx])]["mean"]
                >= by_model[selected_model][str(active_noise_levels[idx + 1])]["mean"] - 0.05
                for idx in range(len(active_noise_levels) - 1)
            ),
        }

        task_output_dir = _ensure_dir(per_task_dir / task_id)
        _save_metric_curve_plot(
            task_output_dir / "noise_curve.png",
            f"Noise Robustness: {task_id}",
            active_noise_levels,
            by_model,
            "Noise level",
        )

    payload = {
        "experiment_id": "EXP-D3",
        "noise_levels": active_noise_levels,
        "seeds": active_seeds,
        "n_samples": n_samples,
        "results": results,
        "task_summary": summary,
    }

    _write_json(output_dir / "config.json", {
        "task_ids": active_task_ids,
        "noise_levels": active_noise_levels,
        "seeds": active_seeds,
        "n_samples": n_samples,
        "model_configs": [
            {
                "family": config.family.value,
                "name": config.name,
                "hyperparams": config.hyperparams,
            }
            for config in active_models
        ],
    })
    _write_json(output_dir / "noise_robustness.json", payload)
    (output_dir / "summary.md").write_text(
        _render_d2_d3_summary(
            "EXP-D3 Noise Robustness",
            {
                task_id: {
                    "selected_model": task_summary["selected_model"],
                    "noise_robustness_score": max(
                        0.0,
                        round(1.0 - task_summary["noise_accuracy_drop"], 4),
                    ),
                    "criterion_noise_robustness": task_summary["smooth_degradation"],
                }
                for task_id, task_summary in summary.items()
            },
            "selected_model",
            "noise_robustness_score",
            "criterion_noise_robustness",
        ),
        encoding="utf-8",
    )
    return DiagnosticExperimentArtifact(
        experiment_id="EXP-D3",
        output_dir=output_dir,
        payload=payload,
    )


def _fit_sklearn_model(
    train_inputs: Sequence[Any],
    train_outputs: Sequence[Any],
    model_config: ModelConfig,
) -> Tuple[SklearnModelWrapper, InputEncoder, LabelEncoder]:
    model = build_model(model_config)
    if not isinstance(model, SklearnModelWrapper):
        raise ValueError(
            f"Feature-importance diagnostics require sklearn models, got {model_config.family}"
        )

    encoder = InputEncoder()
    label_encoder = LabelEncoder()
    X_train = encoder.fit_transform(list(train_inputs))
    y_train = label_encoder.fit_transform([str(output) for output in train_outputs])
    model.fit(X_train, y_train)
    return model, encoder, label_encoder


def _compute_alignment_metrics(
    feature_importances: Mapping[str, float],
    relevant_features: Sequence[str],
) -> Dict[str, Any]:
    relevant = set(relevant_features)
    ordered = [
        feature_name
        for feature_name, _ in sorted(
            feature_importances.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    top_k = ordered[: len(relevant_features)]
    top_k_set = set(top_k)
    intersection = relevant & top_k_set
    jaccard = len(intersection) / max(len(relevant | top_k_set), 1)
    mean_rank = float(
        np.mean([ordered.index(feature_name) + 1 for feature_name in relevant_features])
    )
    return {
        "top_k_features": top_k,
        "precision_at_k": round(len(intersection) / max(len(relevant_features), 1), 4),
        "jaccard_at_k": round(jaccard, 4),
        "mean_relevant_rank": round(mean_rank, 4),
    }


def _save_feature_alignment_plot(
    output_path: Path,
    task_id: str,
    feature_names: Sequence[str],
    by_model: Mapping[str, Dict[str, float]],
) -> None:
    model_names = list(sorted(by_model))
    matrix = np.array(
        [
            [by_model[model_name].get(feature_name, 0.0) for model_name in model_names]
            for feature_name in feature_names
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, len(feature_names) * 0.45)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(model_names)), model_names, rotation=30, ha="right")
    ax.set_yticks(range(len(feature_names)), feature_names)
    ax.set_title(f"Permutation Importance Alignment: {task_id}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_feature_importance_alignment_experiment(
    output_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
    task_ids: Optional[Sequence[str]] = None,
    model_configs: Optional[Sequence[ModelConfig]] = None,
    seeds: Optional[Sequence[int]] = None,
    n_samples: int = TASK14_D4_N_SAMPLES,
    distractor_count: int = TASK14_D4_DISTRACTOR_COUNT,
    train_fraction: float = TASK14_TRAIN_FRACTION,
) -> DiagnosticExperimentArtifact:
    """Run EXP-D4 feature-importance alignment with permutation importance."""

    active_registry = build_default_registry() if registry is None else registry
    active_task_ids = list(task_ids or TASK14_D4_TASK_IDS)
    active_models = list(model_configs or _default_diagnostic_model_configs())
    active_seeds = list(seeds or TASK14_SEEDS)

    results: Dict[str, Dict[str, Any]] = {}
    output_dir = _ensure_dir(Path(output_root) / "EXP-D4")
    per_task_dir = _ensure_dir(output_dir / "per_task")

    for task_id in active_task_ids:
        base_task = active_registry.get(task_id)
        augmented_task = _clone_task_with_distractors(base_task, distractor_count)
        relevant_features = list(TASK14_RELEVANT_FEATURES[task_id])
        task_results: Dict[str, Any] = {}
        plot_payload: Dict[str, Dict[str, float]] = {}
        split_cache = _build_iid_split_cache(
            augmented_task,
            active_seeds,
            n_samples,
            train_fraction,
        )

        for model_config in active_models:
            accuracy_values: List[float] = []
            alignment_values: List[Dict[str, Any]] = []
            importances_by_feature: Dict[str, List[float]] = {}

            for seed in active_seeds:
                split = split_cache[seed]
                model, encoder, label_encoder = _fit_sklearn_model(
                    split.train_inputs,
                    split.train_outputs,
                    model_config,
                )
                X_test = encoder.transform(split.test_inputs)
                y_test = label_encoder.transform([str(output) for output in split.test_outputs])
                y_pred = model.predict(X_test)
                accuracy_values.append(float(np.mean(y_pred == y_test)))

                importance = permutation_importance(
                    model.estimator,
                    X_test,
                    y_test,
                    n_repeats=5,
                    random_state=seed,
                    n_jobs=1,
                )
                feature_importances = {
                    feature_name: float(value)
                    for feature_name, value in zip(encoder.feature_names, importance.importances_mean)
                }
                alignment_values.append(
                    _compute_alignment_metrics(feature_importances, relevant_features)
                )
                for feature_name, value in feature_importances.items():
                    importances_by_feature.setdefault(feature_name, []).append(value)

            mean_importances = {
                feature_name: round(float(np.mean(values)), 4)
                for feature_name, values in sorted(importances_by_feature.items())
            }
            model_name = model_config.name or model_config.family.value
            plot_payload[model_name] = mean_importances
            task_results[model_name] = {
                "accuracy_mean": round(float(np.mean(accuracy_values)), 4),
                "accuracy_std": round(float(np.std(accuracy_values)), 4),
                "precision_at_k_mean": round(
                    float(np.mean([entry["precision_at_k"] for entry in alignment_values])),
                    4,
                ),
                "jaccard_at_k_mean": round(
                    float(np.mean([entry["jaccard_at_k"] for entry in alignment_values])),
                    4,
                ),
                "mean_relevant_rank": round(
                    float(np.mean([entry["mean_relevant_rank"] for entry in alignment_values])),
                    4,
                ),
                "top_k_features": alignment_values[0]["top_k_features"],
                "mean_feature_importances": mean_importances,
            }

        feature_names = list(
            dict.fromkeys(
                relevant_features
                + list(
                    sorted(
                        {
                            feature_name
                            for model_payload in plot_payload.values()
                            for feature_name in model_payload
                        }
                    )
                )
            )
        )
        task_output_dir = _ensure_dir(per_task_dir / task_id)
        _save_feature_alignment_plot(
            task_output_dir / "feature_alignment.png",
            task_id,
            feature_names,
            plot_payload,
        )
        results[task_id] = {
            "relevant_features": relevant_features,
            "distractor_count": distractor_count,
            "models": task_results,
        }

    payload = {
        "experiment_id": "EXP-D4",
        "seeds": active_seeds,
        "n_samples": n_samples,
        "distractor_count": distractor_count,
        "results": results,
    }
    _write_json(output_dir / "config.json", {
        "task_ids": active_task_ids,
        "seeds": active_seeds,
        "n_samples": n_samples,
        "distractor_count": distractor_count,
        "model_configs": [
            {
                "family": config.family.value,
                "name": config.name,
                "hyperparams": config.hyperparams,
            }
            for config in active_models
        ],
    })
    _write_json(output_dir / "feature_importance_alignment.json", payload)

    lines = [
        "# EXP-D4 Feature Importance Alignment",
        "",
        "| Task | Model | Precision@k | Mean Relevant Rank |",
        "|---|---|---|---|",
    ]
    for task_id, task_payload in sorted(results.items()):
        for model_name, model_payload in sorted(task_payload["models"].items()):
            lines.append(
                f"| {task_id} | {model_name} | {model_payload['precision_at_k_mean']:.4f} | "
                f"{model_payload['mean_relevant_rank']:.4f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return DiagnosticExperimentArtifact(
        experiment_id="EXP-D4",
        output_dir=output_dir,
        payload=payload,
    )


def _calibrated_label(
    evidence: Mapping[str, bool],
    best_iid_accuracy: Optional[float],
) -> str:
    minimum_keys = [
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
    all_minimum = all(evidence.get(key, False) for key in minimum_keys)
    optional_count = sum(1 for key in optional_keys if evidence.get(key, False))

    if all_minimum and optional_count >= 2:
        return "STRONG"
    if all_minimum:
        return "MODERATE"
    if evidence.get("criterion_1_high_iid_accuracy", False):
        return "WEAK"
    if best_iid_accuracy is not None and best_iid_accuracy < 0.60:
        return "NEGATIVE"
    return "NEGATIVE" if not any(evidence.values()) else "INCONCLUSIVE"


def run_solvability_calibration_experiment(
    output_root: str | Path = "results",
    results_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
    baseline_records: Optional[Mapping[str, BaselineTaskRecord]] = None,
    d1_payload: Optional[Mapping[str, Any]] = None,
    d2_payload: Optional[Mapping[str, Any]] = None,
    d3_payload: Optional[Mapping[str, Any]] = None,
    d4_payload: Optional[Mapping[str, Any]] = None,
) -> DiagnosticExperimentArtifact:
    """Run EXP-D5 by combining baseline verdicts with D1-D4 diagnostic evidence."""

    active_registry = build_default_registry() if registry is None else registry
    records = (
        collect_baseline_task_records(results_root=results_root, registry=active_registry)
        if baseline_records is None
        else dict(baseline_records)
    )
    if d1_payload is None:
        d1_payload = _load_json(Path(output_root) / "EXP-D1" / "sample_efficiency.json")
    if d2_payload is None:
        d2_payload = _load_json(Path(output_root) / "EXP-D2" / "distractor_robustness.json")
    if d3_payload is None:
        d3_path = Path(output_root) / "EXP-D3" / "noise_robustness.json"
        d3_payload = _load_json(d3_path) if d3_path.exists() else {"task_summary": {}}
    if d4_payload is None:
        d4_path = Path(output_root) / "EXP-D4" / "feature_importance_alignment.json"
        d4_payload = _load_json(d4_path) if d4_path.exists() else {"results": {}}

    d1_curves = dict(d1_payload.get("task_curves", {}))
    d2_summary = dict(d2_payload.get("task_summary", {}))
    d3_summary = dict(d3_payload.get("task_summary", {}))
    d4_results = dict(d4_payload.get("results", {}))

    tasks_payload: Dict[str, Any] = {}
    for task_id, record in sorted(records.items()):
        evidence = dict(record.evidence)
        notes = list(record.notes)
        sample_curve = d1_curves.get(task_id)
        distractor_summary = d2_summary.get(task_id)
        noise_summary = d3_summary.get(task_id)
        alignment_summary = d4_results.get(task_id)

        sample_score = 0.0
        if sample_curve is not None:
            evidence["criterion_8_sample_efficiency"] = bool(
                sample_curve.get("criterion_8_sample_efficiency", False)
            )
            sample_score = float(sample_curve.get("sample_efficiency_score", 0.0))
            notes.append(
                "TASK-14 sample-efficiency curve delta vs control: "
                f"{sample_curve.get('delta_vs_control_auc', 0.0):.4f}."
            )

        distractor_score = 0.0
        if distractor_summary is not None:
            evidence["criterion_7_distractor_robustness"] = bool(
                distractor_summary.get("criterion_7_distractor_robustness", False)
            )
            distractor_score = float(
                distractor_summary.get("distractor_robustness_score", 0.0)
            )
            notes.append(
                "TASK-14 distractor accuracy drop at the max feature count: "
                f"{distractor_summary.get('accuracy_drop', 0.0):.4f}."
            )

        calibrated_label = _calibrated_label(evidence, record.best_iid_accuracy)
        calibrated_score = round(
            float(np.clip(record.score + (0.10 * sample_score) + (0.10 * distractor_score), 0.0, 1.0)),
            4,
        )
        if noise_summary is not None:
            notes.append(
                "Noise robustness selected model "
                f"{noise_summary['selected_model']} drops by "
                f"{noise_summary['noise_accuracy_drop']:.4f}."
            )
        if alignment_summary is not None:
            best_alignment_model = max(
                alignment_summary["models"],
                key=lambda model_name: alignment_summary["models"][model_name]["precision_at_k_mean"],
            )
            notes.append(
                "Feature-importance alignment best model "
                f"{best_alignment_model} reaches precision@k "
                f"{alignment_summary['models'][best_alignment_model]['precision_at_k_mean']:.4f}."
            )

        tasks_payload[task_id] = {
            "task_id": task_id,
            "tier": record.tier,
            "track": record.track,
            "source_experiment": record.source_experiment,
            "baseline_label": record.label,
            "baseline_score": record.score,
            "calibrated_label": calibrated_label,
            "calibrated_score": calibrated_score,
            "best_model": record.best_model_name,
            "best_iid_accuracy": record.best_iid_accuracy,
            "best_ood_accuracy": record.best_ood_accuracy,
            "evidence": evidence,
            "diagnostic_support": {
                "sample_efficiency_score": round(sample_score, 4),
                "distractor_robustness_score": round(distractor_score, 4),
            },
            "notes": notes,
        }

    calibration_checks = {
        "controls_negative_or_weak": all(
            tasks_payload[task_id]["calibrated_label"] in {"NEGATIVE", "WEAK"}
            for task_id in TASK14_CONTROL_TASK_IDS
            if task_id in tasks_payload
        ),
        "trivial_tasks_strong": all(
            tasks_payload[task_id]["calibrated_label"] == "STRONG"
            for task_id in TASK14_TRIVIAL_TASK_IDS
            if task_id in tasks_payload
        ),
    }

    payload = {
        "experiment_id": "EXP-D5",
        "tasks": tasks_payload,
        "calibration_checks": calibration_checks,
    }

    output_dir = _ensure_dir(Path(output_root) / "EXP-D5")
    _write_json(output_dir / "config.json", {
        "baseline_experiment_ids": list(TASK14_BASELINE_EXPERIMENT_IDS),
        "control_task_ids": list(TASK14_CONTROL_TASK_IDS),
        "trivial_task_ids": list(TASK14_TRIVIAL_TASK_IDS),
    })
    _write_json(output_dir / "solvability_calibration.json", payload)

    lines = [
        "# EXP-D5 Solvability Verdict Calibration",
        "",
        f"- Controls negative/weak: {calibration_checks['controls_negative_or_weak']}",
        f"- Trivial tasks strong: {calibration_checks['trivial_tasks_strong']}",
        "",
        "| Task | Baseline | Calibrated | Baseline Score | Calibrated Score |",
        "|---|---|---|---|---|",
    ]
    for task_id, task_payload in sorted(tasks_payload.items()):
        lines.append(
            f"| {task_id} | {task_payload['baseline_label']} | "
            f"{task_payload['calibrated_label']} | {task_payload['baseline_score']:.4f} | "
            f"{task_payload['calibrated_score']:.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return DiagnosticExperimentArtifact(
        experiment_id="EXP-D5",
        output_dir=output_dir,
        payload=payload,
    )


def run_all_diagnostic_experiments(
    output_root: str | Path = "results",
    results_root: str | Path = "results",
    registry: Optional[TaskRegistry] = None,
) -> Dict[str, DiagnosticExperimentArtifact]:
    """Run the full TASK-14 diagnostic suite."""

    active_registry = build_default_registry() if registry is None else registry
    artifacts: Dict[str, DiagnosticExperimentArtifact] = {}
    artifacts["EXP-D1"] = run_sample_efficiency_experiment(
        output_root=output_root,
        results_root=results_root,
        registry=active_registry,
    )
    artifacts["EXP-D2"] = run_distractor_robustness_experiment(
        output_root=output_root,
        registry=active_registry,
    )
    artifacts["EXP-D3"] = run_noise_robustness_experiment(
        output_root=output_root,
        registry=active_registry,
    )
    artifacts["EXP-D4"] = run_feature_importance_alignment_experiment(
        output_root=output_root,
        registry=active_registry,
    )
    baseline_records = collect_baseline_task_records(
        results_root=results_root,
        registry=active_registry,
    )
    artifacts["EXP-D5"] = run_solvability_calibration_experiment(
        output_root=output_root,
        results_root=results_root,
        registry=active_registry,
        baseline_records=baseline_records,
        d1_payload=artifacts["EXP-D1"].payload,
        d2_payload=artifacts["EXP-D2"].payload,
        d3_payload=artifacts["EXP-D3"].payload,
        d4_payload=artifacts["EXP-D4"].payload,
    )
    return artifacts
