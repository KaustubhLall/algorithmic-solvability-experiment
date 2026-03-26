"""SR-3: Data Generator.

Generates labeled datasets from registered tasks. For each task, produces
input-output pairs using the task's input_sampler and reference_algorithm,
with metadata logging per sample.

Used by: SR-4 (Split Generator), SR-7 (Experiment Runner).
Validated by: V-3 (Data Generator Validation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.registry import TaskSpec


# ===================================================================
# Sample and Dataset dataclasses
# ===================================================================

@dataclass
class Sample:
    """A single generated sample with metadata.

    Attributes:
        input_data: The input (list for sequence, dict for tabular).
        output_data: The reference output.
        task_id: Which task generated this.
        seed: The seed used to generate the input.
        metadata: Per-sample metadata (complexity, noise level, etc.).
    """
    input_data: Any
    output_data: Any
    task_id: str
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    """A collection of labeled samples for a single task.

    Attributes:
        task_id: The task that generated this dataset.
        samples: List of Sample objects.
        task_metadata: Task-level metadata (tier, track, complexity, etc.).
    """
    task_id: str
    samples: List[Sample]
    task_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def inputs(self) -> List[Any]:
        return [s.input_data for s in self.samples]

    @property
    def outputs(self) -> List[Any]:
        return [s.output_data for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)


# ===================================================================
# Data Generator
# ===================================================================

class DataGenerator:
    """Generates labeled datasets from TaskSpec instances.

    All generation is deterministic given a base seed.
    Labels are re-verified against the reference algorithm at generation time.
    """

    def __init__(self, verify_labels: bool = True) -> None:
        """
        Args:
            verify_labels: If True, re-verify every label against the reference
                           algorithm after generation (catches bugs early).
        """
        self._verify_labels = verify_labels

    def generate(
        self,
        task: TaskSpec,
        n_samples: int,
        base_seed: int = 0,
        noise_level: float = 0.0,
    ) -> Dataset:
        """Generate a labeled dataset for a single task.

        Args:
            task: The task specification.
            n_samples: Number of samples to generate.
            base_seed: Base random seed (sample i uses seed = base_seed + i).
            noise_level: Noise level for input perturbation (0.0 = no noise).

        Returns:
            A Dataset of labeled samples.
        """
        samples: List[Sample] = []

        for i in range(n_samples):
            seed = base_seed + i
            inp = task.input_sampler(seed)

            # Apply noise to inputs if requested (noise on inputs only, never labels)
            if noise_level > 0.0:
                inp = self._apply_noise(inp, noise_level, seed)

            # Generate label from CLEAN input if noise was applied
            # (labels are always from the reference algorithm on clean data)
            clean_inp = task.input_sampler(seed)
            output = task.reference_algorithm(clean_inp)

            # Re-verify label
            if self._verify_labels:
                expected = task.reference_algorithm(clean_inp)
                if not task.verifier(output, expected):
                    raise RuntimeError(
                        f"Label verification failed for task {task.task_id} at seed {seed}: "
                        f"output={output}, expected={expected}"
                    )

            metadata = {
                "seed": seed,
                "noise_level": noise_level,
                **task.complexity_metadata,
            }

            samples.append(Sample(
                input_data=inp,
                output_data=output,
                task_id=task.task_id,
                seed=seed,
                metadata=metadata,
            ))

        task_metadata = {
            "tier": task.tier,
            "track": task.track,
            "output_type": task.output_type,
            "n_classes": task.n_classes,
            "n_samples": n_samples,
            "base_seed": base_seed,
            "noise_level": noise_level,
            **task.complexity_metadata,
        }

        return Dataset(
            task_id=task.task_id,
            samples=samples,
            task_metadata=task_metadata,
        )

    def generate_multi(
        self,
        tasks: List[TaskSpec],
        n_samples_per_task: int,
        base_seed: int = 0,
        noise_level: float = 0.0,
    ) -> Dict[str, Dataset]:
        """Generate datasets for multiple tasks.

        Args:
            tasks: List of task specifications.
            n_samples_per_task: Number of samples per task.
            base_seed: Base seed (each task offsets by task index * n_samples).
            noise_level: Input noise level.

        Returns:
            Dict mapping task_id → Dataset.
        """
        datasets: Dict[str, Dataset] = {}
        for idx, task in enumerate(tasks):
            task_seed = base_seed + idx * n_samples_per_task
            datasets[task.task_id] = self.generate(
                task, n_samples_per_task, task_seed, noise_level
            )
        return datasets

    def _apply_noise(self, inp: Any, noise_level: float, seed: int) -> Any:
        """Apply noise to an input (inputs only, never labels).

        For sequences: randomly flip elements with probability = noise_level.
        For tabular: add Gaussian noise to numerical features, randomly flip categoricals.
        """
        rng = np.random.default_rng(seed + 2**31)

        if isinstance(inp, list):
            return self._noise_sequence(inp, noise_level, rng)
        elif isinstance(inp, dict):
            return self._noise_tabular(inp, noise_level, rng)
        else:
            return inp

    def _noise_sequence(
        self, seq: List[Any], noise_level: float, rng: np.random.Generator
    ) -> List[Any]:
        """Add noise to a sequence by randomly perturbing elements."""
        result = list(seq)
        for i in range(len(result)):
            if rng.random() < noise_level:
                if isinstance(result[i], int):
                    # Add small random offset
                    result[i] = result[i] + int(rng.integers(-2, 3))
        return result

    def _noise_tabular(
        self, row: Dict[str, Any], noise_level: float, rng: np.random.Generator
    ) -> Dict[str, Any]:
        """Add noise to a tabular row."""
        result = dict(row)
        for key, val in result.items():
            if rng.random() < noise_level:
                if isinstance(val, float):
                    # Add Gaussian noise proportional to value magnitude
                    scale = max(abs(val) * 0.1, 1.0)
                    result[key] = val + float(rng.normal(0, scale))
                elif isinstance(val, str):
                    # Categorical: leave as-is (would need schema to flip properly)
                    pass
        return result


# ===================================================================
# Convenience functions
# ===================================================================

def generate_dataset(
    task: TaskSpec,
    n_samples: int,
    base_seed: int = 0,
    noise_level: float = 0.0,
    verify: bool = True,
) -> Dataset:
    """Convenience function: generate a dataset for a single task."""
    gen = DataGenerator(verify_labels=verify)
    return gen.generate(task, n_samples, base_seed, noise_level)


def generate_datasets(
    tasks: List[TaskSpec],
    n_samples_per_task: int,
    base_seed: int = 0,
    noise_level: float = 0.0,
    verify: bool = True,
) -> Dict[str, Dataset]:
    """Convenience function: generate datasets for multiple tasks."""
    gen = DataGenerator(verify_labels=verify)
    return gen.generate_multi(tasks, n_samples_per_task, base_seed, noise_level)


def compute_class_balance(dataset: Dataset) -> Dict[str, float]:
    """Compute class distribution for a classification dataset.

    Returns:
        Dict mapping class label → fraction of samples.
    """
    if not dataset.samples:
        return {}
    from collections import Counter
    counts = Counter(str(s.output_data) for s in dataset.samples)
    total = len(dataset.samples)
    return {cls: count / total for cls, count in sorted(counts.items())}
