"""SR-1: Task Registry.

Central registry where each task is a named entry exposing a standard interface.
Every experiment looks up its tasks from this registry.

Used by: All experiments.
Validated by: V-1 (Task Registry Validation).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.schemas import (
    CategoricalFeatureSpec,
    Distribution,
    ElementType,
    InputSchema,
    NumericalFeatureSpec,
    SequenceInputSchema,
    TabularInputSchema,
)


# ===================================================================
# TaskSpec dataclass
# ===================================================================

@dataclass
class TaskSpec:
    """Specification for a single registered task.

    Attributes:
        task_id: Unique identifier, e.g. "S1.2_sort".
        tier: Tier label, e.g. "S1", "C2".
        track: "sequence" or "classification".
        description: Human-readable description.
        input_schema: Defines feature types, shapes, ranges.
        output_type: "sequence", "scalar", or "class".
        n_classes: Number of output classes (classification only).
        reference_algorithm: Ground-truth function: input → output.
        input_sampler: Generates random valid inputs given a seed.
        verifier: Checks if a prediction matches the reference output.
        complexity_metadata: Depth, n_features_used, statefulness, etc.
    """
    task_id: str
    tier: str
    track: str
    description: str
    input_schema: InputSchema
    output_type: str
    n_classes: Optional[int]
    reference_algorithm: Callable[[Any], Any]
    input_sampler: Callable[[int], Any]
    verifier: Callable[[Any, Any], bool]
    complexity_metadata: Dict[str, Any] = field(default_factory=dict)


# ===================================================================
# Registry
# ===================================================================

class TaskRegistry:
    """Central registry for all experiment tasks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskSpec] = {}

    def register(self, task: TaskSpec) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"Task '{task.task_id}' is already registered")
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> TaskSpec:
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found in registry")
        return self._tasks[task_id]

    def all_tasks(self) -> List[TaskSpec]:
        return list(self._tasks.values())

    def by_tier(self, tier: str) -> List[TaskSpec]:
        return [t for t in self._tasks.values() if t.tier == tier]

    def by_track(self, track: str) -> List[TaskSpec]:
        return [t for t in self._tasks.values() if t.track == track]

    def task_ids(self) -> List[str]:
        return list(self._tasks.keys())

    def __len__(self) -> int:
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._tasks


# ===================================================================
# Helper: generic verifiers
# ===================================================================

def exact_match_verifier(prediction: Any, expected: Any) -> bool:
    """Verifier for exact match (sequences and scalars)."""
    return prediction == expected


def classification_verifier(prediction: Any, expected: Any) -> bool:
    """Verifier for classification tasks (string class labels)."""
    return str(prediction) == str(expected)


def _stable_hash_int(parts: Sequence[Any]) -> int:
    """Return a reproducible 32-bit hash for arbitrary structured values."""
    joined = "||".join(repr(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


# ===================================================================
# Task builders: Sequence Track
# ===================================================================

def _seq_schema(min_len: int = 3, max_len: int = 15, val_lo: int = 0, val_hi: int = 9,
                elem_type: ElementType = ElementType.INT) -> SequenceInputSchema:
    if elem_type == ElementType.BINARY:
        return SequenceInputSchema(element_type=elem_type, min_length=min_len, max_length=max_len, value_range=(0, 1))
    return SequenceInputSchema(element_type=elem_type, min_length=min_len, max_length=max_len, value_range=(val_lo, val_hi))


def _seq_sampler(schema: SequenceInputSchema) -> Callable[[int], List[int]]:
    def sampler(seed: int) -> List[int]:
        return schema.sample(seed)
    return sampler


def _build_s0_tasks() -> List[TaskSpec]:
    """S0: Control tasks (non-algorithmic)."""
    tasks = []

    # S0.1 Random labels
    schema = _seq_schema()
    def s0_1_ref(inp: List[int]) -> List[int]:
        # Deterministic but unrelated to input structure
        h = _stable_hash_int(inp)
        rng = np.random.default_rng(h)
        return [int(x) for x in rng.integers(0, 10, size=len(inp))]
    tasks.append(TaskSpec(
        task_id="S0.1_random_labels", tier="S0", track="sequence",
        description="Random labels: output is pseudo-random, deterministic but unlearnable",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=s0_1_ref, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 0, "statefulness": False, "is_control": True},
    ))

    # S0.2 Lookup table
    lookup = {i: (i * 7 + 3) % 10 for i in range(10)}
    schema_lu = _seq_schema()
    def s0_2_ref(inp: List[int]) -> List[int]:
        return [lookup.get(x, 0) for x in inp]
    tasks.append(TaskSpec(
        task_id="S0.2_lookup_table", tier="S0", track="sequence",
        description="Lookup table: element-wise fixed mapping, no extrapolatable rule",
        input_schema=schema_lu, output_type="sequence", n_classes=None,
        reference_algorithm=s0_2_ref, input_sampler=_seq_sampler(schema_lu),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": False, "is_control": True},
    ))

    return tasks


def _build_s1_tasks() -> List[TaskSpec]:
    """S1: Simple one-step transforms."""
    tasks = []

    # S1.1 Reverse
    schema = _seq_schema()
    tasks.append(TaskSpec(
        task_id="S1.1_reverse", tier="S1", track="sequence",
        description="Reverse a sequence",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=lambda inp: list(reversed(inp)),
        input_sampler=_seq_sampler(schema), verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": False, "n_ops": 1},
    ))

    # S1.2 Sort
    tasks.append(TaskSpec(
        task_id="S1.2_sort", tier="S1", track="sequence",
        description="Sort a sequence ascending",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=lambda inp: sorted(inp),
        input_sampler=_seq_sampler(schema), verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": False, "n_ops": 1},
    ))

    # S1.3 Rotate by 1
    tasks.append(TaskSpec(
        task_id="S1.3_rotate", tier="S1", track="sequence",
        description="Rotate sequence left by 1 position",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=lambda inp: inp[1:] + inp[:1] if inp else [],
        input_sampler=_seq_sampler(schema), verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": False, "n_ops": 1},
    ))

    # S1.4 Count symbol (count occurrences of 0)
    tasks.append(TaskSpec(
        task_id="S1.4_count_symbol", tier="S1", track="sequence",
        description="Count occurrences of 0 in the sequence",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=lambda inp: [inp.count(0)],
        input_sampler=_seq_sampler(schema), verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": False, "n_ops": 1},
    ))

    # S1.5 Parity
    bin_schema = _seq_schema(elem_type=ElementType.BINARY)
    tasks.append(TaskSpec(
        task_id="S1.5_parity", tier="S1", track="sequence",
        description="XOR parity of a binary sequence",
        input_schema=bin_schema, output_type="sequence", n_classes=None,
        reference_algorithm=lambda inp: [sum(inp) % 2],
        input_sampler=_seq_sampler(bin_schema), verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": True, "n_ops": 1},
    ))

    # S1.6 Prefix sum
    tasks.append(TaskSpec(
        task_id="S1.6_prefix_sum", tier="S1", track="sequence",
        description="Cumulative sum of the input",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=lambda inp: [sum(inp[:i+1]) for i in range(len(inp))],
        input_sampler=_seq_sampler(schema), verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": True, "n_ops": 1},
    ))

    # S1.7 Deduplicate
    def dedup(inp: List[int]) -> List[int]:
        seen: set[int] = set()
        result: List[int] = []
        for x in inp:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result
    tasks.append(TaskSpec(
        task_id="S1.7_deduplicate", tier="S1", track="sequence",
        description="Remove duplicates preserving first occurrence",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=dedup, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": False, "n_ops": 1},
    ))

    # S1.8 Extrema (return [max, min])
    def extrema(inp: List[int]) -> List[int]:
        if not inp:
            return []
        return [max(inp), min(inp)]
    tasks.append(TaskSpec(
        task_id="S1.8_extrema", tier="S1", track="sequence",
        description="Return [max, min] of the sequence",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=extrema, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": False, "n_ops": 1},
    ))

    return tasks


def _build_s2_tasks() -> List[TaskSpec]:
    """S2: Stateful / iterative algorithms."""
    tasks = []

    # S2.1 Cumulative XOR
    bin_schema = _seq_schema(elem_type=ElementType.BINARY)
    def cum_xor(inp: List[int]) -> List[int]:
        result, running = [], 0
        for x in inp:
            running ^= x
            result.append(running)
        return result
    tasks.append(TaskSpec(
        task_id="S2.1_cumulative_xor", tier="S2", track="sequence",
        description="Running XOR over a bit string",
        input_schema=bin_schema, output_type="sequence", n_classes=None,
        reference_algorithm=cum_xor, input_sampler=_seq_sampler(bin_schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": True, "n_ops": 1},
    ))

    # S2.2 Balanced parentheses (0=open, 1=close) → [1] if balanced, [0] otherwise
    bin_schema2 = _seq_schema(elem_type=ElementType.BINARY, min_len=2, max_len=20)
    def balanced_parens(inp: List[int]) -> List[int]:
        depth = 0
        for x in inp:
            if x == 0:
                depth += 1
            else:
                depth -= 1
            if depth < 0:
                return [0]
        return [1 if depth == 0 else 0]
    tasks.append(TaskSpec(
        task_id="S2.2_balanced_parens", tier="S2", track="sequence",
        description="Check if bracket string is balanced (0=open, 1=close)",
        input_schema=bin_schema2, output_type="sequence", n_classes=None,
        reference_algorithm=balanced_parens, input_sampler=_seq_sampler(bin_schema2),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": True, "n_ops": 1},
    ))

    # S2.3 Running minimum
    schema = _seq_schema()
    def running_min(inp: List[int]) -> List[int]:
        if not inp:
            return []
        result, cur = [], inp[0]
        for x in inp:
            cur = min(cur, x)
            result.append(cur)
        return result
    tasks.append(TaskSpec(
        task_id="S2.3_running_min", tier="S2", track="sequence",
        description="Output running minimum sequence",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=running_min, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": True, "n_ops": 1},
    ))

    # S2.5 Checksum (sum mod 10)
    def checksum(inp: List[int]) -> List[int]:
        return [sum(inp) % 10]
    tasks.append(TaskSpec(
        task_id="S2.5_checksum", tier="S2", track="sequence",
        description="Compute modular checksum (sum mod 10)",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=checksum, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 1, "statefulness": True, "n_ops": 1},
    ))

    return tasks


def _build_s3_tasks() -> List[TaskSpec]:
    """S3: Multi-step compositional."""
    tasks = []
    schema = _seq_schema()

    # S3.1 Dedup-sort-count
    def dedup_sort_count(inp: List[int]) -> List[int]:
        seen: set[int] = set()
        deduped: List[int] = []
        for x in inp:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return [len(sorted(deduped))]
    tasks.append(TaskSpec(
        task_id="S3.1_dedup_sort_count", tier="S3", track="sequence",
        description="Deduplicate, sort, count unique elements",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=dedup_sort_count, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 3, "statefulness": False, "n_ops": 3},
    ))

    # S3.2 Filter even → sort → sum
    def filter_sort_sum(inp: List[int]) -> List[int]:
        evens = sorted([x for x in inp if x % 2 == 0])
        return [sum(evens)]
    tasks.append(TaskSpec(
        task_id="S3.2_filter_sort_sum", tier="S3", track="sequence",
        description="Filter even elements, sort, sum",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=filter_sort_sum, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 3, "statefulness": False, "n_ops": 3},
    ))

    # S3.4 Run-length encode
    def rle_encode(inp: List[int]) -> List[int]:
        if not inp:
            return []
        result: List[int] = []
        cur, count = inp[0], 1
        for x in inp[1:]:
            if x == cur:
                count += 1
            else:
                result.extend([cur, count])
                cur, count = x, 1
        result.extend([cur, count])
        return result
    tasks.append(TaskSpec(
        task_id="S3.4_rle_encode", tier="S3", track="sequence",
        description="Run-length encode: [1,1,2,2,2,3] → [1,2,2,3,3,1]",
        input_schema=schema, output_type="sequence", n_classes=None,
        reference_algorithm=rle_encode, input_sampler=_seq_sampler(schema),
        verifier=exact_match_verifier,
        complexity_metadata={"depth": 2, "statefulness": True, "n_ops": 2},
    ))

    return tasks


# ===================================================================
# Task builders: Classification Track
# ===================================================================

def _cls_schema_simple(n_num: int = 2, n_cat: int = 1,
                       num_range: tuple = (0.0, 100.0),
                       cat_values: tuple = ("A", "B", "C")) -> TabularInputSchema:
    nums = tuple(
        NumericalFeatureSpec(name=f"x{i+1}", min_val=num_range[0], max_val=num_range[1])
        for i in range(n_num)
    )
    cats = tuple(
        CategoricalFeatureSpec(name=f"cat{i+1}", values=cat_values)
        for i in range(n_cat)
    )
    return TabularInputSchema(numerical_features=nums, categorical_features=cats)


def _cls_sampler(schema: TabularInputSchema) -> Callable[[int], Dict[str, Any]]:
    def sampler(seed: int) -> Dict[str, Any]:
        return schema.sample(seed)
    return sampler


def _build_c0_tasks() -> List[TaskSpec]:
    """C0: Control tasks."""
    tasks = []
    schema = _cls_schema_simple()

    # C0.1 Random class
    def c0_1_ref(row: Dict[str, Any]) -> str:
        h = _stable_hash_int(sorted((k, str(v)) for k, v in row.items()))
        return ["A", "B", "C"][h % 3]
    tasks.append(TaskSpec(
        task_id="C0.1_random_class", tier="C0", track="classification",
        description="Random class: deterministic but unlearnable",
        input_schema=schema, output_type="class", n_classes=3,
        reference_algorithm=c0_1_ref, input_sampler=_cls_sampler(schema),
        verifier=classification_verifier,
        complexity_metadata={"depth": 0, "n_features_used": 0, "is_control": True},
    ))

    # C0.2 Majority class (always returns "A")
    tasks.append(TaskSpec(
        task_id="C0.2_majority_class", tier="C0", track="classification",
        description="Always returns class A (majority baseline)",
        input_schema=schema, output_type="class", n_classes=3,
        reference_algorithm=lambda row: "A", input_sampler=_cls_sampler(schema),
        verifier=classification_verifier,
        complexity_metadata={"depth": 0, "n_features_used": 0, "is_control": True},
    ))

    return tasks


def _build_c1_tasks() -> List[TaskSpec]:
    """C1: Single-rule threshold / boundary classification."""
    tasks = []

    # C1.1 Numeric threshold: x1 > 50 → A, else B
    schema = _cls_schema_simple(n_num=1, n_cat=0, num_range=(0.0, 100.0))
    tasks.append(TaskSpec(
        task_id="C1.1_numeric_threshold", tier="C1", track="classification",
        description="x1 > 50 → A, else B",
        input_schema=schema, output_type="class", n_classes=2,
        reference_algorithm=lambda row: "A" if row["x1"] > 50.0 else "B",
        input_sampler=_cls_sampler(schema), verifier=classification_verifier,
        complexity_metadata={"depth": 1, "n_features_used": 1},
    ))

    # C1.2 Range binning: 3 bins
    schema_bin = _cls_schema_simple(n_num=1, n_cat=0, num_range=(0.0, 90.0))
    def range_bin(row: Dict[str, Any]) -> str:
        x = row["x1"]
        if x < 30.0:
            return "LOW"
        elif x < 60.0:
            return "MID"
        else:
            return "HIGH"
    tasks.append(TaskSpec(
        task_id="C1.2_range_binning", tier="C1", track="classification",
        description="x1 binned into LOW/MID/HIGH",
        input_schema=schema_bin, output_type="class", n_classes=3,
        reference_algorithm=range_bin, input_sampler=_cls_sampler(schema_bin),
        verifier=classification_verifier,
        complexity_metadata={"depth": 1, "n_features_used": 1},
    ))

    # C1.3 Categorical match
    schema_cat = _cls_schema_simple(n_num=0, n_cat=1)
    tasks.append(TaskSpec(
        task_id="C1.3_categorical_match", tier="C1", track="classification",
        description="cat1 == 'A' → YES, else NO",
        input_schema=schema_cat, output_type="class", n_classes=2,
        reference_algorithm=lambda row: "YES" if row["cat1"] == "A" else "NO",
        input_sampler=_cls_sampler(schema_cat), verifier=classification_verifier,
        complexity_metadata={"depth": 1, "n_features_used": 1},
    ))

    # C1.5 Numeric comparison
    schema_cmp = _cls_schema_simple(n_num=2, n_cat=0)
    tasks.append(TaskSpec(
        task_id="C1.5_numeric_comparison", tier="C1", track="classification",
        description="x1 > x2 → A, else B",
        input_schema=schema_cmp, output_type="class", n_classes=2,
        reference_algorithm=lambda row: "A" if row["x1"] > row["x2"] else "B",
        input_sampler=_cls_sampler(schema_cmp), verifier=classification_verifier,
        complexity_metadata={"depth": 1, "n_features_used": 2},
    ))

    # C1.6 Modular class
    schema_mod = _cls_schema_simple(n_num=1, n_cat=0, num_range=(0.0, 99.0))
    def mod_class(row: Dict[str, Any]) -> str:
        return f"C{int(row['x1']) % 3}"
    tasks.append(TaskSpec(
        task_id="C1.6_modular_class", tier="C1", track="classification",
        description="class = x1 mod 3",
        input_schema=schema_mod, output_type="class", n_classes=3,
        reference_algorithm=mod_class, input_sampler=_cls_sampler(schema_mod),
        verifier=classification_verifier,
        complexity_metadata={"depth": 1, "n_features_used": 1},
    ))

    return tasks


def _build_c2_tasks() -> List[TaskSpec]:
    """C2: Multi-feature conjunctive / disjunctive rules."""
    tasks = []

    # C2.1 AND: x1 > 50 AND cat1 == "A" → YES, else NO
    schema = _cls_schema_simple()
    tasks.append(TaskSpec(
        task_id="C2.1_and_rule", tier="C2", track="classification",
        description="x1 > 50 AND cat1 == 'A' → YES, else NO",
        input_schema=schema, output_type="class", n_classes=2,
        reference_algorithm=lambda row: "YES" if (row["x1"] > 50.0 and row["cat1"] == "A") else "NO",
        input_sampler=_cls_sampler(schema), verifier=classification_verifier,
        complexity_metadata={"depth": 2, "n_features_used": 2},
    ))

    # C2.2 OR: x1 > 80 OR cat1 == "B" → YES, else NO
    tasks.append(TaskSpec(
        task_id="C2.2_or_rule", tier="C2", track="classification",
        description="x1 > 80 OR cat1 == 'B' → YES, else NO",
        input_schema=schema, output_type="class", n_classes=2,
        reference_algorithm=lambda row: "YES" if (row["x1"] > 80.0 or row["cat1"] == "B") else "NO",
        input_sampler=_cls_sampler(schema), verifier=classification_verifier,
        complexity_metadata={"depth": 2, "n_features_used": 2},
    ))

    # C2.3 Nested if-else
    schema_nested = _cls_schema_simple(n_num=2, n_cat=1)
    def nested_ite(row: Dict[str, Any]) -> str:
        if row["x1"] > 60.0:
            return "A" if row["cat1"] == "A" else "B"
        else:
            return "C" if row["x2"] > 50.0 else "D"
    tasks.append(TaskSpec(
        task_id="C2.3_nested_if_else", tier="C2", track="classification",
        description="Nested if-else over x1, cat1, x2",
        input_schema=schema_nested, output_type="class", n_classes=4,
        reference_algorithm=nested_ite, input_sampler=_cls_sampler(schema_nested),
        verifier=classification_verifier,
        complexity_metadata={"depth": 3, "n_features_used": 3},
    ))

    # C2.5 k-of-n: at least 2 of 3 conditions true
    schema_kon = _cls_schema_simple(n_num=2, n_cat=1)
    def k_of_n(row: Dict[str, Any]) -> str:
        conds = [row["x1"] > 50.0, row["x2"] > 50.0, row["cat1"] == "A"]
        return "YES" if sum(conds) >= 2 else "NO"
    tasks.append(TaskSpec(
        task_id="C2.5_k_of_n", tier="C2", track="classification",
        description="At least 2 of 3 conditions true",
        input_schema=schema_kon, output_type="class", n_classes=2,
        reference_algorithm=k_of_n, input_sampler=_cls_sampler(schema_kon),
        verifier=classification_verifier,
        complexity_metadata={"depth": 2, "n_features_used": 3},
    ))

    # C2.6 Categorical + numeric gate
    schema_gate = _cls_schema_simple(n_num=1, n_cat=1)
    def cat_gate(row: Dict[str, Any]) -> str:
        thresholds = {"A": 30.0, "B": 50.0, "C": 70.0}
        t = thresholds.get(row["cat1"], 50.0)
        return "PASS" if row["x1"] > t else "FAIL"
    tasks.append(TaskSpec(
        task_id="C2.6_categorical_gate", tier="C2", track="classification",
        description="Threshold depends on categorical value",
        input_schema=schema_gate, output_type="class", n_classes=2,
        reference_algorithm=cat_gate, input_sampler=_cls_sampler(schema_gate),
        verifier=classification_verifier,
        complexity_metadata={"depth": 2, "n_features_used": 2},
    ))

    return tasks


def _build_c3_tasks() -> List[TaskSpec]:
    """C3: Feature interaction and nonlinear classification."""
    tasks = []

    # C3.1 XOR: (x1 > 50) XOR (x2 > 50)
    schema = _cls_schema_simple(n_num=2, n_cat=0)
    tasks.append(TaskSpec(
        task_id="C3.1_xor", tier="C3", track="classification",
        description="(x1 > 50) XOR (x2 > 50) → A, else B",
        input_schema=schema, output_type="class", n_classes=2,
        reference_algorithm=lambda row: "A" if ((row["x1"] > 50.0) != (row["x2"] > 50.0)) else "B",
        input_sampler=_cls_sampler(schema), verifier=classification_verifier,
        complexity_metadata={"depth": 2, "n_features_used": 2},
    ))

    # C3.3 Rank-based: x1 is the largest among {x1, x2, x3}
    schema_rank = TabularInputSchema(numerical_features=(
        NumericalFeatureSpec(name="x1", min_val=0.0, max_val=100.0),
        NumericalFeatureSpec(name="x2", min_val=0.0, max_val=100.0),
        NumericalFeatureSpec(name="x3", min_val=0.0, max_val=100.0),
    ))
    def rank_rule(row: Dict[str, Any]) -> str:
        vals = [row["x1"], row["x2"], row["x3"]]
        mx = max(vals)
        if row["x1"] == mx:
            return "X1"
        elif row["x2"] == mx:
            return "X2"
        else:
            return "X3"
    tasks.append(TaskSpec(
        task_id="C3.3_rank_based", tier="C3", track="classification",
        description="Which of x1, x2, x3 is largest",
        input_schema=schema_rank, output_type="class", n_classes=3,
        reference_algorithm=rank_rule, input_sampler=_cls_sampler(schema_rank),
        verifier=classification_verifier,
        complexity_metadata={"depth": 2, "n_features_used": 3},
    ))

    # C3.5 Interaction polynomial: x1 * x2 > 2500
    schema_poly = _cls_schema_simple(n_num=2, n_cat=0)
    tasks.append(TaskSpec(
        task_id="C3.5_interaction_poly", tier="C3", track="classification",
        description="x1 * x2 > 2500 → A, else B",
        input_schema=schema_poly, output_type="class", n_classes=2,
        reference_algorithm=lambda row: "A" if row["x1"] * row["x2"] > 2500.0 else "B",
        input_sampler=_cls_sampler(schema_poly), verifier=classification_verifier,
        complexity_metadata={"depth": 2, "n_features_used": 2},
    ))

    return tasks


# ===================================================================
# Build the default registry with all tasks
# ===================================================================

def build_default_registry() -> TaskRegistry:
    """Build and return a TaskRegistry populated with all standard tasks."""
    registry = TaskRegistry()

    builders = [
        _build_s0_tasks,
        _build_s1_tasks,
        _build_s2_tasks,
        _build_s3_tasks,
        _build_c0_tasks,
        _build_c1_tasks,
        _build_c2_tasks,
        _build_c3_tasks,
    ]

    for builder in builders:
        for task in builder():
            registry.register(task)

    return registry
