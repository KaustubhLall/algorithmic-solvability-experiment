"""SR-10: Sequence DSL.

A typed DSL for composing integer-list transformations.
Used to generate S3 and S5 tier tasks programmatically.

Primitives: map, filter, sort, reverse, unique, take, drop, sum, count, max, min,
            parity, prefix_sum, zip, concat, mod, abs, sign.

Used by: S3/S5 tier tasks, SR-1 (Task Registry).
Validated by: V-10 (Sequence DSL Validation).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np


# ===================================================================
# Type system
# ===================================================================

class SeqType(str, Enum):
    """Types in the sequence DSL."""
    LIST_INT = "list[int]"
    INT = "int"


# ===================================================================
# Base class
# ===================================================================

class SeqOp(ABC):
    """Base class for all sequence DSL operations."""

    @abstractmethod
    def evaluate(self, inp: List[int]) -> Any:
        """Apply this operation to an input integer list."""
        ...

    @abstractmethod
    def input_type(self) -> SeqType:
        """Expected input type."""
        ...

    @abstractmethod
    def output_type(self) -> SeqType:
        """Output type produced."""
        ...

    @abstractmethod
    def depth(self) -> int:
        """Nesting depth of this operation tree."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this operation."""
        ...

    def __repr__(self) -> str:
        return self.name()


# ===================================================================
# Leaf primitives: list[int] → list[int]
# ===================================================================

@dataclass(frozen=True)
class Sort(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        return sorted(inp)
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "sort"


@dataclass(frozen=True)
class Reverse(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        return list(reversed(inp))
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "reverse"


@dataclass(frozen=True)
class Unique(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        seen: set[int] = set()
        result: List[int] = []
        for x in inp:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "unique"


@dataclass(frozen=True)
class PrefixSum(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        result: List[int] = []
        running = 0
        for x in inp:
            running += x
            result.append(running)
        return result
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "prefix_sum"


@dataclass(frozen=True)
class MapAbs(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        return [abs(x) for x in inp]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "map_abs"


@dataclass(frozen=True)
class MapSign(SeqOp):
    """Map each element to its sign: -1, 0, or 1."""
    def evaluate(self, inp: List[int]) -> List[int]:
        return [(1 if x > 0 else (-1 if x < 0 else 0)) for x in inp]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "map_sign"


@dataclass(frozen=True)
class MapMod(SeqOp):
    """Map each element to element % modulus."""
    modulus: int = 2

    def __post_init__(self) -> None:
        if self.modulus < 1:
            raise ValueError(f"MapMod modulus must be >= 1, got {self.modulus}")

    def evaluate(self, inp: List[int]) -> List[int]:
        return [x % self.modulus for x in inp]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return f"map_mod({self.modulus})"


@dataclass(frozen=True)
class MapParity(SeqOp):
    """Map each element to 0 (even) or 1 (odd)."""
    def evaluate(self, inp: List[int]) -> List[int]:
        return [x % 2 for x in inp]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "map_parity"


# ===================================================================
# Parameterized list→list primitives
# ===================================================================

@dataclass(frozen=True)
class Take(SeqOp):
    """Take the first n elements."""
    n: int = 3

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"Take n must be >= 0, got {self.n}")

    def evaluate(self, inp: List[int]) -> List[int]:
        return inp[:self.n]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return f"take({self.n})"


@dataclass(frozen=True)
class Drop(SeqOp):
    """Drop the first n elements."""
    n: int = 1

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError(f"Drop n must be >= 0, got {self.n}")

    def evaluate(self, inp: List[int]) -> List[int]:
        return inp[self.n:]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return f"drop({self.n})"


@dataclass(frozen=True)
class FilterGt(SeqOp):
    """Keep elements > threshold."""
    threshold: int = 0

    def evaluate(self, inp: List[int]) -> List[int]:
        return [x for x in inp if x > self.threshold]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return f"filter_gt({self.threshold})"


@dataclass(frozen=True)
class FilterEven(SeqOp):
    """Keep even elements."""
    def evaluate(self, inp: List[int]) -> List[int]:
        return [x for x in inp if x % 2 == 0]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "filter_even"


@dataclass(frozen=True)
class FilterOdd(SeqOp):
    """Keep odd elements."""
    def evaluate(self, inp: List[int]) -> List[int]:
        return [x for x in inp if x % 2 == 1]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "filter_odd"


# ===================================================================
# Reducers: list[int] → int (wrapped as single-element list for composability)
# ===================================================================

@dataclass(frozen=True)
class Sum(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        return [sum(inp)]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "sum"


@dataclass(frozen=True)
class Count(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        return [len(inp)]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "count"


@dataclass(frozen=True)
class Max(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        if not inp:
            return []
        return [max(inp)]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "max"


@dataclass(frozen=True)
class Min(SeqOp):
    def evaluate(self, inp: List[int]) -> List[int]:
        if not inp:
            return []
        return [min(inp)]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "min"


@dataclass(frozen=True)
class Parity(SeqOp):
    """XOR parity of all elements (0 if even count of odds, 1 if odd count)."""
    def evaluate(self, inp: List[int]) -> List[int]:
        result = 0
        for x in inp:
            result ^= (x & 1)
        return [result]
    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int: return 1
    def name(self) -> str: return "parity"


# ===================================================================
# Two-input operations (take two lists, combine)
# ===================================================================

@dataclass(frozen=True)
class Concat(SeqOp):
    """Concatenate input with itself (self-concat).

    For true two-input concat in compositions, the second operand
    is the result of an inner operation on the same input.
    """
    inner: Optional[SeqOp] = None

    def evaluate(self, inp: List[int]) -> List[int]:
        if self.inner is None:
            return inp + inp
        return inp + self.inner.evaluate(inp)

    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int:
        if self.inner is None:
            return 1
        return 1 + self.inner.depth()
    def name(self) -> str:
        if self.inner is None:
            return "concat_self"
        return f"concat({self.inner.name()})"


@dataclass(frozen=True)
class ZipAdd(SeqOp):
    """Element-wise addition of input with result of inner op.

    zip(input, inner(input)) → [a+b for a,b in pairs], truncated to shorter length.
    If no inner, zips with self (doubles each element).
    """
    inner: Optional[SeqOp] = None

    def evaluate(self, inp: List[int]) -> List[int]:
        if self.inner is None:
            return [x + x for x in inp]
        other = self.inner.evaluate(inp)
        min_len = min(len(inp), len(other))
        return [inp[i] + other[i] for i in range(min_len)]

    def input_type(self) -> SeqType: return SeqType.LIST_INT
    def output_type(self) -> SeqType: return SeqType.LIST_INT
    def depth(self) -> int:
        if self.inner is None:
            return 1
        return 1 + self.inner.depth()
    def name(self) -> str:
        if self.inner is None:
            return "zip_add_self"
        return f"zip_add({self.inner.name()})"


# ===================================================================
# Composition: pipe operations sequentially
# ===================================================================

@dataclass(frozen=True)
class Compose(SeqOp):
    """Sequential composition: apply first, then second to its result.

    Compose(first=Sort(), second=Reverse()) means: sort, then reverse.
    """
    first: SeqOp
    second: SeqOp

    def __post_init__(self) -> None:
        if self.first.output_type() != self.second.input_type():
            raise TypeError(
                f"Type mismatch in Compose: {self.first.name()} outputs "
                f"{self.first.output_type().value} but {self.second.name()} expects "
                f"{self.second.input_type().value}"
            )

    def evaluate(self, inp: List[int]) -> Any:
        intermediate = self.first.evaluate(inp)
        return self.second.evaluate(intermediate)

    def input_type(self) -> SeqType: return self.first.input_type()
    def output_type(self) -> SeqType: return self.second.output_type()

    def depth(self) -> int:
        return self.first.depth() + self.second.depth()

    def name(self) -> str:
        return f"{self.first.name()} | {self.second.name()}"


# ===================================================================
# Program: a named, composable sequence transformation
# ===================================================================

@dataclass(frozen=True)
class SeqProgram:
    """A named sequence program wrapping a SeqOp tree.

    Attributes:
        program_id: Unique identifier.
        op: The root operation.
        description: Human-readable description.
    """
    program_id: str
    op: SeqOp
    description: str = ""

    def evaluate(self, inp: List[int]) -> Any:
        return self.op.evaluate(inp)

    def depth(self) -> int:
        return self.op.depth()

    def name(self) -> str:
        return self.op.name()

    def input_type(self) -> SeqType:
        return self.op.input_type()

    def output_type(self) -> SeqType:
        return self.op.output_type()


# ===================================================================
# All leaf operations (no parameters or with defaults)
# ===================================================================

LEAF_OPS_LIST_TO_LIST: List[type] = [
    Sort, Reverse, Unique, PrefixSum, MapAbs, MapSign, MapParity,
]

PARAMETERIZED_OPS: List[type] = [
    Take, Drop, FilterGt, FilterEven, FilterOdd, MapMod,
]

REDUCER_OPS: List[type] = [
    Sum, Count, Max, Min, Parity,
]

ALL_LEAF_TYPES = LEAF_OPS_LIST_TO_LIST + PARAMETERIZED_OPS + REDUCER_OPS


# ===================================================================
# Program Sampler
# ===================================================================

def _sample_leaf_op(rng: np.random.Generator, allow_reducers: bool = True) -> SeqOp:
    """Sample a random leaf operation."""
    if allow_reducers:
        pool = LEAF_OPS_LIST_TO_LIST + PARAMETERIZED_OPS + REDUCER_OPS
    else:
        pool = LEAF_OPS_LIST_TO_LIST + PARAMETERIZED_OPS

    op_cls = pool[int(rng.integers(0, len(pool)))]

    # Handle parameterized ops
    if op_cls is Take:
        return Take(n=int(rng.integers(1, 6)))
    elif op_cls is Drop:
        return Drop(n=int(rng.integers(1, 4)))
    elif op_cls is FilterGt:
        return FilterGt(threshold=int(rng.integers(0, 10)))
    elif op_cls is MapMod:
        return MapMod(modulus=int(rng.integers(2, 6)))
    else:
        return op_cls()


def sample_program(
    seed: int,
    max_depth: int = 2,
    program_id: Optional[str] = None,
) -> SeqProgram:
    """Sample a random sequence program.

    Args:
        seed: Random seed for reproducibility.
        max_depth: Maximum composition depth (1 = single op, 2 = two ops composed, etc.).
        program_id: Optional ID. Defaults to "prog_{seed}_d{max_depth}".

    Returns:
        A SeqProgram instance.
    """
    rng = np.random.default_rng(seed)
    if program_id is None:
        program_id = f"prog_{seed}_d{max_depth}"

    op = _sample_op_tree(rng, max_depth, current_depth=0)
    return SeqProgram(program_id=program_id, op=op, description=op.name())


def _sample_op_tree(
    rng: np.random.Generator,
    max_depth: int,
    current_depth: int,
) -> SeqOp:
    """Recursively sample an operation tree."""
    remaining = max_depth - current_depth

    if remaining <= 1:
        # Must return a leaf (reducers OK here since output is still list[int])
        return _sample_leaf_op(rng, allow_reducers=True)

    # Decide: leaf, compose, or two-input op
    choice = rng.choice(["leaf", "compose", "zip", "concat"], p=[0.2, 0.5, 0.15, 0.15])

    if choice == "leaf":
        return _sample_leaf_op(rng, allow_reducers=True)
    elif choice == "compose":
        # First op should not be a reducer (would make a 1-element list for second)
        first = _sample_leaf_op(rng, allow_reducers=False)
        second = _sample_op_tree(rng, max_depth, current_depth + 1)
        return Compose(first=first, second=second)
    elif choice == "zip":
        inner = _sample_op_tree(rng, max_depth, current_depth + 1)
        return ZipAdd(inner=inner)
    else:  # concat
        inner = _sample_op_tree(rng, max_depth, current_depth + 1)
        return Concat(inner=inner)


def sample_programs_batch(
    n: int,
    seed: int,
    max_depth: int = 2,
) -> List[SeqProgram]:
    """Sample n distinct programs.

    Args:
        n: Number of programs to generate.
        seed: Base seed.
        max_depth: Maximum composition depth.

    Returns:
        List of SeqProgram instances.
    """
    programs = []
    for i in range(n):
        prog = sample_program(
            seed=seed + i,
            max_depth=max_depth,
            program_id=f"prog_{i}_d{max_depth}",
        )
        programs.append(prog)
    return programs


def check_functional_equivalence(
    prog_a: SeqProgram,
    prog_b: SeqProgram,
    n_test_inputs: int = 1000,
    seed: int = 42,
    value_range: Tuple[int, int] = (-9, 9),
    length_range: Tuple[int, int] = (3, 10),
) -> bool:
    """Check if two programs are functionally equivalent on random inputs.

    Args:
        prog_a: First program.
        prog_b: Second program.
        n_test_inputs: Number of random inputs to test.
        seed: Random seed.
        value_range: Range for element values.
        length_range: Range for sequence lengths.

    Returns:
        True if both programs produce identical outputs on all test inputs.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_test_inputs):
        length = int(rng.integers(length_range[0], length_range[1] + 1))
        inp = [int(x) for x in rng.integers(value_range[0], value_range[1] + 1, size=length)]
        try:
            out_a = prog_a.evaluate(inp)
            out_b = prog_b.evaluate(inp)
            if out_a != out_b:
                return False
        except Exception:
            return False
    return True
