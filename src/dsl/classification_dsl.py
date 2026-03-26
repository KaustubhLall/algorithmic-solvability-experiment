"""SR-9: Classification Rule DSL.

A small typed language for specifying classification rules programmatically.
Many classification tasks (C2–C5) are built from compositions of predicates
and combinators. This DSL lets us define them declaratively and generate
many tasks from one implementation.

Used by: C2–C5 tier tasks, SR-1 (Task Registry).
Validated by: V-9 (Classification Rule DSL Validation).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.schemas import (
    CategoricalFeatureSpec,
    Distribution,
    NumericalFeatureSpec,
    TabularInputSchema,
)


# ===================================================================
# Base types
# ===================================================================

class Predicate(ABC):
    """Base class for all predicates (return bool for a given input row)."""

    @abstractmethod
    def evaluate(self, row: Dict[str, Any]) -> bool:
        ...

    @abstractmethod
    def depth(self) -> int:
        ...

    @abstractmethod
    def features_used(self) -> set[str]:
        ...


class Classifier(ABC):
    """Base class for all classifiers (return a class label for a given input row)."""

    @abstractmethod
    def evaluate(self, row: Dict[str, Any]) -> str:
        ...

    @abstractmethod
    def depth(self) -> int:
        ...

    @abstractmethod
    def features_used(self) -> set[str]:
        ...

    @abstractmethod
    def classes(self) -> set[str]:
        ...


class Aggregator(ABC):
    """Base class for aggregators (compute a value from a group of rows).

    Aggregators are used in C4-tier tasks that require group-level computation.
    They take a list of rows and return a numeric value.
    """

    @abstractmethod
    def evaluate(self, rows: List[Dict[str, Any]]) -> float:
        ...

    @abstractmethod
    def features_used(self) -> set[str]:
        ...


# ===================================================================
# Predicates
# ===================================================================

@dataclass(frozen=True)
class Gt(Predicate):
    """feature > threshold"""
    feature: str
    threshold: float

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return float(row[self.feature]) > self.threshold

    def depth(self) -> int:
        return 1

    def features_used(self) -> set[str]:
        return {self.feature}


@dataclass(frozen=True)
class Lt(Predicate):
    """feature < threshold"""
    feature: str
    threshold: float

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return float(row[self.feature]) < self.threshold

    def depth(self) -> int:
        return 1

    def features_used(self) -> set[str]:
        return {self.feature}


@dataclass(frozen=True)
class Eq(Predicate):
    """feature == value (categorical)"""
    feature: str
    value: str

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return row[self.feature] == self.value

    def depth(self) -> int:
        return 1

    def features_used(self) -> set[str]:
        return {self.feature}


@dataclass(frozen=True)
class InSet(Predicate):
    """feature in {values}"""
    feature: str
    values: Tuple[str, ...]

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return row[self.feature] in self.values

    def depth(self) -> int:
        return 1

    def features_used(self) -> set[str]:
        return {self.feature}


@dataclass(frozen=True)
class Between(Predicate):
    """lo <= feature <= hi"""
    feature: str
    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"Between: lo ({self.lo}) > hi ({self.hi})")

    def evaluate(self, row: Dict[str, Any]) -> bool:
        val = float(row[self.feature])
        return self.lo <= val <= self.hi

    def depth(self) -> int:
        return 1

    def features_used(self) -> set[str]:
        return {self.feature}


# ===================================================================
# Combinators
# ===================================================================

@dataclass(frozen=True)
class And(Predicate):
    """Logical AND of multiple predicates."""
    operands: Tuple[Predicate, ...]

    def __post_init__(self) -> None:
        if len(self.operands) < 2:
            raise ValueError("And requires at least 2 operands")

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return all(op.evaluate(row) for op in self.operands)

    def depth(self) -> int:
        return 1 + max(op.depth() for op in self.operands)

    def features_used(self) -> set[str]:
        result: set[str] = set()
        for op in self.operands:
            result |= op.features_used()
        return result


@dataclass(frozen=True)
class Or(Predicate):
    """Logical OR of multiple predicates."""
    operands: Tuple[Predicate, ...]

    def __post_init__(self) -> None:
        if len(self.operands) < 2:
            raise ValueError("Or requires at least 2 operands")

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return any(op.evaluate(row) for op in self.operands)

    def depth(self) -> int:
        return 1 + max(op.depth() for op in self.operands)

    def features_used(self) -> set[str]:
        result: set[str] = set()
        for op in self.operands:
            result |= op.features_used()
        return result


@dataclass(frozen=True)
class Not(Predicate):
    """Logical NOT of a predicate."""
    operand: Predicate

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return not self.operand.evaluate(row)

    def depth(self) -> int:
        return 1 + self.operand.depth()

    def features_used(self) -> set[str]:
        return self.operand.features_used()


@dataclass(frozen=True)
class Xor(Predicate):
    """Logical XOR of two predicates."""
    left: Predicate
    right: Predicate

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return self.left.evaluate(row) != self.right.evaluate(row)

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def features_used(self) -> set[str]:
        return self.left.features_used() | self.right.features_used()


@dataclass(frozen=True)
class KOfN(Predicate):
    """At least k of n predicates are true."""
    k: int
    operands: Tuple[Predicate, ...]

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"KOfN: k must be >= 1, got {self.k}")
        if self.k > len(self.operands):
            raise ValueError(f"KOfN: k ({self.k}) > n ({len(self.operands)})")
        if len(self.operands) < 2:
            raise ValueError("KOfN requires at least 2 operands")

    def evaluate(self, row: Dict[str, Any]) -> bool:
        return sum(1 for op in self.operands if op.evaluate(row)) >= self.k

    def depth(self) -> int:
        return 1 + max(op.depth() for op in self.operands)

    def features_used(self) -> set[str]:
        result: set[str] = set()
        for op in self.operands:
            result |= op.features_used()
        return result


# ===================================================================
# Classifiers
# ===================================================================

@dataclass(frozen=True)
class IfThenElse(Classifier):
    """if predicate then class_if_true else class_if_false"""
    predicate: Predicate
    class_if_true: str
    class_if_false: str

    def evaluate(self, row: Dict[str, Any]) -> str:
        if self.predicate.evaluate(row):
            return self.class_if_true
        return self.class_if_false

    def depth(self) -> int:
        return 1 + self.predicate.depth()

    def features_used(self) -> set[str]:
        return self.predicate.features_used()

    def classes(self) -> set[str]:
        return {self.class_if_true, self.class_if_false}


@dataclass(frozen=True)
class DecisionList(Classifier):
    """Ordered list of (predicate, class) pairs with a default class.

    Evaluates predicates in order; returns the class of the first matching predicate.
    If none match, returns the default class.
    """
    rules: Tuple[Tuple[Predicate, str], ...]
    default_class: str

    def __post_init__(self) -> None:
        if len(self.rules) == 0:
            raise ValueError("DecisionList requires at least one rule")

    def evaluate(self, row: Dict[str, Any]) -> str:
        for pred, cls in self.rules:
            if pred.evaluate(row):
                return cls
        return self.default_class

    def depth(self) -> int:
        max_pred_depth = max(pred.depth() for pred, _ in self.rules)
        return 1 + max_pred_depth

    def features_used(self) -> set[str]:
        result: set[str] = set()
        for pred, _ in self.rules:
            result |= pred.features_used()
        return result

    def classes(self) -> set[str]:
        cls_set = {cls for _, cls in self.rules}
        cls_set.add(self.default_class)
        return cls_set


@dataclass(frozen=True)
class DecisionTreeNode:
    """A node in a decision tree classifier.

    If `predicate` is None, this is a leaf node returning `label`.
    Otherwise, branch on predicate: true → left, false → right.
    """
    predicate: Optional[Predicate] = None
    label: Optional[str] = None
    left: Optional["DecisionTreeNode"] = None
    right: Optional["DecisionTreeNode"] = None

    def __post_init__(self) -> None:
        if self.predicate is None:
            if self.label is None:
                raise ValueError("Leaf node must have a label")
        else:
            if self.left is None or self.right is None:
                raise ValueError("Branch node must have both left and right children")

    @property
    def is_leaf(self) -> bool:
        return self.predicate is None


@dataclass(frozen=True)
class DecisionTreeClassifier(Classifier):
    """Classification via a binary decision tree."""
    root: DecisionTreeNode

    def evaluate(self, row: Dict[str, Any]) -> str:
        node = self.root
        while not node.is_leaf:
            assert node.predicate is not None
            assert node.left is not None and node.right is not None
            if node.predicate.evaluate(row):
                node = node.left
            else:
                node = node.right
        assert node.label is not None
        return node.label

    def depth(self) -> int:
        return self._node_depth(self.root)

    def _node_depth(self, node: DecisionTreeNode) -> int:
        if node.is_leaf:
            return 0
        assert node.predicate is not None
        assert node.left is not None and node.right is not None
        return 1 + node.predicate.depth() + max(
            self._node_depth(node.left),
            self._node_depth(node.right),
        )

    def features_used(self) -> set[str]:
        return self._node_features(self.root)

    def _node_features(self, node: DecisionTreeNode) -> set[str]:
        if node.is_leaf:
            return set()
        assert node.predicate is not None
        assert node.left is not None and node.right is not None
        return (
            node.predicate.features_used()
            | self._node_features(node.left)
            | self._node_features(node.right)
        )

    def classes(self) -> set[str]:
        return self._node_classes(self.root)

    def _node_classes(self, node: DecisionTreeNode) -> set[str]:
        if node.is_leaf:
            assert node.label is not None
            return {node.label}
        assert node.left is not None and node.right is not None
        return self._node_classes(node.left) | self._node_classes(node.right)


# ===================================================================
# Aggregators (for C4-tier tasks)
# ===================================================================

@dataclass(frozen=True)
class MeanAggregator(Aggregator):
    """Compute mean of a numerical feature across rows."""
    feature: str

    def evaluate(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        return sum(float(r[self.feature]) for r in rows) / len(rows)

    def features_used(self) -> set[str]:
        return {self.feature}


@dataclass(frozen=True)
class CountAggregator(Aggregator):
    """Count rows where a predicate is true."""
    predicate: Predicate

    def evaluate(self, rows: List[Dict[str, Any]]) -> float:
        return float(sum(1 for r in rows if self.predicate.evaluate(r)))

    def features_used(self) -> set[str]:
        return self.predicate.features_used()


@dataclass(frozen=True)
class MaxAggregator(Aggregator):
    """Compute max of a numerical feature across rows."""
    feature: str

    def evaluate(self, rows: List[Dict[str, Any]]) -> float:
        if not rows:
            return float("-inf")
        return max(float(r[self.feature]) for r in rows)

    def features_used(self) -> set[str]:
        return {self.feature}


# ===================================================================
# Composite classifier: aggregator + classifier
# ===================================================================

@dataclass(frozen=True)
class AggregateClassifier(Classifier):
    """Classify based on an aggregated value computed from a group of rows.

    First computes the aggregate, injects it as a virtual feature, then classifies.
    Used for C4-tier tasks.
    """
    aggregator: Aggregator
    virtual_feature_name: str
    inner_classifier: Classifier

    def evaluate_group(self, rows: List[Dict[str, Any]]) -> str:
        """Classify a group of rows."""
        agg_val = self.aggregator.evaluate(rows)
        virtual_row = {self.virtual_feature_name: agg_val}
        return self.inner_classifier.evaluate(virtual_row)

    def evaluate(self, row: Dict[str, Any]) -> str:
        return self.inner_classifier.evaluate(row)

    def depth(self) -> int:
        return 1 + self.inner_classifier.depth()

    def features_used(self) -> set[str]:
        return self.aggregator.features_used() | self.inner_classifier.features_used()

    def classes(self) -> set[str]:
        return self.inner_classifier.classes()


# ===================================================================
# Rule Sampler — generate random valid classification rules
# ===================================================================

def sample_predicate(
    schema: TabularInputSchema,
    rng: np.random.Generator,
    max_depth: int = 1,
    current_depth: int = 0,
) -> Predicate:
    """Sample a random valid predicate for the given schema.

    Args:
        schema: The input schema (determines available features).
        rng: NumPy random generator.
        max_depth: Maximum predicate nesting depth.
        current_depth: Current depth in the tree (internal).

    Returns:
        A random Predicate instance.
    """
    num_specs = list(schema.numerical_features)
    cat_specs = list(schema.categorical_features)

    # At max depth or with probability, generate a leaf predicate
    if current_depth >= max_depth or (current_depth > 0 and rng.random() < 0.5):
        return _sample_leaf_predicate(num_specs, cat_specs, rng)

    # Otherwise, generate a combinator
    combinator_type = rng.choice(["and", "or", "not", "xor", "k_of_n"])

    if combinator_type == "not":
        child = sample_predicate(schema, rng, max_depth, current_depth + 1)
        return Not(operand=child)
    elif combinator_type == "xor":
        left = sample_predicate(schema, rng, max_depth, current_depth + 1)
        right = sample_predicate(schema, rng, max_depth, current_depth + 1)
        return Xor(left=left, right=right)
    else:
        n_operands = int(rng.integers(2, 4))  # 2 or 3 operands
        operands = tuple(
            sample_predicate(schema, rng, max_depth, current_depth + 1)
            for _ in range(n_operands)
        )
        if combinator_type == "and":
            return And(operands=operands)
        elif combinator_type == "or":
            return Or(operands=operands)
        else:  # k_of_n
            k = int(rng.integers(1, len(operands) + 1))
            return KOfN(k=k, operands=operands)


def _sample_leaf_predicate(
    num_specs: List[NumericalFeatureSpec],
    cat_specs: List[CategoricalFeatureSpec],
    rng: np.random.Generator,
) -> Predicate:
    """Sample an atomic (depth-1) predicate."""
    has_num = len(num_specs) > 0
    has_cat = len(cat_specs) > 0

    if has_num and has_cat:
        use_numerical = rng.random() < 0.5
    elif has_num:
        use_numerical = True
    elif has_cat:
        use_numerical = False
    else:
        raise ValueError("Schema has no features to build predicates from")

    if use_numerical:
        spec = num_specs[int(rng.integers(0, len(num_specs)))]
        pred_type = rng.choice(["gt", "lt", "between"])
        threshold = float(rng.uniform(spec.min_val, spec.max_val))
        if pred_type == "gt":
            return Gt(feature=spec.name, threshold=threshold)
        elif pred_type == "lt":
            return Lt(feature=spec.name, threshold=threshold)
        else:  # between
            t1 = float(rng.uniform(spec.min_val, spec.max_val))
            t2 = float(rng.uniform(spec.min_val, spec.max_val))
            lo, hi = min(t1, t2), max(t1, t2)
            return Between(feature=spec.name, lo=lo, hi=hi)
    else:
        spec = cat_specs[int(rng.integers(0, len(cat_specs)))]
        pred_type = rng.choice(["eq", "in_set"])
        if pred_type == "eq":
            value = spec.values[int(rng.integers(0, len(spec.values)))]
            return Eq(feature=spec.name, value=value)
        else:  # in_set
            n_vals = int(rng.integers(1, len(spec.values) + 1))
            indices = rng.choice(len(spec.values), size=n_vals, replace=False)
            values = tuple(spec.values[int(i)] for i in sorted(indices))
            return InSet(feature=spec.name, values=values)


def sample_classifier(
    schema: TabularInputSchema,
    rng: np.random.Generator,
    n_classes: int = 2,
    max_depth: int = 2,
    class_labels: Optional[List[str]] = None,
) -> Classifier:
    """Sample a random valid classifier for the given schema.

    Args:
        schema: The input schema.
        rng: NumPy random generator.
        n_classes: Number of output classes.
        max_depth: Maximum depth of the classification rule.
        class_labels: Optional class label names. Defaults to "C0", "C1", etc.

    Returns:
        A random Classifier instance.
    """
    if class_labels is None:
        class_labels = [f"C{i}" for i in range(n_classes)]
    else:
        if len(class_labels) != n_classes:
            raise ValueError(f"class_labels length ({len(class_labels)}) != n_classes ({n_classes})")

    if n_classes < 2:
        raise ValueError(f"n_classes must be >= 2, got {n_classes}")

    # Choose classifier type based on depth and n_classes
    if max_depth <= 1 or n_classes == 2:
        # Simple: if-then-else or short decision list
        clf_type = rng.choice(["if_then_else", "decision_list"])
    else:
        clf_type = rng.choice(["if_then_else", "decision_list", "decision_tree"])

    pred_depth = max(1, max_depth - 1)

    if clf_type == "if_then_else":
        pred = sample_predicate(schema, rng, max_depth=pred_depth)
        idx = rng.choice(len(class_labels), size=2, replace=False)
        return IfThenElse(
            predicate=pred,
            class_if_true=class_labels[int(idx[0])],
            class_if_false=class_labels[int(idx[1])],
        )
    elif clf_type == "decision_list":
        n_rules = min(n_classes - 1, int(rng.integers(1, max(2, n_classes))))
        shuffled = list(class_labels)
        rng.shuffle(shuffled)
        rules = []
        for i in range(n_rules):
            pred = sample_predicate(schema, rng, max_depth=pred_depth)
            rules.append((pred, shuffled[i]))
        default = shuffled[n_rules] if n_rules < len(shuffled) else shuffled[-1]
        return DecisionList(rules=tuple(rules), default_class=default)
    else:  # decision_tree
        root = _sample_tree_node(schema, rng, class_labels, pred_depth, 0, max_depth)
        return DecisionTreeClassifier(root=root)


def _sample_tree_node(
    schema: TabularInputSchema,
    rng: np.random.Generator,
    class_labels: List[str],
    pred_depth: int,
    current_tree_depth: int,
    max_tree_depth: int,
) -> DecisionTreeNode:
    """Recursively sample a decision tree node."""
    # Leaf condition
    if current_tree_depth >= max_tree_depth or rng.random() < 0.3:
        label = class_labels[int(rng.integers(0, len(class_labels)))]
        return DecisionTreeNode(label=label)

    pred = sample_predicate(schema, rng, max_depth=min(pred_depth, 1))
    left = _sample_tree_node(schema, rng, class_labels, pred_depth, current_tree_depth + 1, max_tree_depth)
    right = _sample_tree_node(schema, rng, class_labels, pred_depth, current_tree_depth + 1, max_tree_depth)
    return DecisionTreeNode(predicate=pred, left=left, right=right)


def sample_rule(
    schema: TabularInputSchema,
    seed: int,
    n_classes: int = 2,
    max_depth: int = 2,
    class_labels: Optional[List[str]] = None,
) -> Classifier:
    """Top-level API: sample a random classification rule.

    Args:
        schema: The input schema.
        seed: Random seed for reproducibility.
        n_classes: Number of output classes.
        max_depth: Maximum depth of the classification rule.
        class_labels: Optional class label names.

    Returns:
        A Classifier instance.
    """
    rng = np.random.default_rng(seed)
    return sample_classifier(schema, rng, n_classes, max_depth, class_labels)


def evaluate_rule(
    classifier: Classifier,
    row: Dict[str, Any],
) -> str:
    """Evaluate a classification rule on a single input row.

    This is the public API for applying a rule to an input.
    """
    return classifier.evaluate(row)


def verify_coverage(
    classifier: Classifier,
    schema: TabularInputSchema,
    n_samples: int = 1000,
    seed: int = 42,
) -> bool:
    """Verify that a classifier produces a valid label for every sampled input.

    Args:
        classifier: The classifier to test.
        schema: The schema to sample inputs from.
        n_samples: Number of inputs to test.
        seed: Random seed.

    Returns:
        True if every input produces a label in the classifier's class set.
    """
    valid_classes = classifier.classes()
    samples = schema.sample_batch(seed=seed, n=n_samples)
    for row in samples:
        label = classifier.evaluate(row)
        if label not in valid_classes:
            return False
    return True
