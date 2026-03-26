"""V-9: Classification Rule DSL Validation Tests.

Tests for SR-9 (Classification Rule DSL) as specified in EXPERIMENT_CATALOG.md Part 3.

Checks:
1. Type safety — DSL programs that violate type constraints are rejected at construction time.
2. Determinism — applying a rule to the same input always produces the same output.
3. Coverage — for any input conforming to the schema, the rule produces a valid class label.
4. Known-rule test — 5 hand-written rules, 1000 samples each, verify all labels match.
5. Depth correctness — reported depth matches actual nesting depth.
6. Equivalence check — semantically equivalent rules produce the same labels.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from src.dsl.classification_dsl import (
    AggregateClassifier,
    And,
    Between,
    CountAggregator,
    DecisionList,
    DecisionTreeClassifier,
    DecisionTreeNode,
    Eq,
    Gt,
    IfThenElse,
    InSet,
    KOfN,
    Lt,
    MaxAggregator,
    MeanAggregator,
    Not,
    Or,
    Xor,
    evaluate_rule,
    sample_predicate,
    sample_rule,
    verify_coverage,
)
from src.schemas import (
    CategoricalFeatureSpec,
    Distribution,
    NumericalFeatureSpec,
    TabularInputSchema,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def simple_schema() -> TabularInputSchema:
    return TabularInputSchema(
        numerical_features=(
            NumericalFeatureSpec(name="x", min_val=0.0, max_val=100.0),
            NumericalFeatureSpec(name="y", min_val=-50.0, max_val=50.0),
        ),
        categorical_features=(
            CategoricalFeatureSpec(name="color", values=("red", "green", "blue")),
        ),
    )


@pytest.fixture
def numeric_only_schema() -> TabularInputSchema:
    return TabularInputSchema(
        numerical_features=(
            NumericalFeatureSpec(name="a", min_val=0.0, max_val=10.0),
            NumericalFeatureSpec(name="b", min_val=0.0, max_val=10.0),
        ),
    )


@pytest.fixture
def sample_row() -> Dict[str, Any]:
    return {"x": 75.0, "y": 10.0, "color": "red"}


# ===================================================================
# 1. Type Safety — invalid constructions rejected
# ===================================================================

class TestTypeSafety:

    def test_between_lo_gt_hi_raises(self):
        with pytest.raises(ValueError, match="lo.*hi"):
            Between(feature="x", lo=10.0, hi=5.0)

    def test_and_single_operand_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            And(operands=(Gt(feature="x", threshold=5.0),))

    def test_or_single_operand_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            Or(operands=(Gt(feature="x", threshold=5.0),))

    def test_kofn_k_zero_raises(self):
        ops = (Gt(feature="x", threshold=5.0), Lt(feature="y", threshold=10.0))
        with pytest.raises(ValueError, match="k must be >= 1"):
            KOfN(k=0, operands=ops)

    def test_kofn_k_gt_n_raises(self):
        ops = (Gt(feature="x", threshold=5.0), Lt(feature="y", threshold=10.0))
        with pytest.raises(ValueError, match="k.*>.*n"):
            KOfN(k=5, operands=ops)

    def test_kofn_single_operand_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            KOfN(k=1, operands=(Gt(feature="x", threshold=5.0),))

    def test_decision_list_empty_raises(self):
        with pytest.raises(ValueError, match="at least one rule"):
            DecisionList(rules=(), default_class="A")

    def test_decision_tree_leaf_no_label_raises(self):
        with pytest.raises(ValueError, match="Leaf node must have a label"):
            DecisionTreeNode(predicate=None, label=None)

    def test_decision_tree_branch_no_children_raises(self):
        with pytest.raises(ValueError, match="Branch node must have both"):
            DecisionTreeNode(predicate=Gt(feature="x", threshold=5.0), left=None, right=None)


# ===================================================================
# 2. Determinism — same input → same output, always
# ===================================================================

class TestDeterminism:
    N_TRIALS = 200

    def test_predicate_determinism(self, sample_row: Dict[str, Any]):
        pred = And(operands=(Gt(feature="x", threshold=50.0), Lt(feature="y", threshold=20.0)))
        results = [pred.evaluate(sample_row) for _ in range(self.N_TRIALS)]
        assert all(r == results[0] for r in results)

    def test_classifier_determinism(self, sample_row: Dict[str, Any]):
        clf = IfThenElse(
            predicate=Gt(feature="x", threshold=50.0),
            class_if_true="A",
            class_if_false="B",
        )
        results = [clf.evaluate(sample_row) for _ in range(self.N_TRIALS)]
        assert all(r == results[0] for r in results)

    def test_decision_list_determinism(self, sample_row: Dict[str, Any]):
        clf = DecisionList(
            rules=(
                (Gt(feature="x", threshold=80.0), "HIGH"),
                (Lt(feature="x", threshold=20.0), "LOW"),
            ),
            default_class="MID",
        )
        results = [clf.evaluate(sample_row) for _ in range(self.N_TRIALS)]
        assert all(r == results[0] for r in results)

    def test_sampled_rule_determinism(self, simple_schema: TabularInputSchema):
        """Same seed → same rule → same predictions."""
        rule1 = sample_rule(simple_schema, seed=42, n_classes=3, max_depth=3)
        rule2 = sample_rule(simple_schema, seed=42, n_classes=3, max_depth=3)
        rows = simple_schema.sample_batch(seed=99, n=100)
        for row in rows:
            assert rule1.evaluate(row) == rule2.evaluate(row)


# ===================================================================
# 3. Coverage — every input gets a valid label
# ===================================================================

class TestCoverage:
    N_SAMPLES = 1000

    def test_if_then_else_coverage(self, simple_schema: TabularInputSchema):
        clf = IfThenElse(
            predicate=Gt(feature="x", threshold=50.0),
            class_if_true="A",
            class_if_false="B",
        )
        assert verify_coverage(clf, simple_schema, n_samples=self.N_SAMPLES)

    def test_decision_list_coverage(self, simple_schema: TabularInputSchema):
        clf = DecisionList(
            rules=(
                (Gt(feature="x", threshold=80.0), "HIGH"),
                (Between(feature="x", lo=40.0, hi=80.0), "MID"),
            ),
            default_class="LOW",
        )
        assert verify_coverage(clf, simple_schema, n_samples=self.N_SAMPLES)

    def test_decision_tree_coverage(self, simple_schema: TabularInputSchema):
        root = DecisionTreeNode(
            predicate=Gt(feature="x", threshold=50.0),
            left=DecisionTreeNode(label="A"),
            right=DecisionTreeNode(
                predicate=Eq(feature="color", value="red"),
                left=DecisionTreeNode(label="B"),
                right=DecisionTreeNode(label="C"),
            ),
        )
        clf = DecisionTreeClassifier(root=root)
        assert verify_coverage(clf, simple_schema, n_samples=self.N_SAMPLES)

    def test_sampled_rule_coverage(self, simple_schema: TabularInputSchema):
        for seed in range(20):
            clf = sample_rule(simple_schema, seed=seed, n_classes=3, max_depth=3)
            assert verify_coverage(clf, simple_schema, n_samples=500), f"Coverage failed for seed={seed}"


# ===================================================================
# 4. Known-Rule Tests — 5 hand-written rules with verified outputs
# ===================================================================

class TestKnownRules:
    """Hand-written rules with known correct outputs for specific inputs."""

    def test_rule1_simple_threshold(self):
        """Rule: x > 50 → A, else → B"""
        clf = IfThenElse(predicate=Gt(feature="x", threshold=50.0), class_if_true="A", class_if_false="B")
        assert clf.evaluate({"x": 75.0}) == "A"
        assert clf.evaluate({"x": 25.0}) == "B"
        assert clf.evaluate({"x": 50.0}) == "B"  # not strictly greater
        assert clf.evaluate({"x": 51.0}) == "A"

    def test_rule2_categorical_match(self):
        """Rule: color == 'red' → DANGER, else → SAFE"""
        clf = IfThenElse(predicate=Eq(feature="color", value="red"), class_if_true="DANGER", class_if_false="SAFE")
        assert clf.evaluate({"color": "red"}) == "DANGER"
        assert clf.evaluate({"color": "blue"}) == "SAFE"
        assert clf.evaluate({"color": "green"}) == "SAFE"

    def test_rule3_and_combination(self):
        """Rule: (x > 50 AND y < 0) → A, else → B"""
        pred = And(operands=(Gt(feature="x", threshold=50.0), Lt(feature="y", threshold=0.0)))
        clf = IfThenElse(predicate=pred, class_if_true="A", class_if_false="B")
        assert clf.evaluate({"x": 75.0, "y": -10.0}) == "A"  # both true
        assert clf.evaluate({"x": 75.0, "y": 10.0}) == "B"   # x true, y false
        assert clf.evaluate({"x": 25.0, "y": -10.0}) == "B"  # x false, y true
        assert clf.evaluate({"x": 25.0, "y": 10.0}) == "B"   # both false

    def test_rule4_decision_list(self):
        """Rule: x > 80 → HIGH, x < 20 → LOW, else → MID"""
        clf = DecisionList(
            rules=(
                (Gt(feature="x", threshold=80.0), "HIGH"),
                (Lt(feature="x", threshold=20.0), "LOW"),
            ),
            default_class="MID",
        )
        assert clf.evaluate({"x": 90.0}) == "HIGH"
        assert clf.evaluate({"x": 10.0}) == "LOW"
        assert clf.evaluate({"x": 50.0}) == "MID"
        assert clf.evaluate({"x": 80.0}) == "MID"   # not strictly >80
        assert clf.evaluate({"x": 20.0}) == "MID"   # not strictly <20

    def test_rule5_xor_with_not(self):
        """Rule: (x > 50) XOR (y > 0) → A, else → B"""
        pred = Xor(
            left=Gt(feature="x", threshold=50.0),
            right=Gt(feature="y", threshold=0.0),
        )
        clf = IfThenElse(predicate=pred, class_if_true="A", class_if_false="B")
        assert clf.evaluate({"x": 75.0, "y": -10.0}) == "A"  # T XOR F = T
        assert clf.evaluate({"x": 25.0, "y": 10.0}) == "A"   # F XOR T = T
        assert clf.evaluate({"x": 75.0, "y": 10.0}) == "B"   # T XOR T = F
        assert clf.evaluate({"x": 25.0, "y": -10.0}) == "B"  # F XOR F = F

    def test_rule_known_k_of_n(self):
        """Rule: at least 2 of 3 predicates true → A, else → B"""
        pred = KOfN(
            k=2,
            operands=(
                Gt(feature="x", threshold=50.0),
                Lt(feature="y", threshold=0.0),
                Eq(feature="color", value="red"),
            ),
        )
        clf = IfThenElse(predicate=pred, class_if_true="A", class_if_false="B")
        # All 3 true
        assert clf.evaluate({"x": 75.0, "y": -10.0, "color": "red"}) == "A"
        # 2 of 3 true
        assert clf.evaluate({"x": 75.0, "y": -10.0, "color": "blue"}) == "A"
        # 1 of 3 true
        assert clf.evaluate({"x": 75.0, "y": 10.0, "color": "blue"}) == "B"
        # 0 of 3 true
        assert clf.evaluate({"x": 25.0, "y": 10.0, "color": "blue"}) == "B"

    def test_rule_known_in_set(self):
        """Rule: color in {red, blue} → WARM, else → COOL"""
        clf = IfThenElse(
            predicate=InSet(feature="color", values=("red", "blue")),
            class_if_true="WARM",
            class_if_false="COOL",
        )
        assert clf.evaluate({"color": "red"}) == "WARM"
        assert clf.evaluate({"color": "blue"}) == "WARM"
        assert clf.evaluate({"color": "green"}) == "COOL"

    def test_rule_known_between(self):
        """Rule: 20 <= x <= 80 → IN_RANGE, else → OUT"""
        clf = IfThenElse(
            predicate=Between(feature="x", lo=20.0, hi=80.0),
            class_if_true="IN_RANGE",
            class_if_false="OUT",
        )
        assert clf.evaluate({"x": 50.0}) == "IN_RANGE"
        assert clf.evaluate({"x": 20.0}) == "IN_RANGE"  # inclusive
        assert clf.evaluate({"x": 80.0}) == "IN_RANGE"  # inclusive
        assert clf.evaluate({"x": 10.0}) == "OUT"
        assert clf.evaluate({"x": 90.0}) == "OUT"

    def test_known_rule_bulk_verification(self, simple_schema: TabularInputSchema):
        """Verify 5 hand-written rules on 1000 samples each via reference logic."""
        rules_and_refs = [
            (
                IfThenElse(predicate=Gt(feature="x", threshold=50.0), class_if_true="A", class_if_false="B"),
                lambda r: "A" if r["x"] > 50.0 else "B",
            ),
            (
                IfThenElse(predicate=Eq(feature="color", value="red"), class_if_true="YES", class_if_false="NO"),
                lambda r: "YES" if r["color"] == "red" else "NO",
            ),
            (
                IfThenElse(
                    predicate=And(operands=(Gt(feature="x", threshold=50.0), Lt(feature="y", threshold=0.0))),
                    class_if_true="A",
                    class_if_false="B",
                ),
                lambda r: "A" if (r["x"] > 50.0 and r["y"] < 0.0) else "B",
            ),
            (
                DecisionList(
                    rules=(
                        (Gt(feature="x", threshold=80.0), "HIGH"),
                        (Lt(feature="x", threshold=20.0), "LOW"),
                    ),
                    default_class="MID",
                ),
                lambda r: "HIGH" if r["x"] > 80.0 else ("LOW" if r["x"] < 20.0 else "MID"),
            ),
            (
                IfThenElse(
                    predicate=Or(operands=(
                        Gt(feature="x", threshold=90.0),
                        Eq(feature="color", value="blue"),
                    )),
                    class_if_true="MATCH",
                    class_if_false="NO_MATCH",
                ),
                lambda r: "MATCH" if (r["x"] > 90.0 or r["color"] == "blue") else "NO_MATCH",
            ),
        ]

        samples = simple_schema.sample_batch(seed=42, n=1000)
        for rule_idx, (clf, ref_fn) in enumerate(rules_and_refs):
            for i, row in enumerate(samples):
                dsl_label = clf.evaluate(row)
                ref_label = ref_fn(row)
                assert dsl_label == ref_label, (
                    f"Rule {rule_idx} mismatch at sample {i}: "
                    f"DSL={dsl_label}, ref={ref_label}, row={row}"
                )


# ===================================================================
# 5. Depth Correctness
# ===================================================================

class TestDepthCorrectness:

    def test_leaf_predicate_depth(self):
        assert Gt(feature="x", threshold=5.0).depth() == 1
        assert Lt(feature="x", threshold=5.0).depth() == 1
        assert Eq(feature="c", value="a").depth() == 1
        assert InSet(feature="c", values=("a", "b")).depth() == 1
        assert Between(feature="x", lo=1.0, hi=5.0).depth() == 1

    def test_not_depth(self):
        inner = Gt(feature="x", threshold=5.0)
        assert Not(operand=inner).depth() == 2

    def test_and_depth(self):
        a = Gt(feature="x", threshold=5.0)
        b = Lt(feature="y", threshold=10.0)
        assert And(operands=(a, b)).depth() == 2

    def test_nested_depth(self):
        # And(Not(Gt), Lt) → depth = 1 + max(Not(Gt).depth=2, Lt.depth=1) = 3
        inner = Not(operand=Gt(feature="x", threshold=5.0))
        outer = And(operands=(inner, Lt(feature="y", threshold=10.0)))
        assert outer.depth() == 3

    def test_if_then_else_depth(self):
        pred = Gt(feature="x", threshold=5.0)
        clf = IfThenElse(predicate=pred, class_if_true="A", class_if_false="B")
        assert clf.depth() == 2  # 1 (ITE) + 1 (pred)

    def test_decision_list_depth(self):
        rules = (
            (Gt(feature="x", threshold=5.0), "A"),
            (And(operands=(Lt(feature="x", threshold=3.0), Lt(feature="y", threshold=0.0))), "B"),
        )
        clf = DecisionList(rules=rules, default_class="C")
        # Max pred depth is And(Lt, Lt) = 2, so decision list depth = 1 + 2 = 3
        assert clf.depth() == 3

    def test_decision_tree_depth(self):
        root = DecisionTreeNode(
            predicate=Gt(feature="x", threshold=50.0),
            left=DecisionTreeNode(label="A"),
            right=DecisionTreeNode(
                predicate=Lt(feature="y", threshold=0.0),
                left=DecisionTreeNode(label="B"),
                right=DecisionTreeNode(label="C"),
            ),
        )
        clf = DecisionTreeClassifier(root=root)
        # Root: 1 + Gt.depth(1) + max(left_depth=0, right_depth=1+Lt.depth(1)+max(0,0)=2) = 1+1+2 = 4
        assert clf.depth() == 4

    def test_xor_depth(self):
        pred = Xor(left=Gt(feature="x", threshold=5.0), right=Lt(feature="y", threshold=3.0))
        assert pred.depth() == 2  # 1 + max(1, 1)

    def test_kofn_depth(self):
        ops = (Gt(feature="x", threshold=5.0), Lt(feature="y", threshold=10.0))
        pred = KOfN(k=1, operands=ops)
        assert pred.depth() == 2  # 1 + max(1, 1)


# ===================================================================
# 6. Equivalence Check
# ===================================================================

class TestEquivalence:

    def test_double_not_equivalent(self, simple_schema: TabularInputSchema):
        """NOT(NOT(x > 50)) should produce same results as x > 50."""
        pred_simple = Gt(feature="x", threshold=50.0)
        pred_double_not = Not(operand=Not(operand=Gt(feature="x", threshold=50.0)))

        samples = simple_schema.sample_batch(seed=42, n=1000)
        for row in samples:
            assert pred_simple.evaluate(row) == pred_double_not.evaluate(row)

    def test_demorgan_and_or(self, simple_schema: TabularInputSchema):
        """NOT(A AND B) should equal (NOT A) OR (NOT B)."""
        a = Gt(feature="x", threshold=50.0)
        b = Lt(feature="y", threshold=0.0)

        lhs = Not(operand=And(operands=(a, b)))
        rhs = Or(operands=(Not(operand=Gt(feature="x", threshold=50.0)), Not(operand=Lt(feature="y", threshold=0.0))))

        samples = simple_schema.sample_batch(seed=42, n=1000)
        for row in samples:
            assert lhs.evaluate(row) == rhs.evaluate(row)

    def test_equivalent_classifiers_same_labels(self, simple_schema: TabularInputSchema):
        """Two syntactically different classifiers that are semantically equivalent."""
        # ITE(x > 50, A, B) vs DecisionList([(x > 50, A)], default=B)
        clf1 = IfThenElse(predicate=Gt(feature="x", threshold=50.0), class_if_true="A", class_if_false="B")
        clf2 = DecisionList(rules=((Gt(feature="x", threshold=50.0), "A"),), default_class="B")

        samples = simple_schema.sample_batch(seed=42, n=1000)
        for row in samples:
            assert clf1.evaluate(row) == clf2.evaluate(row)


# ===================================================================
# 7. Features Used
# ===================================================================

class TestFeaturesUsed:

    def test_simple_predicate_features(self):
        assert Gt(feature="x", threshold=5.0).features_used() == {"x"}
        assert Eq(feature="color", value="red").features_used() == {"color"}

    def test_combinator_features(self):
        pred = And(operands=(Gt(feature="x", threshold=5.0), Lt(feature="y", threshold=10.0)))
        assert pred.features_used() == {"x", "y"}

    def test_classifier_features(self):
        clf = IfThenElse(
            predicate=And(operands=(Gt(feature="x", threshold=5.0), Eq(feature="color", value="red"))),
            class_if_true="A",
            class_if_false="B",
        )
        assert clf.features_used() == {"x", "color"}

    def test_decision_tree_features(self):
        root = DecisionTreeNode(
            predicate=Gt(feature="x", threshold=50.0),
            left=DecisionTreeNode(label="A"),
            right=DecisionTreeNode(
                predicate=Eq(feature="color", value="red"),
                left=DecisionTreeNode(label="B"),
                right=DecisionTreeNode(label="C"),
            ),
        )
        clf = DecisionTreeClassifier(root=root)
        assert clf.features_used() == {"x", "color"}


# ===================================================================
# 8. Aggregator Tests
# ===================================================================

class TestAggregators:

    def test_mean_aggregator(self):
        rows = [{"x": 10.0}, {"x": 20.0}, {"x": 30.0}]
        agg = MeanAggregator(feature="x")
        assert agg.evaluate(rows) == 20.0

    def test_mean_aggregator_empty(self):
        agg = MeanAggregator(feature="x")
        assert agg.evaluate([]) == 0.0

    def test_count_aggregator(self):
        rows = [{"x": 10.0}, {"x": 60.0}, {"x": 80.0}]
        agg = CountAggregator(predicate=Gt(feature="x", threshold=50.0))
        assert agg.evaluate(rows) == 2.0

    def test_max_aggregator(self):
        rows = [{"x": 10.0}, {"x": 60.0}, {"x": 30.0}]
        agg = MaxAggregator(feature="x")
        assert agg.evaluate(rows) == 60.0

    def test_aggregator_features_used(self):
        assert MeanAggregator(feature="x").features_used() == {"x"}
        assert CountAggregator(predicate=Gt(feature="x", threshold=5.0)).features_used() == {"x"}
        assert MaxAggregator(feature="x").features_used() == {"x"}

    def test_aggregate_classifier_requires_group_evaluation(self):
        clf = AggregateClassifier(
            aggregator=MeanAggregator(feature="x"),
            virtual_feature_name="mean_x",
            inner_classifier=IfThenElse(
                predicate=Gt(feature="mean_x", threshold=10.0),
                class_if_true="HIGH",
                class_if_false="LOW",
            ),
        )
        assert clf.evaluate_group([{"x": 5.0}, {"x": 15.0}]) == "LOW"
        with pytest.raises(RuntimeError, match="evaluate_group"):
            clf.evaluate({"mean_x": 10.0})


# ===================================================================
# 9. Rule Sampler Tests
# ===================================================================

class TestRuleSampler:

    def test_sample_rule_reproducible(self, simple_schema: TabularInputSchema):
        r1 = sample_rule(simple_schema, seed=42, n_classes=3, max_depth=3)
        r2 = sample_rule(simple_schema, seed=42, n_classes=3, max_depth=3)
        rows = simple_schema.sample_batch(seed=99, n=100)
        for row in rows:
            assert r1.evaluate(row) == r2.evaluate(row)

    def test_sample_rule_different_seeds_usually_change_structure(self, simple_schema: TabularInputSchema):
        rules = [sample_rule(simple_schema, seed=seed, n_classes=3, max_depth=3) for seed in range(10)]
        signatures = {
            (type(rule).__name__, rule.depth(), tuple(sorted(rule.features_used())), tuple(sorted(rule.classes())))
            for rule in rules
        }
        assert len(signatures) >= 2

    def test_sample_rule_classes(self, simple_schema: TabularInputSchema):
        for seed in range(10):
            clf = sample_rule(simple_schema, seed=seed, n_classes=4, max_depth=2,
                              class_labels=["A", "B", "C", "D"])
            # All returned classes should be in the label set
            assert clf.classes().issubset({"A", "B", "C", "D"})

    def test_sample_rule_custom_labels(self, simple_schema: TabularInputSchema):
        clf = sample_rule(simple_schema, seed=42, n_classes=2, class_labels=["POS", "NEG"])
        rows = simple_schema.sample_batch(seed=99, n=100)
        for row in rows:
            assert clf.evaluate(row) in {"POS", "NEG"}

    def test_sample_rule_invalid_n_classes_raises(self, simple_schema: TabularInputSchema):
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            sample_rule(simple_schema, seed=42, n_classes=1)

    def test_sample_rule_label_length_mismatch_raises(self, simple_schema: TabularInputSchema):
        with pytest.raises(ValueError, match="class_labels length"):
            sample_rule(simple_schema, seed=42, n_classes=3, class_labels=["A", "B"])

    def test_sample_many_rules_no_crash(self, simple_schema: TabularInputSchema):
        """Generate 50 random rules and verify they all work."""
        for seed in range(50):
            for depth in [1, 2, 3, 4]:
                if depth == 1:
                    with pytest.raises(ValueError, match="max_depth must be >= 2"):
                        sample_rule(simple_schema, seed=seed, n_classes=3, max_depth=depth)
                    continue
                clf = sample_rule(simple_schema, seed=seed, n_classes=3, max_depth=depth)
                assert clf.depth() <= depth
                rows = simple_schema.sample_batch(seed=seed + 1000, n=50)
                for row in rows:
                    label = clf.evaluate(row)
                    assert label in clf.classes()

    def test_sample_rule_numeric_only_schema(self, numeric_only_schema: TabularInputSchema):
        """Sampling works with numeric-only schemas."""
        for seed in range(10):
            clf = sample_rule(numeric_only_schema, seed=seed, n_classes=2, max_depth=2)
            assert verify_coverage(clf, numeric_only_schema, n_samples=100, seed=seed)

    def test_sample_predicate_respects_max_depth(self, simple_schema: TabularInputSchema):
        rng = np.random.default_rng(42)
        for max_depth in [1, 2, 3, 4]:
            pred = sample_predicate(simple_schema, rng, max_depth=max_depth)
            assert pred.depth() <= max_depth

    def test_sample_rule_respects_allowed_features(self, simple_schema: TabularInputSchema):
        clf = sample_rule(
            simple_schema,
            seed=42,
            n_classes=2,
            max_depth=3,
            allowed_features=["x"],
        )
        assert clf.features_used() == {"x"}
