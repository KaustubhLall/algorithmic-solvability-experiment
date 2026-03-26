# TASK-02: Classification Rule DSL

> **Status:** COMPLETE ✓
> **Builds:** SR-9
> **Validation:** V-9
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-9), Part 3 (V-9), Part 4 (TASK-02)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| Predicates: `Gt`, `Lt`, `Eq`, `InSet`, `Between` | `src/dsl/classification_dsl.py` | DONE |
| Combinators: `And`, `Or`, `Not`, `Xor`, `KOfN` | `src/dsl/classification_dsl.py` | DONE |
| Classifiers: `IfThenElse`, `DecisionList`, `DecisionTreeClassifier` | `src/dsl/classification_dsl.py` | DONE |
| Aggregators: `MeanAggregator`, `CountAggregator`, `MaxAggregator` | `src/dsl/classification_dsl.py` | DONE |
| Composite: `AggregateClassifier` | `src/dsl/classification_dsl.py` | DONE |
| Rule sampling: `sample_predicate`, `sample_classifier`, `sample_rule` | `src/dsl/classification_dsl.py` | DONE |
| Evaluation: `evaluate_rule`, `verify_coverage` | `src/dsl/classification_dsl.py` | DONE |
| V-9 test suite | `tests/test_classification_dsl.py` | DONE (55 tests) |

---

## Implementation Notes

### Design Choices
- **ABC base classes** — `Predicate`, `Classifier`, and `Aggregator` are abstract base classes with `evaluate()`, `depth()`, `features_used()`, and `name()` methods.
- **Frozen dataclasses** — All concrete predicate, combinator, classifier, and aggregator classes use `@dataclass(frozen=True)` for immutability, consistent with ADR-006.
- **`__post_init__` validation** — `And`/`Or` require ≥2 operands, `KOfN` requires 1 ≤ k ≤ n and ≥2 operands, `Between` requires lo ≤ hi, `DecisionList` requires non-empty rules, `DecisionTreeNode` validates structure.
- **`features_used()` recursion** — Predicates return their single feature, combinators union their children's features, classifiers union across all branches. This enables downstream feature-use tracking.
- **Rule sampler depth control** — `sample_rule()` uses `max_depth` to limit predicate/classifier nesting. At depth 1, only leaf predicates are created. Higher depths compose with `And`/`Or`/`Not`/`Xor`/`KOfN`.
- **`evaluate_rule()` wraps classifier evaluation** — Takes a `Classifier` and a row dict, returns the predicted label.
- **`verify_coverage()` checks all labels appear** — Given a classifier and test rows, verifies every declared label is produced at least once.

### Edge Cases Handled
- `KOfN` with k=1 (equivalent to OR) and k=n (equivalent to AND) both work correctly.
- `DecisionList` with a single rule (trivial) works correctly.
- `Not` of `Not` is functionally equivalent to the original (tested via De Morgan's law).
- Empty feature sets for aggregators that operate on classifier outputs.

---

## Acceptance Criteria Results

V-9 validation: **55 tests, all passing** (pytest, 0.28s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| Type safety | Invalid configs rejected | 9 tests pass | YES |
| Determinism | Same input → same output | 4 tests pass | YES |
| Coverage | All labels produced | 4 tests pass | YES |
| Known rules | 5+ hand-verified rules | 9 tests pass (8 specific + 1 bulk) | YES |
| Depth correctness | Reported depth matches nesting | 9 tests pass | YES |
| Equivalence | De Morgan's, double-Not | 3 tests pass | YES |
| Features used | Correct feature tracking | 4 tests pass | YES |
| Aggregators | Mean/Count/Max correct | 5 tests pass | YES |
| Rule sampler | Reproducible, varied, stress-tested | 8 tests pass | YES |

---

## Deviations from Plan

_None. Implementation follows SR-9 spec exactly._

---

## Completion Summary

TASK-02 delivered the full Classification Rule DSL in `src/dsl/classification_dsl.py` (735 lines). The DSL provides a typed language for specifying classification rules programmatically, with predicates, boolean combinators, classifiers (if-then-else, decision lists, decision trees), aggregators, and a rule sampler with depth control. All 55 V-9 tests pass. The implementation uses frozen dataclasses and ABC base classes, consistent with the schema system. Future tasks (TASK-04 Task Registry, TASK-05 Data Generator) import from this module to create C-tier classification tasks.
