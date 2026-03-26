# TASK-01: Input Schema System

> **Status:** COMPLETE ✓
> **Builds:** SR-2
> **Validation:** V-2
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-2), Part 3 (V-2), Part 4 (TASK-01)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `SequenceInputSchema` | `src/schemas.py` | DONE |
| `TabularInputSchema` | `src/schemas.py` | DONE |
| `NumericalFeatureSpec` | `src/schemas.py` | DONE |
| `CategoricalFeatureSpec` | `src/schemas.py` | DONE |
| `Distribution` enum | `src/schemas.py` | DONE |
| `ElementType` enum | `src/schemas.py` | DONE |
| V-2 test suite | `tests/test_schemas.py` | DONE (52 tests) |

---

## Implementation Notes

### Design Choices

- **Frozen dataclasses** — All schema and feature spec classes use `@dataclass(frozen=True)` for immutability. This prevents accidental mutation and supports hashing.
- **Tuples over lists** — `CategoricalFeatureSpec.values`, `CategoricalFeatureSpec.weights`, and `TabularInputSchema` feature lists use tuples for immutability compatibility with frozen dataclasses.
- **`np.random.Generator`** — Used modern NumPy random API (`default_rng`) instead of legacy `np.random.RandomState`. Every `sample()` method takes a seed and creates its own generator for full reproducibility.
- **Distribution handling for Normal** — Normal distribution uses mean=center, std=range/6 with clipping to bounds. This ensures ~99.7% of samples fall in range naturally, with hard clipping for edge cases.
- **Distribution handling for Exponential** — Exponential uses lambda=3/range (right-skewed, mean at 1/3 of range), shifted to start at min_val, clipped to max_val.
- **`validate_input()` on both schema types** — Allows downstream modules (Data Generator, Model Harness) to verify inputs against schemas without knowing the schema type.
- **`with_extra_irrelevant()`** — `TabularInputSchema` has a method to return a new schema with additional distractor features. This is designed for `DistractorSplit` (SR-4, TASK-06).
- **`features()` iterator** — Both schema types expose a `features()` method for generic iteration. For `SequenceInputSchema` it returns empty (no named features); for `TabularInputSchema` it yields (name, spec) pairs.
- **Batch sampling** — Both schemas have `sample_batch(seed, n)` which uses a single RNG for the entire batch (sequential draws). This is different from calling `sample(seed+i)` for each — the batch method produces correlated but well-distributed sequences from one generator.

### Edge Cases Handled

- min_val == max_val for numerical features (constant feature) — works correctly
- min_length == max_length for sequences (fixed-length) — works correctly
- Binary element type enforces value_range=(0,1)
- Char element type requires alphabet

---

## Acceptance Criteria Results

V-2 validation: **52 tests, all passing** (pytest, 4.13s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| Schema completeness (construction validation) | Invalid configs rejected | 14 tests pass | YES |
| Sampling validity — type checks | All samples have correct types | 500 samples × multiple schemas | YES |
| Sampling validity — range checks | All values in specified range | 500 samples × multiple schemas | YES |
| Sampling validity — cardinality checks | Categorical values in allowed set | 500 samples × multiple schemas | YES |
| Reproducibility — same seed same output | Identical outputs | Both schema types verified | YES |
| Reproducibility — different seeds differ | Different outputs | Both schema types verified | YES |
| Distribution — uniform numerical (KS test) | p > 0.01 | PASS (10k samples) | YES |
| Distribution — normal numerical | Mean near center | PASS (10k samples) | YES |
| Distribution — exponential numerical | Right-skewed (median < mean) | PASS (10k samples) | YES |
| Distribution — uniform categorical (chi2) | p > 0.01 | PASS (10k samples) | YES |
| Distribution — weighted categorical (chi2) | p > 0.01 | PASS (10k samples) | YES |
| Distribution — sequence lengths (chi2) | p > 0.01 | PASS (10k samples) | YES |
| Distribution — int element values (chi2) | p > 0.01 | PASS (10k samples) | YES |

---

## Deviations from Plan

### DEV-001: Added Distribution and ElementType enums (minor scope addition)

- **Type:** SCOPE_CHANGE
- **What changed:** Added `Distribution` and `ElementType` as proper Enum classes rather than raw strings.
- **Why:** Type safety — prevents typos in distribution/element_type strings and enables IDE autocomplete.
- **Impact:** Downstream modules should import and use these enums when constructing schemas.

### DEV-002: Tuples instead of lists for frozen dataclass compatibility

- **Type:** INTERFACE_CHANGE (minor)
- **What changed:** SR-2 spec shows `list[str]` for `CategoricalFeatureSpec.values` and `list[float]` for weights. Implementation uses `Tuple[str, ...]` and `Tuple[float, ...]`.
- **Why:** Frozen dataclasses require hashable fields. Tuples are hashable; lists are not.
- **Impact:** Callers must pass tuples, e.g. `values=("a", "b")` not `values=["a", "b"]`.

---

## Completion Summary

TASK-01 delivered all four schema/spec classes plus two enums in `src/schemas.py` (340 lines). The implementation uses frozen dataclasses with numpy's modern random API for reproducible sampling. All 52 V-2 validation tests pass, covering construction validation, sampling validity (type/range/cardinality), reproducibility, and distribution checks (KS and chi-squared). Two minor deviations from the spec: enums instead of raw strings, and tuples instead of lists for frozen dataclass compatibility. Future tasks should import `Distribution` and `ElementType` enums, and use tuples when constructing specs.
