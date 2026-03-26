# TASK-03: Sequence DSL

> **Status:** COMPLETE ✓
> **Builds:** SR-10
> **Validation:** V-10
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-10), Part 3 (V-10), Part 4 (TASK-03)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| List-to-list ops: `Sort`, `Reverse`, `Unique`, `PrefixSum`, `MapAbs`, `MapSign`, `MapParity` | `src/dsl/sequence_dsl.py` | DONE |
| Parameterized ops: `Take`, `Drop`, `FilterGt`, `FilterEven`, `FilterOdd`, `MapMod` | `src/dsl/sequence_dsl.py` | DONE |
| Reducers: `Sum`, `Count`, `Max`, `Min`, `Parity` | `src/dsl/sequence_dsl.py` | DONE |
| Two-input ops: `Concat`, `ZipAdd` | `src/dsl/sequence_dsl.py` | DONE |
| Composition: `Compose` with type checking | `src/dsl/sequence_dsl.py` | DONE |
| Program wrapper: `SeqProgram` | `src/dsl/sequence_dsl.py` | DONE |
| Sampler: `sample_program`, `sample_programs_batch` | `src/dsl/sequence_dsl.py` | DONE |
| Equivalence: `check_functional_equivalence` | `src/dsl/sequence_dsl.py` | DONE |
| V-10 test suite | `tests/test_sequence_dsl.py` | DONE (56 tests) |

---

## Implementation Notes

### Design Choices
- **`SeqOp` ABC** — All operations inherit from `SeqOp` with `evaluate()`, `input_type()`, `output_type()`, `depth()`, and `name()` abstract methods.
- **Uniform `list[int]` type** — All operations take `list[int]` and return `list[int]`. Reducers (Sum, Count, etc.) wrap their scalar result in a single-element list for composability. This avoids type-mismatch issues in `Compose`.
- **`SeqType` enum** — Types are `LIST_INT` and `INT`, though currently all ops use `LIST_INT` since reducers wrap output.
- **`Compose` type checking** — Validates `first.output_type() == second.input_type()` at construction. Currently always passes since all ops are `LIST_INT → LIST_INT`.
- **`Concat` and `ZipAdd` with inner ops** — Two-input operations take an optional `inner` operation. `Concat(inner=Reverse())` concatenates input with `Reverse(input)`. Without inner, operates on self (e.g., `ZipAdd()` doubles each element).
- **Depth is additive for Compose** — `Compose(first, second).depth() = first.depth() + second.depth()`. For `Concat`/`ZipAdd` with inner, depth is `1 + inner.depth()`.
- **Program sampler** — `sample_program(seed, max_depth)` uses weighted random choices: 20% leaf, 50% compose, 15% zip, 15% concat. First op in compose is never a reducer (to avoid composing on single-element lists).
- **Functional equivalence check** — `check_functional_equivalence()` tests two programs on 1000 random inputs to detect semantic duplicates.

### Edge Cases Handled
- Empty input sequences work for all operations (Max/Min return `[]` on empty).
- `Take(n=0)` returns `[]`, `Take(n > len)` returns full input.
- `Drop(n > len)` returns `[]`.
- Negative values handled correctly by `MapSign`, `MapAbs`, `Sort`.

---

## Acceptance Criteria Results

V-10 validation: **56 tests, all passing** (pytest, 0.16s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| Type safety | Invalid params rejected | 4 tests pass | YES |
| Determinism | Same input → same output | 4 tests pass | YES |
| Known programs | 5+ hand-verified programs | 27 tests (22 per-op + 5 composed + bulk) | YES |
| Depth correctness | Reported depth matches nesting | 7 tests pass | YES |
| Deduplication | Sampled programs mostly distinct | 2 tests pass | YES |
| Sampler | Reproducible, stress-tested | 5 tests pass | YES |
| Edge cases | Empty, single, boundary inputs | 7 tests pass | YES |

---

## Deviations from Plan

### DEV-003: Reducers return single-element lists instead of scalars

- **Type:** INTERFACE_CHANGE (minor)
- **What changed:** SR-10 spec implies reducers produce `int`. Implementation wraps in `[int]` for uniform composability.
- **Why:** All ops being `list[int] → list[int]` eliminates type-mismatch errors in `Compose` and simplifies the type system.
- **Impact:** Downstream consumers must unwrap `[result]` if a scalar is needed.

---

## Completion Summary

TASK-03 delivered the full Sequence DSL in `src/dsl/sequence_dsl.py` (590 lines). The DSL covers 18+ primitives across 4 categories (list-to-list, parameterized, reducers, two-input), with typed sequential composition, a program sampler with depth control, and functional equivalence checking. All 56 V-10 tests pass. One minor deviation: reducers return single-element lists for uniform composability. The DSL is used by TASK-04 (Task Registry) for S-tier tasks and will be used by TASK-12 for S5-tier DSL program experiments.
