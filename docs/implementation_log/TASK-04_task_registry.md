# TASK-04: Task Registry

> **Status:** COMPLETE ✓
> **Builds:** SR-1
> **Validation:** V-1
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-1), Part 3 (V-1), Part 4 (TASK-04)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25
> **Milestone:** FOUNDATION (TASK-01–04 complete)

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `TaskSpec` dataclass | `src/registry.py` | DONE |
| `TaskRegistry` class (get, by_tier, by_track, register, contains) | `src/registry.py` | DONE |
| Generic verifiers: `exact_match_verifier`, `classification_verifier` | `src/registry.py` | DONE |
| `build_default_registry()` factory | `src/registry.py` | DONE |
| 28 registered tasks across 8 tiers | `src/registry.py` | DONE |
| V-1 test suite | `tests/test_registry.py` | DONE (34 tests) |

---

## Implementation Notes

### Design Choices
- **`TaskSpec` dataclass** — Each task exposes a standard interface: `task_id`, `tier`, `track`, `description`, `input_schema`, `output_type`, `n_classes`, `reference_algorithm` (callable), `input_sampler` (callable), `verifier` (callable), `complexity_metadata` (dict).
- **Callable-based interface** — `reference_algorithm`, `input_sampler`, and `verifier` are plain callables (functions/lambdas), not class instances. This keeps the interface minimal and avoids unnecessary abstraction.
- **28 tasks across 8 tiers** — S0 (2 control tasks), S1 (8 simple transforms), S2 (4 stateful), S3 (3 multi-step), C0 (2 controls), C1 (5 single-rule), C2 (5 multi-feature), C3 (3 interaction).
- **S0 control tasks use deterministic hash-based pseudo-randomness** — `S0.1_random_labels` uses a stable content hash as a seed to generate "random" but deterministic outputs. This keeps the task reproducible across processes and machines while remaining unlearnable.
- **`build_default_registry()`** — Factory function that builds and populates the registry with all standard tasks. Each tier has its own builder function (`_build_s0_tasks()`, etc.) for organization.

### Registered Tasks

| Tier | Count | Examples |
|---|---|---|
| S0 | 2 | random_labels, lookup_table |
| S1 | 8 | reverse, sort, rotate, count_symbol, parity, prefix_sum, deduplicate, extrema |
| S2 | 4 | cumulative_xor, balanced_parens, running_min, checksum |
| S3 | 3 | dedup_sort_count, filter_sort_sum, rle_encode |
| C0 | 2 | random_class, majority_class |
| C1 | 5 | numeric_threshold, range_binning, categorical_match, numeric_comparison, modular_class |
| C2 | 5 | and_rule, or_rule, nested_if_else, k_of_n, categorical_gate |
| C3 | 3 | xor, rank_based, interaction_poly |

---

## Acceptance Criteria Results

V-1 validation: **34 tests, all passing** (pytest, 0.22s)

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| Registry structure | All tiers present, unique IDs | 10 tests pass | YES |
| Task spec completeness | All required fields populated | 4 tests pass | YES |
| Determinism | Same input → same output for all tasks | 2 tests pass (50 samples × 28 tasks) | YES |
| Input sampler validity | Sampled inputs conform to schema | 2 tests pass (100 samples × 28 tasks) | YES |
| Verifier agreement | Verifier accepts correct output, rejects wrong | 3 tests pass | YES |
| Output types | Sequence outputs are `list[int]`, classification outputs are `str` | 2 tests pass | YES |
| Spot checks | 9 specific tasks verified against known outputs | 9 tests pass | YES |
| Reproducibility | Same seed → same input → same output | 2 tests pass | YES |

---

## Deviations from Plan

### DEV-004: Not all task tiers registered yet (S4, S5, C4, C5 deferred)

- **Type:** SCOPE_DEFERRAL
- **What changed:** S4 (structural/graph), S5 (DSL programs), C4 (stateful aggregation), and C5 (multi-step compositional) tasks are not registered in the initial registry. Only S0–S3 and C0–C3 are included.
- **Why:** S4/S5/C4/C5 tasks require more complex reference implementations. The initial registry covers all tiers needed for TASK-05 through TASK-11 (smoke tests). Higher tiers will be added when TASK-12/13 (experiment tasks) are implemented.
- **Impact:** No impact on pipeline development. Higher-tier tasks will be added to the registry when needed.

---

## Completion Summary

TASK-04 delivered the Task Registry in `src/registry.py`. The registry holds 28 tasks across 8 tiers (S0–S3, C0–C3), each with a standard interface exposing input schema, reference algorithm, sampler, and verifier. All 34 V-1 tests pass, covering structure, completeness, determinism, input validity, verifier agreement, output types, spot checks, and reproducibility. This completes the FOUNDATION milestone (TASK-01–04). One deviation: S4/S5/C4/C5 tiers deferred to experiment implementation phase.
