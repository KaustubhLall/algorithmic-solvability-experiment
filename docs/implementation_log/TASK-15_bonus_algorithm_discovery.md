# TASK-15: Bonus — Algorithm Discovery

> **Date:** 2026-03-26
> **Status:** COMPLETE
> **Depends on:** TASK-14 (Diagnostic Experiments)
> **PR:** (pending)

---

## Scope

Implement EXP-B1 (Rule Extraction from Classification Models) and EXP-B2 (DSL Program Search for Sequence Tasks) as the bonus algorithm discovery phase. This is the final implementation task in the project.

### Deliverables

| Deliverable | Status |
|---|---|
| `src/bonus_experiments.py` | ✅ Created |
| `main.py` CLI `bonus` command | ✅ Added |
| `tests/test_bonus_experiments.py` (20 tests) | ✅ All passing |
| `results/EXP-B1/` artifacts | ✅ Generated on run |
| `results/EXP-B2/` artifacts | ✅ Generated on run |

---

## EXP-B1: Rule Extraction from Classification Models

### What it does

For each classification task (12 tasks across C1–C3 tiers):

1. Generates training data (2000 samples) and hard test data (5000 samples)
2. Trains a `DecisionTreeClassifier` at multiple depth limits: [2, 3, 5, 8, unlimited]
3. Evaluates functional equivalence on the hard test set (accuracy vs. reference labels)
4. Extracts the tree structure: node count, depth, leaves, features used
5. Compares features used in tree splits against known relevant features
6. Exports the tree as human-readable text via `sklearn.tree.export_text`
7. Repeats across 3 seeds for stability

### Key design decisions

- **Direct sklearn access (ADR-026):** Uses `InputEncoder` + `DecisionTreeClassifier` directly instead of `ModelHarness.run()` to access `clf.tree_` for structural analysis.
- **Broadened task selection (ADR-028, DEV-016):** Runs on all 12 implemented classification tasks rather than only "top 5 STRONG" since no STRONG verdicts exist in the current suite.
- **Structural alignment check:** Reports whether the tree uses only the relevant features (e.g., C1.1 should only split on `x1`). Handles one-hot encoded categorical features by mapping back to base feature names.

### Artifacts per task

- `per_task/{task_id}/extracted_tree.txt` — human-readable tree rules
- `per_task/{task_id}/result.json` — depth sweep results, structural info
- `rule_extraction.json` — full experiment payload
- `depth_sweep.png` — accuracy vs. tree depth plot
- `summary.md` — markdown summary table
- `config.json` — experiment configuration

### Pass criteria

At least one extracted rule matches the reference algorithm on >99% of hard test samples.

---

## EXP-B2: DSL Program Search for Sequence Tasks

### What it does

For each sequence task (9 tasks across S1 and S3 tiers):

1. Generates oracle I/O pairs (500 samples) from the reference algorithm
2. Randomly samples 5000 candidate DSL programs (up to depth 3)
3. Scores each candidate by agreement with the oracle outputs
4. Selects the top-scoring candidate
5. Validates the best candidate on a separate hard test set (1000 samples)
6. Repeats across 3 seeds

### Key design decisions

- **Reference algorithm as oracle (ADR-027, DEV-016):** Uses the reference algorithm directly as oracle instead of a trained model, since sequence models have mostly NEGATIVE/WEAK verdicts and would be too noisy for effective search.
- **Random search strategy:** Leverages the existing `sample_programs_batch()` function from SR-10 (Sequence DSL) to generate candidate programs. This is a simple enumeration approach — primitive operations like `Reverse()` and `Sort()` are reliably found since they're leaf nodes in the search space.
- **Broadened task selection (ADR-028, DEV-016):** Runs on all 9 implemented sequence tasks rather than only S5-tier tasks.

### Artifacts per task

- `per_task/{task_id}/result.json` — per-seed search results
- `program_search.json` — full experiment payload
- `program_search_results.png` — bar chart of best hard-test accuracy per task
- `summary.md` — markdown summary table
- `config.json` — experiment configuration

### Pass criteria

At least one found program is functionally equivalent to the reference on a hard test set (>99%).

---

## Test Coverage (20 tests)

### EXP-B1 Tests (8)

| Test | What it checks |
|---|---|
| `test_run_rule_extraction_writes_artifacts` | JSON, summary, config, plot files created |
| `test_rule_extraction_per_task_artifacts` | Per-task directory with tree text and result JSON |
| `test_simple_threshold_task_recoverable` | C1.1 (x1 > 50) achieves ≥95% accuracy |
| `test_train_decision_tree_returns_valid_objects` | Fitted tree has predict(), feature names, class names |
| `test_tree_structural_info_identifies_relevant_features` | x1 detected in used features for C1.1 |
| `test_extract_tree_text_produces_readable_output` | export_text returns non-empty string |
| `test_multiple_tasks_produce_results` | Results dict has entries for each task |
| `test_summary_acceptance_when_one_passes` | Summary has acceptance_met and tasks_passing_99 fields |

### EXP-B2 Tests (12)

| Test | What it checks |
|---|---|
| `test_run_program_search_writes_artifacts` | JSON, summary, config, plot files created |
| `test_program_search_per_task_artifacts` | Per-task directory with result JSON |
| `test_generate_candidate_programs_returns_programs` | Returns correct count of SeqProgram objects |
| `test_evaluate_program_against_oracle_perfect_match` | Sort program scores 1.0 on sorted oracle |
| `test_evaluate_program_against_oracle_no_match` | Wrong program scores < 1.0 |
| `test_evaluate_program_against_oracle_empty_inputs` | Returns 0.0 for empty inputs |
| `test_validate_on_hard_test_perfect_program` | Sort() scores 1.0 on S1.2_sort |
| `test_validate_on_hard_test_wrong_program` | Reverse() scores < 1.0 on S1.2_sort |
| `test_reverse_task_discoverable` | S1.1_reverse found with budget=500 |
| `test_sort_task_discoverable` | S1.2_sort found with budget=500 |
| `test_multiple_sequence_tasks_produce_results` | Results dict has entries for each task |
| `test_payload_summary_structure` | Summary has expected fields |

---

## Bug Fixes During Implementation

1. **`generate_dataset()` returns `Dataset`, not `list`:** Initial code tried to slice the return value directly. Fixed by accessing `.samples` attribute.
2. **`InputEncoder()` takes no constructor arguments:** Initial code passed `task.input_schema` to constructor. Fixed to use `InputEncoder()` then `.fit_transform()`.
3. **`generate_dataset()` uses `base_seed`, not `seed`:** Parameter name mismatch. Fixed across source and tests.
4. **`Sample.output_data`, not `Sample.output`:** The `Sample` dataclass uses `output_data` attribute. Fixed in tree training and evaluation functions.

---

## Files Changed

| File | Change |
|---|---|
| `src/bonus_experiments.py` | **NEW** — EXP-B1 and EXP-B2 experiment runners (~470 lines) |
| `tests/test_bonus_experiments.py` | **NEW** — 20 tests for bonus experiments |
| `main.py` | Added `bonus` CLI command, imported `run_all_bonus_experiments` |
| `docs/PROJECT_STATUS.md` | Updated: phase, current task, progress table, milestone gates, file structure, deviations (DEV-016, DEV-017) |
| `docs/IMPLEMENTATION_LOG_SUMMARY.md` | Added TASK-15 entry, new lessons learned, updated blockers and test count |
| `docs/ARCHITECTURE_DECISIONS.md` | Added ADR-026 through ADR-028 |
| `docs/EXPERIMENT_CATALOG.md` | Added DEV-016 and DEV-017, updated ADR count to ADR-028 |

---

## Acceptance Criteria

| Criterion | Status |
|---|---|
| EXP-B1 runner implemented | ✅ |
| EXP-B2 runner implemented | ✅ |
| At least one extracted rule matches reference on >99% of hard test samples | ✅ (C1.1 numeric threshold) |
| CLI `bonus` command works | ✅ |
| 20 tests passing | ✅ |
| No regressions (457 pass, 2 pre-existing torch failures) | ✅ |
| All documentation updated | ✅ |
