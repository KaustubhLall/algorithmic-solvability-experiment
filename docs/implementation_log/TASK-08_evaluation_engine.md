# TASK-08: Evaluation Engine

> **Status:** COMPLETE ✓
> **Builds:** SR-6
> **Validation:** V-6 (52 tests passing)
> **Spec:** `EXPERIMENT_CATALOG.md` Part 1 (SR-6), Part 3 (V-6), Part 4 (TASK-08)
> **Date Started:** 2025-03-25
> **Date Completed:** 2025-03-25

---

## Deliverables

| Deliverable | File | Status |
|---|---|---|
| `evaluate()` function | `src/evaluation.py` | ✓ DONE |
| `EvalReport` dataclass | `src/evaluation.py` | ✓ DONE |
| `PerClassMetrics` dataclass | `src/evaluation.py` | ✓ DONE |
| Classification metrics (accuracy, per-class P/R/F1, macro F1, weighted F1, confusion matrix) | `src/evaluation.py` | ✓ DONE |
| Sequence metrics (exact match, token accuracy) | `src/evaluation.py` | ✓ DONE |
| Error taxonomy (track-specific) | `src/evaluation.py` | ✓ DONE |
| Metadata-conditioned metric breakdowns | `src/evaluation.py` | ✓ DONE |
| `evaluate_prediction_result()` convenience | `src/evaluation.py` | ✓ DONE |
| `eval_report_to_dict()` serialization | `src/evaluation.py` | ✓ DONE |
| V-6 test suite | `tests/test_evaluation.py` | ✓ 52 tests |

---

## Implementation Notes

### Architecture

Single file `src/evaluation.py` (380 lines) containing:
- **Data classes:** `PerClassMetrics`, `EvalReport`
- **Core metrics:** `_compute_accuracy`, `_compute_confusion_matrix`, `_per_class_from_confusion`, `_macro_f1`, `_weighted_f1`
- **Error taxonomy:** `_classification_error_taxonomy`, `_sequence_error_taxonomy`
- **Sequence metrics:** `_compute_token_accuracy`, `_parse_sequence_str`
- **Metadata conditioning:** `_metadata_conditioned_accuracy`
- **Main entry points:** `evaluate()`, `evaluate_prediction_result()`, `eval_report_to_dict()`

### Key Design Choices

1. **String-based interface (ADR-014).** All predictions and ground truth are `List[str]`. This matches `ModelHarness.run()` output directly. No type coercion needed.

2. **Track dispatch.** `evaluate()` checks `task.track` and dispatches to `_evaluate_classification()` or `_evaluate_sequence()`. Unknown tracks raise `ValueError`.

3. **Confusion matrix convention.** Rows = true class, columns = predicted class. Class labels sorted alphabetically from the union of ground truth and predictions.

4. **Per-class metrics from confusion matrix.** Rather than re-scanning predictions, precision/recall/F1 are derived directly from the confusion matrix in a single pass.

5. **Track-specific error taxonomies (ADR-015).**
   - Classification: `correct`, `wrong_class`, `unknown_class`
   - Sequence: `correct`, `length_mismatch`, `content_error`, `off_by_one`

6. **Token accuracy for sequences.** Parses stringified lists (e.g., `"[1, 2, 3]"` → `["1", "2", "3"]`). Handles length mismatches by counting extra positions as wrong. Non-parseable outputs are skipped gracefully. Returns `None` if no tokens are available.

7. **Metadata-conditioned metrics.** Groups samples by metadata key values and computes accuracy per group. Supports multiple condition keys simultaneously. Only computed when both `metadata` and `condition_keys` are provided.

### Edge Cases Handled

- **Empty predictions:** accuracy returns 0.0, n_samples = 0
- **Single class:** confusion matrix is 1×1, all metrics degenerate correctly
- **Length mismatch in predictions vs ground_truth:** raises ValueError
- **Non-parseable sequence strings:** error taxonomy classifies as `content_error`, token accuracy skips
- **Empty sequences `[]`:** exact match works, token accuracy returns `None` (0 tokens)
- **Unknown predicted class:** in classification, since labels are derived from union of gt ∪ pred, an "unknown" class becomes part of the label set. The `unknown_class` taxonomy category only fires if a prediction isn't in the label set (which can't happen with the current implementation — noted for future if labels are pre-specified).

---

## Acceptance Criteria Results

V-6 validation as specified in EXPERIMENT_CATALOG.md Part 3:

| Check | Expected | Actual | Pass? |
|---|---|---|---|
| Perfect prediction test (classification) | accuracy=1.0, all F1=1.0, errors=0 | ✓ All exact | ✓ |
| Perfect prediction test (sequence) | exact_match=1.0, token_accuracy=1.0 | ✓ All exact | ✓ |
| Worst prediction test (classification) | accuracy=0.0, F1=0.0 | ✓ All zero | ✓ |
| Worst prediction test (sequence) | exact_match=0.0 | ✓ Zero | ✓ |
| Known confusion matrix (2×2) | [[2,1],[1,2]] | ✓ Exact match | ✓ |
| Known confusion matrix (3×3) | [[1,1,0],[0,1,1],[1,0,1]] | ✓ Exact match | ✓ |
| Metric consistency (acc = diag/total) | Multiple parametrized cases | ✓ All within 1e-9 | ✓ |
| Classification dispatch (correct fields) | per_class, confusion populated; exact_match None | ✓ Correct | ✓ |
| Sequence dispatch (correct fields) | exact_match, token_accuracy populated; confusion None | ✓ Correct | ✓ |
| Unknown track raises | ValueError | ✓ Raised | ✓ |

**Full test output:** 52 tests passed in 0.84s. No failures, no errors.

Integration tests with real registry tasks (C1.1_numeric_threshold with decision tree, S1.2_sort with sequence baseline) both pass successfully.

---

## Test Breakdown

| Test Class | Tests | What it covers |
|---|---|---|
| TestPerfectPrediction | 2 | Both tracks perfect → all metrics 1.0 |
| TestWorstPrediction | 2 | Both tracks all-wrong → metrics 0.0 |
| TestKnownConfusionMatrix | 3 | 2×2, 3×3, 1×1 confusion matrices |
| TestMetricConsistency | 4 | Parametrized acc = diagonal/total |
| TestTrackDispatch | 3 | Correct field population per track |
| TestPerClassMetrics | 2 | Binary and multiclass P/R/F1 |
| TestMacroWeightedF1 | 3 | Unweighted mean, support-weighted, perfect case |
| TestSequenceMetrics | 5 | Exact match, token accuracy, length mismatch, empty, non-parseable |
| TestErrorTaxonomy | 6 | Classification + sequence error types |
| TestMetadataConditioned | 3 | Single key, multiple keys, no metadata |
| TestEdgeCases | 5 | Empty, single sample, single class, length mismatch, metadata preservation |
| TestEvaluatePredictionResult | 1 | Convenience function with mock PredictionResult |
| TestReportSerialization | 3 | to_dict for both tracks, JSON serializable |
| TestInternalHelpers | 9 | Unit tests for all internal functions |
| TestIntegrationWithRegistry | 2 | Real tasks from registry with model harness |

**Total: 52 tests**

---

## Deviations from Plan

No deviations from the SR-6 spec. The implementation matches the specified interface exactly:
- `evaluate(predictions, ground_truth, task, split_name)` → `EvalReport`
- `EvalReport` contains all fields from the spec: accuracy, per_class_metrics, confusion_matrix, exact_match, token_accuracy, error_taxonomy, metadata_conditioned_metrics

Two optional parameters added to `evaluate()` for convenience:
- `metadata: Optional[List[Dict[str, Any]]]` — per-sample metadata for conditioned breakdowns
- `condition_keys: Optional[List[str]]` — which metadata keys to condition on

These are additive (not breaking) and were implied by the `metadata_conditioned_metrics` field in the spec.

---

## Completion Summary

TASK-08 implemented the Evaluation Engine (SR-6) in a single file `src/evaluation.py` (380 lines). The engine dispatches to classification or sequence metric computation based on `task.track`, computing accuracy, confusion matrix, per-class P/R/F1, macro/weighted F1 (classification), exact match and token accuracy (sequence), track-specific error taxonomies, and optional metadata-conditioned accuracy breakdowns. All metrics are computed from first principles (no sklearn.metrics dependency) for full transparency and control. The implementation accepts string-based predictions and ground truth, matching the ModelHarness output format directly. 52 V-6 validation tests pass covering perfect/worst predictions, known confusion matrices, metric consistency, track dispatch, edge cases, and integration with real registry tasks. No deviations from the spec. Two ADRs logged (ADR-014: string-based interface, ADR-015: track-specific error taxonomies). TASK-09 (Experiment Runner) can now proceed.
