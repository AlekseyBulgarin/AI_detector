# Model Comparison: Production vs v3

Generated: 2026-09-26T17:01:15.923004+00:00

| Metric | Production | Candidate | Delta |
| --- | ---: | ---: | ---: |
| f1 | 1.0000 | 1.0000 | +0.0000 |
| roc_auc | 1.0000 | 1.0000 | +0.0000 |
| false_positive_rate | 0.0000 | 0.0000 | +0.0000 |
| false_negative_rate | 0.0000 | 0.0000 | +0.0000 |

## Notes

- Production metrics come from the frozen test evaluation recorded when that model was promoted.
- Candidate metrics come from the same frozen evaluation dataset, so the delta is directly comparable.
- Human text classified as AI (false positives) is weighted most heavily because it is the most damaging error for a teacher.

## Promotion Gates

- [PASS] `candidate_evaluation_present` - Frozen test metrics are available
- [PASS] `frozen_test_non_empty` - 5 frozen test samples
- [PASS] `no_regression_f1` - f1: 1.0000 -> 1.0000 (delta +0.0000, limit f1_delta_min=-0.02)
- [PASS] `no_regression_roc_auc` - roc_auc: 1.0000 -> 1.0000 (delta +0.0000, limit roc_auc_delta_min=-0.02)
- [PASS] `no_regression_false_positive_rate` - false_positive_rate: 0.0000 -> 0.0000 (delta +0.0000, limit false_positive_rate_delta_max=0.02)
- [PASS] `no_regression_false_negative_rate` - false_negative_rate: 0.0000 -> 0.0000 (delta +0.0000, limit false_negative_rate_delta_max=0.02)
- [PASS] `model_loads` - candidate pipeline loaded
- [PASS] `regression_tests` - 85 passed in 4.94s
- [PASS] `data_leakage_checks` - leakage acknowledged explicitly: HIGH LENGTH LEAKAGE, TOPIC LEAKAGE
- [PASS] `api_smoke_tests` - health and prediction smoke checks passed

**Outcome:** all gates passed; promotion still requires an explicit `tools/promote_model.py` invocation.
