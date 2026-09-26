# Dataset Quality Report

Generated: 2026-09-26T16:41:41.154233+00:00

**Overall status:** PASS

## Summary

- Total samples: 35
- Human samples: 16
- AI samples: 19
- Feedback samples: 0

## Quality Gates

| Gate | Status | Checked | Failed | Blocking |
| --- | --- | ---: | ---: | --- |
| valid_encoding | PASS | 35 | 0 | yes |
| valid_label | PASS | 35 | 0 | yes |
| sufficient_length | PASS | 35 | 0 | yes |
| consent | PASS | 0 | 0 | yes |
| approved_review_status | PASS | 0 | 0 | yes |
| exact_duplicate | PASS | 35 | 0 | yes |
| near_duplicate | PASS | 35 | 0 | yes |
| source_metadata | PASS | 35 | 0 | no |

## Bias And Leakage

- **HIGH LENGTH LEAKAGE** - AI mean words 69.21 vs human 31 (ratio 2.233, Cohen's d 1.677). The detector may learn 'long text = AI'.
- **TOPIC LEAKAGE** - topic metadata unavailable (coverage 0.0).

## Length Distributions

| Field | Human mean | AI mean | Ratio | Cohen's d |
| --- | ---: | ---: | ---: | ---: |
| characters | 209.38 | 431.63 | 2.062 | 1.566 |
| words | 31 | 69.21 | 2.233 | 1.677 |
| sentences | 3.06 | 9.68 | 3.162 | 1.872 |

## Duplicates

- Exact duplicate groups: 0
- Normalized duplicate groups: 0
