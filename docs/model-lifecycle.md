# Model lifecycle

A model moves through training, evaluation, gating, explicit promotion and
rollback. Training alone never changes what teachers see.

```text
train_model.py                 -> models/candidates/<version>/
tools/evaluate_candidate.py    -> reports/evaluation/<version>.json
tools/compare_models.py        -> reports/model_comparison_<version>.md + gates
tools/promote_model.py         -> models/production/ + models/model.pkl
tools/rollback_model.py        -> restore the previous archived production
```

## Registry layout

| Path | Role |
| --- | --- |
| `models/candidates/<version>/` | Trained candidates: `model.pkl`, `metadata.json`, `metrics.json`, `thresholds.json` |
| `models/production/` | Current promoted artifacts plus `promotion.json` / `rollback.json` |
| `models/archive/` | Timestamped snapshots of every replaced production model |
| `models/model.pkl` | Compatibility mirror updated only during promotion or rollback |

`*.pkl` files are gitignored except the legacy flat artifacts required for
deployment.

## Training

```bash
venv\Scripts\python.exe train_model.py --dataset v2 --model-version v3
```

- `--promote` was removed on purpose; training is candidate-only
- the stored dataset split is used when available
- thresholds are calibrated on the **validation** split (`source: validation_roc`)
  so the frozen test set is never used to pick a decision boundary
- metrics include `split_summary` and `quality_warnings`
- output ends with `Candidate is NOT promoted.`

## Evaluation

```bash
venv\Scripts\python.exe tools\evaluate_candidate.py --version v3
venv\Scripts\python.exe tools\evaluate_candidate.py --production
```

Both use the same frozen test records and the same metric definitions, so the
delta between them is meaningful. The report records `quality_warnings` from the
dataset manifest and a `sample_size_warning` when fewer than 30 frozen samples
are available.

## Gates

```bash
venv\Scripts\python.exe tools\compare_models.py --version v3
```

`src/promotion.py::evaluate_gates` runs:

| Gate | Rule |
| --- | --- |
| `candidate_evaluation_present` | candidate evaluation exists |
| `frozen_test_non_empty` | at least one frozen sample |
| `no_regression_f1` | F1 delta ≥ `-0.02` |
| `no_regression_roc_auc` | ROC-AUC delta ≥ `-0.02` |
| `no_regression_false_positive_rate` | FPR delta ≤ `+0.02` |
| `no_regression_false_negative_rate` | FNR delta ≤ `+0.02` |
| `model_loads` | candidate pipeline loads |
| `regression_tests` | `pytest` passes |
| `data_leakage_checks` | no leakage, or explicitly acknowledged |
| `api_smoke_tests` | `/health` and `/api/check` respond |

`tools/promotion_checks.py` supplies the last four. `data_leakage_checks` fails
while the corpus carries `HIGH LENGTH LEAKAGE` or `TOPIC LEAKAGE`; the only way
to pass it is a deliberate `--acknowledge-leakage`.

`compare_models.py` exits with code `2` when any gate fails.

## Promotion

```bash
# preview only, writes reports/gates/<version>.json and the comparison markdown
venv\Scripts\python.exe tools\promote_model.py --version v3 --dry-run

# real promotion - a human decision
venv\Scripts\python.exe tools\promote_model.py --version v3 --acknowledge-leakage
```

Promotion archives the current production model, copies the candidate into
`models/production/`, refreshes `models/model.pkl`, installs the candidate
thresholds into `reports/thresholds.json` (the file `src/decision.py` reads) and
appends to `reports/promotions.jsonl`.

## Rollback

```bash
venv\Scripts\python.exe tools\rollback_model.py --dry-run   # preview only
venv\Scripts\python.exe tools\rollback_model.py             # apply
venv\Scripts\python.exe tools\rollback_model.py --reason quality_regression
```

Rollback restores the most recent archive, rewrites the compatibility mirror,
reinstalls the archived thresholds and appends to `reports/rollbacks.jsonl`.
With no archive present it fails loudly.

## Honest reporting

The frozen test set currently holds 5 samples and the corpus has strong length
leakage. Directional metrics may be compared between models; they must never be
presented as proof that a specific student used AI.
