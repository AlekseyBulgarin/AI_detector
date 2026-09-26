# Dataset pipeline

The dataset is versioned so every metric is traceable to an immutable snapshot.
A metric without a dataset version is not comparable to anything.

## Stages

```text
data/raw/*.txt
    -> tools/dataset_analyzer.py          quality report
    -> tools/export_feedback_dataset.py   approved feedback samples
    -> tools/build_dataset_version.py     frozen snapshot in data/datasets/<version>/
    -> train_model.py --dataset <version> candidate in models/candidates/<version>/
```

## Quality analysis

`python tools/dataset_analyzer.py` writes `reports/dataset_quality.json` and
`reports/dataset_quality.md`: class balance, duplicates, length distribution,
topic distribution and leakage warnings.

Current corpus (do not over-claim):

- 38 raw files, 19 human / 19 AI
- 35 unique samples after normalized duplicate filtering
- human median ≈ 33 words vs AI median ≈ 69 words → **HIGH LENGTH LEAKAGE**
- topics do not align with the raw corpus → **TOPIC LEAKAGE**

Length leakage means a model can score well by looking at length alone. Metrics
must always be reported together with these warnings.

## Versioned snapshots

`python tools/build_dataset_version.py --version v2 [--no-feedback]` writes:

```text
data/datasets/<version>/manifest.json   counts, balance, quality_warnings
data/datasets/<version>/samples.jsonl   the frozen samples (raw text, not committed)
data/datasets/<version>/split.json      train/validation/frozen_test assignments
data/manifests/<version>.json           metadata copy for the manifest store
```

`data/datasets/` and `data/feedback/` are gitignored because they contain raw
student text. The manifests are committed.

When no `--dataset` version is supplied, `train_model.py --include-feedback`
still trains on the raw corpus plus `data/feedback/approved/`. Prefer
`--dataset` once a registry version exists: only the registry snapshot has a
frozen split and quality warnings.

## Splitting

`src/splitter.py` splits by **group**, not by row:

- group key order: `author`, `prompt`, `generation_session`, `topic`, then
  `normalized_hash`, then `id`
- `source` is deliberately **not** part of the key: the whole raw corpus shares
  one source value, which would collapse everything into a single group and
  silently disable the group-aware split
- related documents always land in the same split
- feedback-derived samples are never placed in `frozen_test`

Default proportions: 70% train, 15% validation, 15% frozen test.

The frozen test set is the only set used for reporting a candidate. Validation
is used for model selection and threshold calibration. Training code prefers the
stored split (`train_model._stored_split`) so a re-run can never re-split and
leak frozen samples into training.

## Rules

- Never evaluate on training data.
- Never change a frozen split in place; build a new dataset version.
- Report `quality_warnings` and `sample_size_warning` alongside any metric.
- With only 5 frozen samples, all metrics are directional. Do not quote them as
  accuracy.
