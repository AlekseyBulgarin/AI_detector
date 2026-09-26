# Project Status

## Current Phase

Human-in-the-loop ML improvement platform (Milestones A-G) is implemented locally.
Phase 2 baseline behavior is preserved.

## Frontend Status

- Responsive teacher-oriented dashboard is implemented in the existing Flask template.
- Analysis history and interface preferences use browser `localStorage` by design.
- Results now show a three-way decision, threshold zones, model/dataset version, confidence and per-stage latency.
- Feedback controls include label, confidence, training consent and a "prediction is incorrect" flag.
- A retention toggle lets a teacher opt out of storing the analyzed text.

## Performance Status

- Cold local `import app` improved from approximately 4,449ms to 1,934ms by removing unconditional NLTK downloader work.
- Warm prediction remains in the single-digit millisecond range and is below the 500ms local target.
- Repeated text uses a model-version-aware bounded cache and indexed SQLite lookup.
- Request timings are emitted as structured logs, and `/api/check` returns `feature_ms`, `prediction_ms`, `model_total_ms`, `db_write_ms` and `latency_ms`.
- Detailed measurements and caveats are documented in `reports/performance_audit.md` and `reports/performance_before_after.md`.

## Privacy Status

- `store_text` (retention, default true, `STORE_TEXT_DEFAULT`) and `allow_training` (training consent) are independent controls.
- With retention off only hashes, metadata and the prediction are stored; the raw text never reaches the database.
- Export requires approved status + consent + retained text + a `human`/`ai` label.
- Details: `docs/privacy.md`.

## Feedback Learning Loop

- Feedback stores the analyzed text, prediction snapshot, model version, consent, review status, hard-case categories and timestamps through an idempotent SQLite migration.
- Hard-case mining ranks false positives, false negatives, high disagreement, low confidence and `ai-assisted` records first.
- `/admin/feedback` is review-only and protected: `403` without `ADMIN_TOKEN`, `401` without a valid token, plus status and hard-case queue filters.
- `tools/export_feedback_dataset.py` writes reviewed samples into `data/feedback/approved/` with metadata (gitignored, raw text stays out of the repo).
- Feedback never rewrites the production model. Details: `docs/feedback-system.md`.

## Dataset Status

- Raw dataset: 38 samples, balanced by directory label.
- Unique training samples after normalized duplicate filtering: 35.
- Versioned snapshots live in `data/datasets/<version>/` with `manifest.json`, `samples.jsonl` and `split.json`.
- Split is group-aware (`author`/`prompt`/`generation_session`/`topic`); feedback samples never enter the frozen test set.
- Frozen test set: 5 samples.
- Known warnings recorded in every manifest: `HIGH LENGTH LEAKAGE`, `TOPIC LEAKAGE`.
- Details: `docs/dataset-pipeline.md`.

## ML Status

- Candidate models: baseline, word TF-IDF, character TF-IDF, combined.
- Selected local model: combined pipeline, production `phase2-v1`.
- Training writes only to `models/candidates/<version>/`; `--promote` no longer exists.
- Thresholds are calibrated on the validation split (`source: validation_roc`); production thresholds live in `reports/thresholds.json` and change only during promotion.
- Evaluation reports live in `reports/evaluation/<version>.json` with `quality_warnings` and `sample_size_warning`.
- Important limitation: the dataset is too small and has strong length/class differences; metrics are directional only.

## Model Lifecycle Status

- Registry: `models/candidates/`, `models/production/`, `models/archive/`, plus the `models/model.pkl` compatibility mirror.
- Gates: candidate evaluation present, non-empty frozen test, F1/ROC-AUC/FPR/FNR regression limits, model load, pytest, leakage acknowledgement, API smoke tests.
- Leakage warnings block promotion until `--acknowledge-leakage` is passed explicitly.
- Promotion and rollback are explicit human commands (`tools/promote_model.py`, `tools/rollback_model.py`).
- Dry-run verified: `python tools/promote_model.py --version v3 --dry-run --acknowledge-leakage` passes all gates and promotes nothing.
- Details: `docs/model-lifecycle.md`.

## Testing Status

- `python -m pytest -q` -> 85 tests passing.
- Coverage includes API contracts, admin auth, feedback validation/export, privacy policy, decision layer, splitter, model registry, promotion gates and monitoring.

## Next Work

1. Expand and rebalance the dataset to remove length/topic leakage before making any accuracy claim.
2. Increase the frozen test set well above 5 samples so metric deltas become meaningful.
3. Promote a candidate only after a human reviews the leakage warnings and the comparison report.
4. Consider transformer baselines once the dataset can support them.

## Deployment Status

- Render configuration is present in `render.yaml`.
- Python 3.11 is declared in `runtime.txt` and `.python-version`.
- Gunicorn is the production start command.
- Model loading and API failure handling are hardened.
- SQLite feedback remains ephemeral on Render until a managed database is connected.
- `ADMIN_TOKEN` must be set in production; without it the review queue is disabled rather than exposed.
