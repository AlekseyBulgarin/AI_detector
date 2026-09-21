# Project Status

## Current Phase

Phase 2 is implemented locally.

## Frontend Status

- Responsive teacher-oriented dashboard is implemented in the existing Flask template.
- Analysis history and interface preferences use browser `localStorage` by design.
- Backend endpoints and ML behavior remain unchanged.

## Performance Status

- Cold local `import app` improved from approximately 4,449ms to 1,934ms by removing unconditional NLTK downloader work.
- Warm prediction remains in the single-digit millisecond range and is below the 500ms local target.
- Repeated text uses a model-version-aware bounded cache and indexed SQLite lookup.
- Request timings are emitted as structured logs for production measurement.
- Detailed measurements and caveats are documented in `reports/performance_audit.md` and `reports/performance_before_after.md`.

## Feedback Learning Loop

- Feedback stores the analyzed text, prediction snapshot, model version, consent, review status, and timestamps through an idempotent SQLite migration.
- `/admin/feedback` is review-only; approved and consented human/AI labels are the only records exported for training.
- `tools/export_feedback_dataset.py` writes reviewed samples into `data/raw/feedback/` with metadata.
- `train_model.py` uses original data by default and requires `--include-feedback` to include the exported approved dataset. Promotion requires `--promote`.

## ML Status

- Raw dataset: 38 samples, balanced by directory label.
- Unique training samples after normalized duplicate filtering: 35.
- Candidate models: baseline, word TF-IDF, character TF-IDF, combined.
- Selected local model: combined pipeline.
- Evaluation: 80/20 stratified holdout plus 3-fold cross-validation on the training portion.
- Important limitation: the dataset is too small and has strong length/class differences.

## Feedback Status

- Predictions are recorded in `data/feedback.db`.
- User labels are stored as `pending`.
- Feedback is not used for automatic retraining.

## Next Work

Expand and rebalance the dataset before making claims about accuracy or adding transformer models.

## Deployment Status

- Render configuration is present in `render.yaml`.
- Python 3.11 is declared in `runtime.txt` and `.python-version`.
- Gunicorn is the production start command.
- Model loading and API failure handling are hardened.
- SQLite feedback remains ephemeral on Render until a managed database is connected.
