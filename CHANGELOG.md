# Changelog

## HITL ML Improvement Platform

- Added a central privacy policy separating per-prediction retention (`store_text`) from training consent (`allow_training`), with export requiring approved status, consent, retained text and a `human`/`ai` label.
- Added hard-case mining (`src/priority.py`): false-positive, false-negative, high-disagreement, low-confidence, `ai-assisted` and edge-case categories plus a 0-100 priority score; feedback records now store `hard_case_categories`.
- Added a protected review queue: `ADMIN_TOKEN` gate (403 when unset, 401 with a login form otherwise), queue filters for every status and hard-case category, and review actions that record reviewer notes and timestamps.
- Added a versioned dataset registry (`data/datasets/<version>/` with manifest, samples and split) plus `tools/build_dataset_version.py` and `tools/dataset_analyzer.py`.
- Added group-aware splitting (`src/splitter.py`) keyed on author/prompt/session/topic so related documents never cross splits; feedback samples are excluded from the frozen test set and `source` is excluded from the group key.
- Added an evaluation platform (`src/evaluation.py`) with shared metrics, segment breakdowns, threshold calibration and versioned reports under `reports/evaluation/`.
- Added a confidence-aware decision layer (`src/decision.py`) with a three-way verdict, calibrated thresholds from `reports/thresholds.json` and evidence-oriented wording that never claims proof of authorship.
- Added a model registry (`src/model_registry.py`): candidates, production, archive and a compatibility mirror; promotion archives the previous model and installs its thresholds, rollback restores them.
- Added promotion gating (`src/promotion.py`, `tools/promotion_checks.py`, `tools/compare_models.py`, `tools/promote_model.py`, `tools/rollback_model.py`): F1/ROC-AUC/FPR/FNR regression limits, model load, pytest, leakage acknowledgement and API smoke checks; promotion remains an explicit human action.
- Removed `train_model.py --promote`; training now only writes candidates and prints that the candidate is not promoted.
- Moved threshold calibration to the validation split so the frozen test set is never used to choose a decision boundary.
- Added production monitoring (`src/monitoring.py`) and `GET /api/metrics` with feedback rate, disagreement rate, latency, low-confidence rate and hard-case counts.
- Added stage timings (features, classifier, DB write, total), a retention checkbox, decision-based verdicts, threshold scale zones and latency reporting to the UI.
- Added gitignore rules for raw-text artifacts (`data/datasets/`, `data/feedback/`, model registry binaries).
- Added tests for the decision layer, privacy policy, splitter, model registry, promotion gates, monitoring and the v2 API contract (85 tests).

## Performance Optimization

- Reduced normal cold-start work by avoiding an unconditional NLTK resource download.
- Added model-version-aware SHA-256 analysis reuse with a bounded in-process cache and SQLite lookup index.
- Enabled SQLite WAL/busy-timeout settings and structured request timing logs.
- Tuned Render Gunicorn conservatively to one worker and two threads.
- Debounced frontend text statistics without changing prediction or feedback behavior.

## Human Feedback Learning Loop

- Added feedback schema migration with prediction snapshots, text retention, consent, review status, and review timestamps.
- Added feedback validation, duplicate prevention, protected admin review actions, and approved dataset export.
- Added explicit versioned training outputs; pending or unconsented feedback is never included, and promotion is opt-in.
- Added frontend consent preference and user-facing feedback confirmation.

## SaaS Frontend

- Added a responsive teacher dashboard with navigation for analysis, history, settings, documentation, and project information.
- Added local analysis history, theme and density preferences, animation control, drag-and-drop text loading, and future-feature placeholders.
- Added privacy guidance, contact link, accessible result feedback, and responsive mobile navigation without changing backend contracts.

## Phase 2

- Added dataset quality analysis and metadata generation.
- Added deterministic duplicate-aware dataset loading.
- Added baseline, word TF-IDF, character TF-IDF, and combined pipelines.
- Added cross-validation, held-out test evaluation, and model metadata.
- Added SQLite analysis and pending feedback storage.
- Added teacher feedback controls to the result view.
- Added dataset, feature, and API tests.

## Render Deployment Preparation

- Added Render Blueprint configuration and Python runtime declarations.
- Deduplicated the Gunicorn dependency pin.
- Added model metadata compatibility logging and safe load failures.
- Added HTTP 400/503 handling for invalid requests and unavailable predictions.
- Documented SQLite persistence limitations and future `DATABASE_URL` migration.
