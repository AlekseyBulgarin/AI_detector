# Changelog

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
