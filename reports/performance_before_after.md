# Performance Before and After

Date: 2026-09-21

## Method

Measurements used Python 3.11, the checked-in `models/model.pkl`, Flask's test client, and the same sample text family. Each process loaded the model once. API timings below disable logging output during the microbenchmark so logging I/O does not dominate the comparison. Render network latency and cold-container scheduling were not available locally.

## Results

| Metric | Before | After | Improvement |
| --- | ---: | ---: | --- |
| Cold `import app` | 4,449 ms | 1,934 ms | 56.5% lower startup time |
| Warm direct prediction | 4.26 ms | 5.22 ms unique analysis path | Comparable single-digit latency; extra hash/cache/lookup bookkeeping |
| Repeated direct analysis | Not cached | 0.19 ms | New cache-hit path; avoids model and database work in-process |
| `/api/check` unique, logging disabled | 3.52 ms* | 3.74 ms | Comparable; adds persistent-result lookup |
| `/api/check` repeated, logging disabled | Not cached | 0.30 ms | New cache-hit path |
| Homepage | 0.81 ms | 0.83 ms | No regression within local noise |

`*` The before API measurement stubbed analysis persistence to isolate the old request/model path. The after unique measurement uses the same persistence stub and disables the new logging output, but includes the new cache-key logic. These are microbenchmarks, not production SLA measurements.

## Changes

### Startup and ML

- Kept eager process-level model loading. This is already a singleton and avoids moving the 1.9s load cost into the first prediction.
- Changed NLTK initialization to load the already-built stopword resource directly. The downloader is now a fallback only when the resource is absent.
- Expected improvement: lower cold startup while preserving feature values and prediction logic.
- Risk: a missing local resource still triggers a download; offline environments without the resource will fail as before.

### Repeated analysis

- Added a bounded 256-entry in-process LRU cache keyed by SHA-256 text hash and model version.
- Added an indexed SQLite lookup for the same hash/model pair so separate Gunicorn workers can reuse persisted results.
- Expected improvement: repeated in-process requests avoid model inference and database writes; measured at about 0.19ms direct and 0.30ms through `/api/check`.
- Risk: the in-process cache is per worker and is invalidated naturally by model-version changes or process restarts.

### SQLite

- Enabled WAL mode during initialization, a 5-second busy timeout, and an index on `(text_hash, model_version)`.
- Expected improvement: better read/write concurrency and faster persisted duplicate lookup.
- Risk: SQLite WAL creates sidecar files and remains unsuitable for durable Render Free storage; PostgreSQL migration is still required for production durability.

### Observability and Render

- Added structured request completion timing logs with method, path, status, and duration.
- Configured Render Gunicorn for one worker and two threads with explicit timeout and stdout/stderr logs.
- Expected improvement: measurable production latency and modest concurrent request handling without blindly multiplying model memory.
- Risk: one worker limits CPU parallelism; adding workers later must be based on Render memory and load measurements.

### Frontend

- Debounced text statistics updates by 80ms.
- Preserved loading state, feedback requests, navigation, history, and all existing visual behavior.
- Expected improvement: less main-thread work during rapid typing or large paste operations.
- Risk: displayed counts and short-text warning can lag typing by at most 80ms; submit validation remains immediate.

## Not Changed Deliberately

- No model architecture or feature semantics were changed.
- No lazy loader was introduced because it would worsen first-prediction latency.
- No response-compression dependency was added because local HTML/API responses are small and Render's edge behavior was not measured here.
- No extra Gunicorn worker was selected because Render Free memory/CPU limits make that a risky default for a model-backed process.
- No long-lived static cache headers were added because current asset URLs are not fingerprinted.
