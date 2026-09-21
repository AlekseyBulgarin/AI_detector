# Performance Audit

Date: 2026-09-21

## Scope

This audit covers the Flask application, the loaded scikit-learn pipeline, SQLite feedback storage, the Jinja/CSS/JavaScript frontend, and the Render/Gunicorn startup configuration. Measurements were taken locally with Python 3.11 and the checked-in model artifact. They are not representative of Render network latency or cold-container scheduling.

## Baseline Measurements

| Metric | Baseline | Notes |
| --- | ---: | --- |
| Cold `import app` | 4,449 ms | Includes Flask/scikit-learn/NLTK imports, stopword initialization, SQLite initialization, and `joblib.load()` |
| Warm direct prediction | 4.26 ms average | 30 predictions, combined pipeline already loaded |
| Warm `/api/check` | 3.52 ms average | 20 requests with analysis persistence stubbed to isolate request/model cost |
| Warm homepage | 0.81 ms average | 20 Flask test-client requests |
| Model artifact | 636,782 bytes | Loaded once at application import |
| HTML | 19,165 bytes | Before HTTP transfer encoding |
| CSS | 25,490 bytes | One stylesheet |
| JavaScript | 13,357 bytes | One script, no bundler/minifier |

The prediction target of less than 500ms is already met locally by a wide margin. The practical performance problem is cold startup, especially on a Render Free instance that may sleep and wake.

## Checklist

### Already Optimized

- [x] `joblib.load()` runs once during module initialization; requests reuse the global model.
- [x] The selected combined pipeline performs one `predict_proba()` call per prediction.
- [x] NLTK stopwords are materialized into a module-level set instead of loaded for every request.
- [x] Flask uses Jinja template rendering without per-request filesystem model loading.
- [x] Input validation rejects short payloads before model inference.
- [x] The frontend uses one CSS file and one JavaScript file, with no per-keystroke network requests.
- [x] The submit button is disabled while the form is being submitted.
- [x] Gunicorn is used in production instead of Flask's development server.

### Needs Improvement

- [x] Cold startup: importing scikit-learn and NLTK plus model deserialization dominates startup time. The unconditional NLTK downloader was removed from the normal path; the resource is still downloaded only when missing.
- [x] Repeated text: identical text is recomputed and inserted as a new analysis instead of reusing the existing result. A bounded cache and model-version-aware SQLite lookup now reuse results.
- [x] SQLite concurrency: each write opens a new connection and WAL/busy-timeout settings are not configured. WAL, busy timeout, and a lookup index are now enabled.
- [x] Observability: request timing is not recorded, so production latency cannot be compared with local timings. Structured completion timing logs are now emitted.
- [x] Gunicorn tuning: the current command leaves worker/thread policy implicit and does not document the Free-plan tradeoff. Render now uses one worker and two threads explicitly.
- [x] Frontend input work: word/character counts run synchronously for every input event. Updates are coalesced with an 80ms debounce.
- [ ] Frontend delivery: Google Fonts and the Lucide CDN are external render dependencies; assets are not minified or fingerprinted.
- [ ] Cache headers: static assets do not have an explicit long-lived cache policy in the application.

The static cache-header item remains intentionally open. A long-lived cache is unsafe without versioned asset URLs; adding it is a separate change.

### Risky Changes

- [ ] Replacing NLTK stopwords with a hand-maintained list could change model features and predictions.
- [ ] Lazy model loading would move cold-start cost into the first prediction and make the latency target worse for the first user.
- [ ] Multiple Gunicorn workers can multiply Python/scikit-learn memory on Render Free; `--preload` changes failure and fork behavior.
- [ ] Compressing responses with a new middleware/dependency can alter streaming/error behavior and build time.
- [ ] Persistent cross-worker caching requires database schema/API semantics and invalidation by model version.
- [ ] Removing external icons or fonts changes the established visual interface.

## Findings

| Current bottleneck | Impact | Priority | Recommended fix |
| --- | --- | --- | --- |
| Cold imports plus model initialization | Multi-second wake/start latency | P0 | Keep eager one-time loading for first-request latency; remove only redundant runtime resource downloads and measure startup after each change |
| No repeated-analysis cache | Wastes inference and creates duplicate analysis records | P1 | Add a bounded in-process cache keyed by model version and SHA-256 text hash, with SQLite lookup for cross-worker reuse |
| One SQLite connection per operation, default journaling | Small overhead and poorer write concurrency | P1 | Enable WAL and a busy timeout; add an index for text hash/model version; keep the existing connection abstraction |
| No request timing | Cannot validate production performance | P1 | Add lightweight `before_request`/`after_request` structured timing logs |
| Implicit Gunicorn defaults | Conservative concurrency but undocumented behavior | P1 | Use one worker and two threads on Free by default; allow an environment override rather than blindly adding workers |
| Synchronous text statistics | Minor main-thread work on long pasted text | P2 | Coalesce input updates with `requestAnimationFrame`; preserve immediate validation on submit |
| External fonts/icons | First-paint dependency on third parties | P2 | Consider self-hosting or system-font fallback in a separate visual change; do not remove blindly |

## Architecture Notes

- The current global `model` is already a process-local singleton. Introducing a second loader would add indirection without reducing work.
- The combined pipeline uses `TextFeatureFrame`, TF-IDF vectorizers, and linguistic features. Feature extraction is already warm and measured at milliseconds, so model simplification is not justified by latency.
- `DATABASE_URL` deliberately reserves a future PostgreSQL adapter. This audit does not change that contract or attempt a partial migration.
- Render Free has limited CPU and memory. More workers improve isolation/concurrency but duplicate interpreter/model state unless copy-on-write is effective; one worker with a small thread pool is the conservative default.

## Validation Plan

After implementation, compare the same local benchmark for cold import, warm prediction, `/api/check`, and homepage. Also run the full pytest suite, Python compilation, JavaScript syntax validation, API smoke tests, and frontend browser smoke tests. Any claimed improvement must be based on the before/after report, not intuition.
