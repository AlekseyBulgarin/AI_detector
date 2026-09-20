# Deployment Ready

## Completed

### Repository

- [x] Git changes reviewed before deployment preparation.
- [x] `models/model.pkl` exists and is ready to be tracked.
- [x] `.gitignore` reviewed; the production model is explicitly unignored.
- [x] `README.md` updated with Render instructions.
- [x] `CHANGELOG.md` updated.

### Backend

- [x] Flask app exposes `app`.
- [x] Debug mode is disabled by default.
- [x] Host binds to `0.0.0.0`.
- [x] `PORT` is read from the environment.
- [x] `gunicorn app:app` is configured for Render.
- [x] `/health` reports model availability.
- [x] `/api/check` returns 400, 503, or 200 as appropriate.
- [x] `/api/feedback` returns 400, 404, or 201 as appropriate.

### Machine Learning

- [x] `model.pkl` exists locally and loads successfully.
- [x] `metadata.json` exists.
- [x] Model metadata records the training scikit-learn version.
- [x] Runtime uses the pinned scikit-learn version.
- [x] Model loading failures are logged and handled.
- [x] `MODEL_PATH` can override the default model location.

### Database

- [x] SQLite feedback storage works locally.
- [x] Database directories are created automatically.
- [x] Database access is isolated in `src/database.py`.
- [x] Feedback remains pending and is not used for automatic retraining.
- [x] Render SQLite persistence limitations are documented.

### Render

- [x] `render.yaml` exists.
- [x] Pinned dependency install is configured.
- [x] NLTK stopwords are downloaded during the build.
- [x] `gunicorn app:app` is configured as the start command.
- [x] `/health` is configured as `healthCheckPath`.
- [x] Python 3.11 is declared in `runtime.txt` and `.python-version`.

### Testing

- [x] Unit and API tests pass.
- [x] Dataset and feature tests pass.
- [x] Browser smoke test completed.
- [x] Local homepage, analysis, feedback, and invalid JSON smoke checks pass.

## Remaining Manual Actions

1. Push the committed changes to GitHub.
2. Confirm `models/model.pkl` is included in the pushed commit.
3. Create or update the Render Blueprint from `render.yaml`.
4. Wait for the Render build and confirm the NLTK download succeeds.
5. Confirm the service logs contain successful model loading and startup events.
6. Verify `/`, `/health`, `/api/check`, and `/api/feedback` on the deployed URL.
7. Migrate feedback storage to PostgreSQL before relying on production feedback.

## Render Configuration

- Runtime: Python 3.11
- Build: `pip install -r requirements.txt` plus NLTK stopwords download
- Start: `gunicorn app:app`
- Health check: `/health`
- Model override: `MODEL_PATH`
- Environment mode: `FLASK_ENV`, defaulting to `production`
- Database setting: `DATABASE_URL` is reserved for the future PostgreSQL adapter

## Known Limitations

- The dataset is small and metrics are unstable.
- The current model should be retrained after dataset expansion.
- SQLite is not persistent on Render Free services.
- Gunicorn cannot be executed on Windows because it requires Unix `fcntl`; Render runs Linux.
- The current application recognizes PostgreSQL `DATABASE_URL` but intentionally requires a future PostgreSQL adapter before using it.
- The model artifact is a trusted local pickle and must only be loaded from a trusted repository.
