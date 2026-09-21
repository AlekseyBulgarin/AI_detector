# Deployment Checklist

## Completed

- Flask application exposes `app` from `app.py`.
- Production startup uses one Gunicorn worker with two threads and explicit logs/timeouts for Render Free.
- `/health` reports model availability for Render health checks.
- Render Blueprint added in `render.yaml`.
- Python 3.11 runtime declared in `runtime.txt` and `.python-version`.
- Dependencies are pinned and Gunicorn has one pinned version.
- NLTK stopwords are downloaded during the Render build.
- Model artifact is no longer ignored by Git.
- Model metadata records the training scikit-learn version.
- Model loading failures are logged and handled without crashing import.
- API prediction failures return HTTP 503.
- Invalid JSON and invalid `text` values return HTTP 400.
- SQLite creates its parent directory automatically.
- Existing tests and feedback workflow remain available.
- Admin review requires `ADMIN_TOKEN` in production; without it, the local admin page is intentionally open for development.

## Missing or External

- Render dashboard deployment has not been run from this environment.
- The GitHub repository must contain `models/model.pkl` before deployment.
- A Render health check must be confirmed after the first deploy.
- SQLite data is stored on the service filesystem and is not durable on the Free plan.

## Fixed

- Empty `runtime.txt`.
- Duplicate Gunicorn requirements.
- Ignored production model artifact.
- Unguarded model deserialization.
- Misleading missing-model probability fallback.
- Malformed JSON API handling.

## Remaining

- Migrate feedback storage to a managed database for durable production data.
- Set a strong `ADMIN_TOKEN` secret before exposing `/admin/feedback`.
- Add the PostgreSQL implementation behind `DATABASE_URL` when a managed database is provisioned.
- Configure a custom domain and production monitoring in Render if needed.

## Database Note

The current application uses SQLite at `data/feedback.db`. Render services have an
ephemeral filesystem by default, so feedback can be lost during redeploys or restarts.
`DATABASE_URL` is reserved for a future PostgreSQL adapter; setting it today does not
change the SQLite implementation.
