import os
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLASK_ENV = os.environ.get("FLASK_ENV", "production").lower()
DATABASE_URL = os.environ.get("DATABASE_URL", "")


def resolve_path(value, default):
    path = Path(value) if value else Path(default)
    return path if path.is_absolute() else PROJECT_ROOT / path


MODELS_ROOT = PROJECT_ROOT / "models"
PRODUCTION_DIR = MODELS_ROOT / "production"
CANDIDATES_DIR = MODELS_ROOT / "candidates"
ARCHIVE_DIR = MODELS_ROOT / "archive"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _default_model_path():
    """Prefer the model registry, fall back to the legacy flat artifact."""
    registry_model = PRODUCTION_DIR / "model.pkl"
    if registry_model.exists():
        return registry_model
    return MODELS_ROOT / "model.pkl"


MODEL_PATH = resolve_path(os.environ.get("MODEL_PATH"), _default_model_path())
MODEL_METADATA_PATH = (
    MODEL_PATH.parent / "metadata.json"
    if MODEL_PATH.parent.name == "production"
    else MODELS_ROOT / "metadata.json"
)

THRESHOLDS_PATH = REPORTS_DIR / "thresholds.json"
EVALUATION_DIR = REPORTS_DIR / "evaluation"
COMPARISON_DIR = REPORTS_DIR / "comparison"

# Static assets are served without fingerprinted URLs, so only a short cache
# lifetime is safe.
STATIC_CACHE_SECONDS = int(os.environ.get("STATIC_CACHE_SECONDS", "3600"))


def sqlite_path_from_url(database_url=DATABASE_URL):
    """Return a SQLite path or None for the future PostgreSQL backend."""
    if not database_url:
        return PROJECT_ROOT / "data" / "feedback.db"
    if database_url == ":memory:":
        return database_url
    if database_url.startswith("sqlite:///"):
        parsed = urlparse(database_url)
        return Path(unquote(parsed.path))
    if database_url.startswith(("postgresql://", "postgres://")):
        return None
    raise ValueError("DATABASE_URL must be a SQLite or PostgreSQL URL")
