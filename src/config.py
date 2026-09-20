import os
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FLASK_ENV = os.environ.get("FLASK_ENV", "production").lower()
DATABASE_URL = os.environ.get("DATABASE_URL", "")


def resolve_path(value, default):
    path = Path(value) if value else Path(default)
    return path if path.is_absolute() else PROJECT_ROOT / path


MODEL_PATH = resolve_path(
    os.environ.get("MODEL_PATH"),
    PROJECT_ROOT / "models" / "model.pkl",
)
MODEL_METADATA_PATH = MODEL_PATH.parent / "metadata.json"


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
