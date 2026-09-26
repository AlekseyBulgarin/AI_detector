"""Version identifiers attached to every analysis and model artifact.

These are single sources of truth so that prediction logs, dataset manifests,
and model metadata always agree on which pipeline produced a score.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_VERSION = "linguistic-v1+tfidf-v1"

# Filled in when a dataset registry manifest exists; otherwise a static label
# describing the pre-registry raw corpus.
DEFAULT_DATASET_VERSION = "raw-corpus-v0"

DATASET_VERSION_PATH = PROJECT_ROOT / "data" / "manifests" / "current.txt"


def current_dataset_version():
    try:
        value = DATASET_VERSION_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_DATASET_VERSION
    return value or DEFAULT_DATASET_VERSION
