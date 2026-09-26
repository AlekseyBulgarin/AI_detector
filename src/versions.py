"""Version identifiers attached to every analysis and model artifact.

These are single sources of truth so that prediction logs, dataset manifests,
and model metadata always agree on which pipeline produced a score.
"""

FEATURE_VERSION = "linguistic-v1+tfidf-v1"

# Static label describing the pre-registry raw corpus that the current
# production model was trained on. Registry snapshots (``data/manifests/``)
# carry their own version and are passed explicitly via ``--dataset``.
DEFAULT_DATASET_VERSION = "raw-corpus-v0"
