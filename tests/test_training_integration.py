import json
from pathlib import Path

from train_model import load_training_records, parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_defaults_to_original_data_only():
    records = load_training_records(include_feedback=False)

    assert records
    assert {record["source"] for record in records} == {"original"}


def test_training_feedback_is_explicit_and_versioned():
    args = parse_args(["--include-feedback", "--model-version", "phase2-v2", "--promote"])

    assert args.include_feedback is True
    assert args.model_version == "phase2-v2"
    assert args.promote is True


def test_versioned_model_metadata_contains_feedback_provenance():
    for version in ("v1", "v2"):
        assert (PROJECT_ROOT / "models" / f"model_{version}.pkl").exists()
        metadata = json.loads((PROJECT_ROOT / "models" / f"metadata_{version}.json").read_text())
        assert metadata["version"] == f"phase2-{version}"
        assert metadata["dataset_size"] >= metadata["feedback_samples"]
        assert "metrics" in metadata
        assert "created_at" in metadata
