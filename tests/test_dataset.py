from pathlib import Path

from tools.dataset_loader import load_dataset, validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_loader_is_deterministic_and_filters_normalized_duplicates():
    first = load_dataset(PROJECT_ROOT / "data" / "raw")
    second = load_dataset(PROJECT_ROOT / "data" / "raw")

    assert [record["filename"] for record in first] == [record["filename"] for record in second]
    assert len(first) == 35
    assert {record["label"] for record in first} == {"human", "ai"}
    assert validate_dataset(first) == []


def test_metadata_fields_are_available():
    record = load_dataset(PROJECT_ROOT / "data" / "raw")[0]
    assert {
        "id",
        "label",
        "filename",
        "hash",
        "normalized_hash",
        "characters",
        "words",
        "sentences",
        "created_at",
    }.issubset(record)
