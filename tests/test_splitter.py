"""Leakage-aware splitting guarantees."""

import pytest

from src.splitter import (
    GROUP_PREFERENCE,
    group_key,
    group_split,
    is_feedback_record,
    split_manifest,
)


def _records(count, **fields):
    records = []
    for index in range(count):
        record = {"id": f"r{index}", "label": "ai" if index % 2 else "human"}
        record.update(fields)
        records.append(record)
    return records


def test_group_preference_excludes_source():
    # The whole raw corpus shares one ``source`` value: including it would fold
    # every document into a single group and silently disable group splitting.
    assert "source" not in GROUP_PREFERENCE
    assert GROUP_PREFERENCE == ("author", "prompt", "generation_session", "topic")


def test_group_key_ignores_source_but_keeps_topic():
    first = {"id": "a", "topic": "climate", "source": "raw-corpus"}
    second = {"id": "b", "topic": "climate", "source": "approved_feedback-v0"}

    assert group_key(first) == group_key(second)
    assert group_key(first) == "topic:climate"


def test_group_key_falls_back_to_normalized_hash():
    record = {"id": "a", "normalized_hash": "abc"}

    assert group_key(record) == "norm:abc"


def test_feedback_records_are_detected():
    assert is_feedback_record({"source": "approved_feedback-v0"})
    assert is_feedback_record({"from_feedback": True})
    assert not is_feedback_record({"source": "raw"})


def test_related_records_never_cross_splits():
    records = []
    for index in range(12):
        records.append(
            {
                "id": f"r{index}",
                "label": "ai" if index % 2 else "human",
                "topic": f"topic-{index % 4}",
            }
        )

    assignment, summary = group_split(records, seed=7)

    groups = {}
    for record in records:
        groups.setdefault(group_key(record), set()).add(assignment[record["id"]])

    assert all(len(splits) == 1 for splits in groups.values())
    assert summary["group_count"] == 4
    assert sum(summary["counts"].values()) == len(records)


def test_feedback_is_never_placed_in_the_frozen_test_set():
    records = [
        {
            "id": f"fb{index}",
            "label": "human",
            "source": "approved_feedback-v0",
        }
        for index in range(12)
    ]

    assignment, summary = group_split(
        records, proportions={"train": 0.2, "validation": 0.2, "frozen_test": 0.6}
    )

    assert set(assignment.values()) <= {"train", "validation"}
    assert summary["counts"]["frozen_test"] == 0
    assert summary["feedback_in_frozen_test"] is False


def test_split_rejects_invalid_proportions():
    with pytest.raises(ValueError, match="sum to 1"):
        group_split(
            _records(3),
            proportions={"train": 0.5, "validation": 0.4, "frozen_test": 0.4},
        )
    with pytest.raises(ValueError, match="missing"):
        group_split(_records(3), proportions={"train": 0.5, "validation": 0.5})


def test_split_manifest_is_reproducible():
    records = _records(9, topic="shared")

    assignment, summary = group_split(records, seed=1)
    first = split_manifest(records, assignment, summary, "v1")
    second = split_manifest(records, assignment, summary, "v1")

    assert first["split_digest"] == second["split_digest"]
    assert first["algorithm"] == "group_aware_v1"
    assert set(first["assignments"]) == {record["id"] for record in records}
