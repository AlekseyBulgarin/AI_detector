"""Deterministic, leakage-aware dataset splitting.

Naive random splitting is unsafe when related documents exist: the same author,
prompt, generation session or topic can land in both train and test, which makes
metrics optimistic. Groups are therefore kept intact, and feedback-derived
samples are never allowed into the frozen test set.
"""

import hashlib
import random
from collections import defaultdict

DEFAULT_SEED = 42
# ``source`` is deliberately absent: the whole raw corpus shares one source
# value, which would collapse every document into a single group and silently
# disable the group-aware split. Source balance is a separate quality check.
GROUP_PREFERENCE = ("author", "prompt", "generation_session", "topic")
SPLIT_NAMES = ("train", "validation", "frozen_test")


def group_key(record):
    """Best available grouping key for a record."""
    for field in GROUP_PREFERENCE:
        value = record.get(field)
        if value:
            return f"{field}:{value}"
    normalized = record.get("normalized_hash")
    if normalized:
        return f"norm:{normalized}"
    return f"id:{record.get('id')}"


def is_feedback_record(record):
    return str(record.get("source", "")).startswith("approved_feedback") or bool(
        record.get("from_feedback")
    )


def _targets(counts, proportions):
    total = sum(counts.values())
    return {name: total * proportions[name] for name in SPLIT_NAMES}


def group_split(
    records,
    proportions=None,
    seed=DEFAULT_SEED,
    allow_feedback_in_test=False,
):
    """Assign every record to train / validation / frozen_test.

    Returns a dict ``{record_id: split_name}`` plus a summary. Related records
    always share one split.
    """
    proportions = dict(proportions or {"train": 0.7, "validation": 0.15, "frozen_test": 0.15})
    missing = [name for name in SPLIT_NAMES if name not in proportions]
    if missing:
        raise ValueError(f"Split proportions missing: {', '.join(missing)}")
    if abs(sum(proportions.values()) - 1.0) > 1e-6:
        raise ValueError("Split proportions must sum to 1")

    groups = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)

    rng = random.Random(seed)
    ordered = sorted(groups.items(), key=lambda item: item[0])
    rng.shuffle(ordered)

    assigned = {}
    placed = {name: {"human": 0, "ai": 0, "other": 0} for name in SPLIT_NAMES}
    totals = {"human": 0, "ai": 0, "other": 0}
    for record in records:
        label = record.get("label", "other")
        totals[label if label in totals else "other"] += 1
    target = {
        name: {label: totals[label] * proportions[name] for label in totals}
        for name in SPLIT_NAMES
    }

    def choose(record_list, locked_split=None):
        if locked_split:
            return locked_split
        per_label = defaultdict(int)
        for record in record_list:
            label = record.get("label", "other")
            per_label[label if label in totals else "other"] += 1
        best_split = None
        best_score = None
        for name in SPLIT_NAMES:
            score = 0.0
            for label, weight in per_label.items():
                need = target[name][label] - placed[name][label]
                if need > 0:
                    score += min(weight, need)
                else:
                    score += need * 0.5
            if best_score is None or score > best_score:
                best_score = score
                best_split = name
        return best_split

    for _, record_list in ordered:
        contains_feedback = any(is_feedback_record(record) for record in record_list)
        locked = None
        if contains_feedback and not allow_feedback_in_test:
            locked = choose(record_list, locked_split=None)
            # Feedback groups may use train or validation, never frozen test.
            if locked == "frozen_test":
                locked = "train"
        split = choose(record_list, locked) if locked is None else locked
        for record in record_list:
            assigned[record["id"]] = split
            label = record.get("label", "other")
            placed[split][label if label in totals else "other"] += 1

    summary = {
        "counts": {
            name: sum(1 for value in assigned.values() if value == name)
            for name in SPLIT_NAMES
        },
        "by_label": {
            name: {
                label: sum(
                    1
                    for record in records
                    if assigned.get(record["id"]) == name
                    and record.get("label") == label
                )
                for label in ("human", "ai")
            }
            for name in SPLIT_NAMES
        },
        "group_count": len(groups),
        "seed": seed,
        "proportions": proportions,
        "feedback_in_frozen_test": False,
    }
    return assigned, summary


def split_manifest(records, assignment, summary, dataset_version):
    """Build a reproducible, hashable description of a split."""
    payload = {
        "dataset_version": dataset_version,
        "algorithm": "group_aware_v1",
        "summary": summary,
        "assignments": {
            record["id"]: {
                "split": assignment[record["id"]],
                "group": group_key(record),
                "hash": record.get("hash", ""),
            }
            for record in records
        },
    }
    digest = hashlib.sha256(
        repr(sorted(payload["assignments"].items())).encode("utf-8")
    ).hexdigest()
    payload["split_digest"] = digest
    return payload
