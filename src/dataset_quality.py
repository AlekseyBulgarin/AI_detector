"""Dataset quality gates and bias analysis.

Nothing here deletes data. Gates report what is wrong so a human can decide
whether a dataset version is usable for training.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.feedback_validator import MIN_TEXT_LENGTH, near_duplicate_ratio, normalize_text
from tools.dataset_loader import load_dataset, validate_dataset


MIN_TRAINING_LENGTH = MIN_TEXT_LENGTH
METADATA_FIELDS = ("source", "topic", "author", "prompt", "generation_session")
REPORT_JSON = Path("reports/dataset_quality.json")
REPORT_MD = Path("reports/dataset_quality.md")

LENGTH_LEAKAGE_RATIO = 1.5
STANDARDIZED_EFFECT_SIZE = 0.8
NEAR_DUPLICATE_THRESHOLD = 0.85
MISSING_METADATA_THRESHOLD = 0.5

BLOCKING_GATES = {
    "valid_label",
    "sufficient_length",
    "consent",
    "approved_review_status",
    "exact_duplicate",
    "near_duplicate",
}


def _stats(values):
    values = list(values)
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
        "median": statistics.median(values),
        "std": round(statistics.pstdev(values), 2) if len(values) > 1 else 0,
    }


def _gate(name, failed, total, details=None, blocking=None):
    return {
        "gate": name,
        "passed": failed == 0,
        "checked": total,
        "failed": failed,
        "blocking": (name in BLOCKING_GATES) if blocking is None else blocking,
        "details": details or [],
    }


def _is_feedback(record):
    return str(record.get("source", "")).startswith("approved_feedback") or bool(
        record.get("from_feedback")
    )


def run_quality_gates(records, read_errors=None):
    """Run every dataset gate and return a structured report."""
    read_errors = list(read_errors or [])
    total = len(records)
    results = []

    results.append(
        _gate(
            "valid_encoding",
            len(read_errors),
            total,
            read_errors,
            blocking=True,
        )
    )
    results.append(
        _gate(
            "valid_label",
            sum(
                1
                for record in records
                if record.get("label") not in {"human", "ai", "ai_assisted"}
            ),
            total,
        )
    )
    results.append(
        _gate(
            "sufficient_length",
            sum(
                1
                for record in records
                if len(str(record.get("text", "")).strip()) < MIN_TRAINING_LENGTH
            ),
            total,
        )
    )

    feedback_records = [record for record in records if _is_feedback(record)]
    consent_failures = [
        record.get("id")
        for record in feedback_records
        if not record.get("consent", True)
    ]
    results.append(
        _gate(
            "consent",
            len(consent_failures),
            len(feedback_records),
            consent_failures,
        )
    )
    status_failures = [
        record.get("id")
        for record in feedback_records
        if record.get("review_status") != "approved"
    ]
    results.append(
        _gate(
            "approved_review_status",
            len(status_failures),
            len(feedback_records),
            status_failures,
        )
    )

    exact_groups = defaultdict(list)
    for record in records:
        exact_groups[record.get("hash", "")].append(record.get("id"))
    exact_duplicates = [ids for ids in exact_groups.values() if len(ids) > 1]
    results.append(
        _gate("exact_duplicate", len(exact_duplicates), total, exact_duplicates)
    )

    near_duplicates = []
    seen = []
    for record in records:
        text = str(record.get("text", ""))
        for other_id, other_text in seen:
            if near_duplicate_ratio(text, other_text) >= NEAR_DUPLICATE_THRESHOLD:
                near_duplicates.append([record.get("id"), other_id])
                break
        seen.append((record.get("id"), text))
    results.append(
        _gate("near_duplicate", len(near_duplicates), total, near_duplicates)
    )

    missing_metadata = [
        record.get("id")
        for record in records
        if not any(record.get(field) for field in METADATA_FIELDS)
    ]
    results.append(
        _gate(
            "source_metadata",
            len(missing_metadata),
            total,
            missing_metadata[:20],
            blocking=False,
        )
    )

    blocking_failures = [
        item["gate"] for item in results if item["blocking"] and not item["passed"]
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": total,
        "feedback_samples": len(feedback_records),
        "gates": results,
        "blocking_failures": blocking_failures,
        "passed": not blocking_failures,
    }


def _class_values(records, field):
    return {label: [record.get(field, 0) for record in records if record["label"] == label] for label in ("human", "ai")}


def _effect_size(left, right):
    if not left or not right:
        return 0.0
    pooled = statistics.pstdev(left + right)
    if pooled == 0:
        return 0.0
    return round((statistics.mean(right) - statistics.mean(left)) / pooled, 3)


def _metadata_leakage(records, field, warning):
    present = [record for record in records if record.get(field)]
    if not present:
        return {
            "warning": warning,
            "reason": f"{field} metadata unavailable",
            "coverage": 0.0,
            "triggered": len(records) >= 3,
        }
    coverage = len(present) / len(records)
    mapping = defaultdict(set)
    for record in present:
        mapping[record[field]].add(record["label"])
    leaky = sorted(key for key, labels in mapping.items() if len(labels) == 1)
    return {
        "warning": warning,
        "reason": f"{field} perfectly predicts label"
        if leaky
        else f"{field} values are mixed across labels",
        "coverage": round(coverage, 3),
        "triggered": bool(leaky) or coverage < 1 - MISSING_METADATA_THRESHOLD,
        "leaky_values": leaky[:20],
        "unmapped_records": len(records) - len(present),
    }


def analyze_bias(records):
    """Compare human and AI distributions and emit leakage warnings."""
    by_field = {}
    for field in ("characters", "words", "sentences"):
        grouped = _class_values(records, field)
        by_field[field] = {
            "human": _stats(grouped["human"]),
            "ai": _stats(grouped["ai"]),
            "ratio": round(
                (statistics.mean(grouped["ai"]) / statistics.mean(grouped["human"]))
                if grouped["human"] and statistics.mean(grouped["human"])
                else 0,
                3,
            ),
            "effect_size_d": _effect_size(grouped["human"], grouped["ai"]),
        }

    warnings = []
    length = by_field["words"]
    if length["ratio"] >= LENGTH_LEAKAGE_RATIO or length["ratio"] <= (
        1 / LENGTH_LEAKAGE_RATIO
    ) or abs(length["effect_size_d"]) >= STANDARDIZED_EFFECT_SIZE:
        warnings.append(
            {
                "warning": "HIGH LENGTH LEAKAGE",
                "detail": (
                    f"AI mean words {length['ai']['mean']} vs human "
                    f"{length['human']['mean']} (ratio {length['ratio']}, "
                    f"Cohen's d {length['effect_size_d']}). The detector may "
                    "learn 'long text = AI'."
                ),
            }
        )

    for field, warning in (("source", "SOURCE LEAKAGE"), ("topic", "TOPIC LEAKAGE")):
        result = _metadata_leakage(records, field, warning)
        if result["triggered"]:
            warnings.append(
                {
                    "warning": warning,
                    "detail": f"{result['reason']} (coverage {result['coverage']}).",
                }
            )

    normalized = defaultdict(set)
    for record in records:
        normalized[record.get("normalized_hash")].add(record["label"])
    cross_class = [key for key, labels in normalized.items() if len(labels) > 1]
    intra_duplicates = sum(
        1 for count in Counter(record.get("normalized_hash") for record in records).values() if count > 1
    )
    if cross_class:
        warnings.append(
            {
                "warning": "DUPLICATE LEAKAGE",
                "detail": f"{len(cross_class)} normalized hashes appear in both classes.",
            }
        )
    elif intra_duplicates:
        warnings.append(
            {
                "warning": "DUPLICATE LEAKAGE",
                "detail": f"{intra_duplicates} normalized hashes appear more than once.",
            }
        )

    return {
        "distributions": by_field,
        "class_balance": {
            label: sum(1 for record in records if record["label"] == label)
            for label in ("human", "ai")
        },
        "style": {
            "sentence_length": {
                label: [
                    round(
                        record["characters"] / max(record["sentences"], 1),
                        2,
                    )
                    for record in records
                    if record["label"] == label
                ]
                for label in ("human", "ai")
            }
        },
        "metadata_coverage": {
            field: round(
                sum(1 for record in records if record.get(field)) / max(len(records), 1),
                3,
            )
            for field in METADATA_FIELDS
        },
        "warnings": warnings,
    }


def duplicate_report(records):
    exact = defaultdict(list)
    normalized = defaultdict(list)
    for record in records:
        exact[record.get("hash")].append(record.get("filename", record.get("id")))
        normalized[record.get("normalized_hash")].append(
            record.get("filename", record.get("id"))
        )
    return {
        "exact": [files for files in exact.values() if len(files) > 1],
        "normalized": [files for files in normalized.values() if len(files) > 1],
    }


def build_quality_report(records, read_errors=None):
    """Assemble the full quality report used for dataset version approval."""
    quality = run_quality_gates(records, read_errors)
    bias = analyze_bias(records)
    validation_errors = validate_dataset(records) if records else ["Dataset is empty"]
    return {
        "total_samples": len(records),
        "samples_per_class": {
            label: sum(1 for record in records if record["label"] == label)
            for label in ("human", "ai")
        },
        "statistics": {
            field: _stats(record.get(field, 0) for record in records)
            for field in ("characters", "words", "sentences")
        },
        "duplicates": duplicate_report(records),
        "validation_errors": validation_errors,
        "quality_gates": quality,
        "bias_analysis": bias,
        "warnings": [item["warning"] for item in bias["warnings"]],
    }


def render_markdown(report):
    gates = report["quality_gates"]
    lines = [
        "# Dataset Quality Report",
        "",
        f"Generated: {gates['generated_at']}",
        "",
        f"**Overall status:** {'PASS' if gates['passed'] else 'FAIL'}",
        "",
        "## Summary",
        "",
        f"- Total samples: {report['total_samples']}",
        f"- Human samples: {report['samples_per_class'].get('human', 0)}",
        f"- AI samples: {report['samples_per_class'].get('ai', 0)}",
        f"- Feedback samples: {gates['feedback_samples']}",
        "",
        "## Quality Gates",
        "",
        "| Gate | Status | Checked | Failed | Blocking |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for gate in gates["gates"]:
        lines.append(
            f"| {gate['gate']} | {'PASS' if gate['passed'] else 'FAIL'} | "
            f"{gate['checked']} | {gate['failed']} | "
            f"{'yes' if gate['blocking'] else 'no'} |"
        )
    if gates["blocking_failures"]:
        lines.extend(["", f"Blocking failures: {', '.join(gates['blocking_failures'])}"])

    lines.extend(["", "## Bias And Leakage", ""])
    if report["bias_analysis"]["warnings"]:
        for warning in report["bias_analysis"]["warnings"]:
            lines.append(f"- **{warning['warning']}** - {warning['detail']}")
    else:
        lines.append("- No leakage warnings detected.")

    lines.extend(["", "## Length Distributions", ""])
    lines.append("| Field | Human mean | AI mean | Ratio | Cohen's d |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for field, values in report["bias_analysis"]["distributions"].items():
        lines.append(
            f"| {field} | {values['human']['mean']} | {values['ai']['mean']} | "
            f"{values['ratio']} | {values['effect_size_d']} |"
        )

    lines.extend(["", "## Duplicates", ""])
    exact = report["duplicates"]["exact"]
    normalized = report["duplicates"]["normalized"]
    lines.append(f"- Exact duplicate groups: {len(exact)}")
    lines.append(f"- Normalized duplicate groups: {len(normalized)}")
    lines.append("")
    return "\n".join(lines)


def write_reports(report, json_path=REPORT_JSON, md_path=REPORT_MD):
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def analyze(data_root="data/raw", include_duplicates=True, json_path=REPORT_JSON, md_path=REPORT_MD):
    records = load_dataset(data_root, include_duplicates=include_duplicates)
    report = build_quality_report(records)
    write_reports(report, json_path, md_path)
    return report


if __name__ == "__main__":
    result = analyze()
    print(json.dumps(result["quality_gates"], ensure_ascii=False, indent=2))
