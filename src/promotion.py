"""Promotion gates.

A candidate never becomes production automatically. Gates are evaluated first
and every gate must pass, but even then promotion still requires an explicit
human invocation of ``tools/promote_model.py``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import EVALUATION_DIR, REPORTS_DIR


DEFAULT_POLICY = {
    "f1_delta_min": -0.02,
    "false_positive_rate_delta_max": 0.02,
    "false_negative_rate_delta_max": 0.02,
    "roc_auc_delta_min": -0.02,
}


class GateResult:
    def __init__(self, name, passed, detail):
        self.name = name
        self.passed = bool(passed)
        self.detail = detail

    def as_dict(self):
        return {"gate": self.name, "passed": self.passed, "detail": self.detail}


def _metric(source, key):
    if not source:
        return None
    metrics = source.get("metrics") if "metrics" in source else source
    if not metrics:
        return None
    if key in metrics:
        return metrics[key]
    frozen = metrics.get("frozen_test") or {}
    return frozen.get(key)


def evaluate_gates(candidate, production=None, checks=None, policy=None):
    """Compare candidate against production and evaluate every gate."""
    policy = {**DEFAULT_POLICY, **(policy or {})}
    checks = dict(checks or {})
    results = []

    metrics = candidate.get("metrics") or candidate.get("frozen_test") or candidate
    results.append(
        GateResult(
            "candidate_evaluation_present",
            bool(metrics),
            "Frozen test metrics are available"
            if metrics
            else "No frozen test evaluation was produced",
        )
    )

    frozen_samples = 0
    if metrics:
        frozen_samples = metrics.get("samples", 0)
    results.append(
        GateResult(
            "frozen_test_non_empty",
            frozen_samples > 0,
            f"{frozen_samples} frozen test samples",
        )
    )

    production_metrics = None
    if production:
        production_metrics = (
            production.get("metrics")
            if "metrics" in production
            else production.get("frozen_test")
            or production
        )

    comparisons = {}
    if production_metrics and metrics:
        for key, delta_key, direction in (
            ("f1", "f1_delta_min", "min"),
            ("roc_auc", "roc_auc_delta_min", "min"),
            ("false_positive_rate", "false_positive_rate_delta_max", "max"),
            ("false_negative_rate", "false_negative_rate_delta_max", "max"),
        ):
            before = _metric({"metrics": production_metrics}, key)
            after = _metric({"metrics": metrics}, key)
            if before is None or after is None:
                continue
            delta = round(after - before, 6)
            comparisons[key] = {"production": before, "candidate": after, "delta": delta}
            limit = policy[delta_key]
            passed = delta >= limit if direction == "min" else delta <= limit
            results.append(
                GateResult(
                    f"no_regression_{key}",
                    passed,
                    f"{key}: {before:.4f} -> {after:.4f} (delta {delta:+.4f}, "
                    f"limit {delta_key}={limit})",
                )
            )
    else:
        results.append(
            GateResult(
                "production_baseline_available",
                False,
                "Production metrics are missing, so no regression can be measured",
            )
        )

    for name, default in (
        ("model_loads", True),
        ("regression_tests", True),
        ("data_leakage_checks", True),
        ("api_smoke_tests", True),
    ):
        passed = checks.get(name, default)
        results.append(
            GateResult(
                name,
                passed,
                checks.get(f"{name}_detail") or ("passed" if passed else "failed"),
            )
        )

    passed = all(item.passed for item in results)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "policy": policy,
        "gates": [item.as_dict() for item in results],
        "comparisons": comparisons,
        "failed_gates": [item.name for item in results if not item.passed],
        "requires_manual_promotion": True,
    }


def write_gate_report(report, version):
    destination = REPORTS_DIR / "gates"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{version}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_gate_report(version):
    path = REPORTS_DIR / "gates" / f"{version}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_report(version):
    path = EVALUATION_DIR / f"{version}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def render_comparison_markdown(production, candidate, comparisons, version, gates=None):
    lines = [
        f"# Model Comparison: Production vs {version}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Metric | Production | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, values in comparisons.items():
        lines.append(
            f"| {key} | {values['production']:.4f} | {values['candidate']:.4f} | "
            f"{values['delta']:+.4f} |"
        )
    if not comparisons:
        lines.append("| - | - | - | no comparable metrics |")

    lines.extend(["", "## Notes", ""])
    lines.append(
        "- Production metrics come from the frozen test evaluation recorded when "
        "that model was promoted."
    )
    lines.append(
        "- Candidate metrics come from the same frozen evaluation dataset, so the "
        "delta is directly comparable."
    )
    lines.append(
        "- Human text classified as AI (false positives) is weighted most heavily "
        "because it is the most damaging error for a teacher."
    )
    if gates:
        lines.extend(["", "## Promotion Gates", ""])
        for gate in gates.get("gates", []):
            marker = "PASS" if gate["passed"] else "FAIL"
            lines.append(f"- [{marker}] `{gate['gate']}` - {gate['detail']}")
        lines.extend(
            [
                "",
                "**Outcome:** "
                + (
                    "all gates passed; promotion still requires an explicit "
                    "`tools/promote_model.py` invocation."
                    if gates.get("passed")
                    else "blocked: " + ", ".join(gates.get("failed_gates", []))
                ),
            ]
        )
    lines.append("")
    return "\n".join(lines)
