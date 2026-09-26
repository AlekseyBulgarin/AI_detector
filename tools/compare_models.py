"""Compare production against a candidate and write a human-readable report.

Usage:
    python tools/compare_models.py --version v2
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import load_evaluation  # noqa: E402
from src.model_registry import get_production  # noqa: E402
from src.promotion import (  # noqa: E402
    evaluate_gates,
    load_gate_report,
    render_comparison_markdown,
    write_gate_report,
)

REPORT_DIR = PROJECT_ROOT / "reports"


def _metrics_from(report):
    if not report:
        return None
    return report.get("metrics") or report.get("frozen_test")


def compare(version, run_gates=True, acknowledge_leakage=False):
    production = get_production()
    try:
        production_report = load_evaluation(production["version"])
    except FileNotFoundError:
        production_report = None
    candidate_report = load_evaluation(version)

    production_metrics = _metrics_from(production_report)
    candidate_metrics = _metrics_from(candidate_report)

    gates = None
    if run_gates:
        from tools.promotion_checks import build_checks

        checks = build_checks(
            version,
            candidate_report=candidate_report,
            acknowledge_leakage=acknowledge_leakage,
        )
        gates = evaluate_gates(
            {"metrics": candidate_metrics},
            production={"metrics": production_metrics},
            checks=checks,
        )
        write_gate_report(gates, version)

    comparisons = {}
    if production_metrics and candidate_metrics:
        for key in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "false_positive_rate",
            "false_negative_rate",
        ):
            before = production_metrics.get(key)
            after = candidate_metrics.get(key)
            if before is None or after is None:
                continue
            comparisons[key] = {
                "production": before,
                "candidate": after,
                "delta": round(after - before, 6),
            }
        if production_metrics.get("latency_ms") is not None:
            comparisons["latency_ms"] = {
                "production": production_metrics["latency_ms"],
                "candidate": candidate_metrics.get("latency_ms"),
                "delta": None,
            }

    markdown = render_comparison_markdown(
        production_metrics, candidate_metrics, comparisons, version, gates
    )
    path = REPORT_DIR / f"model_comparison_{version}.md"
    path.write_text(markdown, encoding="utf-8")
    return path, comparisons, gates


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare production and candidate")
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--acknowledge-leakage",
        action="store_true",
        help="Record explicit acknowledgement of reported dataset leakage warnings",
    )
    parser.add_argument("--no-gates", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        path, comparisons, gates = compare(
            args.version,
            run_gates=not args.no_gates,
            acknowledge_leakage=args.acknowledge_leakage,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Comparison written: {path}")
    for key, values in comparisons.items():
        print(f"  {key}: {values['production']:.4f} -> {values['candidate']:.4f}")
    if gates is not None:
        print(f"Promotion gates: {'PASS' if gates['passed'] else 'FAIL'}")
        for failed in gates["failed_gates"]:
            print(f"  failed: {failed}")
        if not gates["passed"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
