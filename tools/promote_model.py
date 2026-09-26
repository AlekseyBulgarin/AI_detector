"""Manually promote a candidate into production.

Promotion is always an explicit human action. Every gate must pass first, and
the previous production model is archived so the action is reversible.

Usage:
    python tools/promote_model.py --version v2
    python tools/promote_model.py --version v2 --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORTS_DIR, THRESHOLDS_PATH  # noqa: E402
from src.evaluation import load_evaluation  # noqa: E402
from src.model_registry import (  # noqa: E402
    ModelRegistryError,
    get_candidate,
    get_production,
    install_production,
)
from src.promotion import (  # noqa: E402
    evaluate_gates,
    render_comparison_markdown,
    write_gate_report,
)
from tools.promotion_checks import build_checks  # noqa: E402


def _metrics(report):
    if not report:
        return None
    return report.get("metrics") or report.get("frozen_test")


def promote(version, dry_run=False, acknowledge_leakage=False, promoted_by="cli", run_tests=True):
    candidate = get_candidate(version)
    candidate_evaluation = load_evaluation(version)

    production = None
    production_evaluation = None
    try:
        production = get_production()
    except ModelRegistryError:
        production = None
    if production is not None:
        try:
            production_evaluation = load_evaluation(production["version"])
        except FileNotFoundError:
            production_evaluation = None

    checks = build_checks(
        version,
        candidate_report=candidate_evaluation,
        acknowledge_leakage=acknowledge_leakage,
        run_tests=run_tests,
    )
    gates = evaluate_gates(
        {"metrics": _metrics(candidate_evaluation)},
        production={"metrics": _metrics(production_evaluation)} if production else None,
        checks=checks,
    )
    gates["candidate"] = version
    gates["production_before"] = production["version"] if production else None
    gate_path = write_gate_report(gates, version)

    comparisons = gates.get("comparisons", {})
    markdown = render_comparison_markdown(
        _metrics(production_evaluation),
        _metrics(candidate_evaluation),
        comparisons,
        version,
        gates,
    )
    report_path = REPORTS_DIR / f"model_comparison_{version}.md"
    report_path.write_text(markdown, encoding="utf-8")

    if not gates["passed"]:
        return {
            "promoted": False,
            "gates": gates,
            "gate_report": str(gate_path),
            "comparison_report": str(report_path),
        }

    if dry_run:
        return {
            "promoted": False,
            "dry_run": True,
            "gates": gates,
            "gate_report": str(gate_path),
            "comparison_report": str(report_path),
        }

    record = install_production(version, promoted_by=promoted_by, gates=gates)
    thresholds_source = candidate["directory"] / "thresholds.json"
    if thresholds_source.exists():
        shutil.copy2(thresholds_source, THRESHOLDS_PATH)
    log_path = REPORTS_DIR / "promotions.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "promoted": True,
        "record": record,
        "gates": gates,
        "gate_report": str(gate_path),
        "comparison_report": str(report_path),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Promote a candidate model")
    parser.add_argument("--version", required=True, help="Candidate version")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate gates only")
    parser.add_argument(
        "--acknowledge-leakage",
        action="store_true",
        help="Explicitly acknowledge reported dataset leakage warnings",
    )
    parser.add_argument("--promoted-by", default="cli")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Do not run the regression suite (gate is recorded as failed)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = promote(
            args.version,
            dry_run=args.dry_run,
            acknowledge_leakage=args.acknowledge_leakage,
            promoted_by=args.promoted_by,
            run_tests=not args.skip_tests,
        )
    except (ModelRegistryError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    gates = result["gates"]
    print(f"Gate report: {result['gate_report']}")
    print(f"Comparison report: {result['comparison_report']}")
    print(f"Promotion gates: {'PASS' if gates['passed'] else 'FAIL'}")
    for gate in gates["gates"]:
        marker = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{marker}] {gate['gate']}: {gate['detail']}")

    if not gates["passed"]:
        print("Promotion blocked.")
        return 2
    if result.get("dry_run"):
        print("Dry run only. Nothing was promoted.")
        return 0
    print(f"Promoted {args.version}. Previous production archived.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
