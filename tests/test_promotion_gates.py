"""Promotion gates: a candidate never becomes production automatically."""

from src.promotion import DEFAULT_POLICY, evaluate_gates, render_comparison_markdown


def _metrics(**overrides):
    metrics = {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "roc_auc": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "samples": 5,
    }
    metrics.update(overrides)
    return metrics


def test_equal_candidate_passes_every_gate():
    report = evaluate_gates(
        {"metrics": _metrics()}, {"metrics": _metrics()}
    )

    assert report["passed"] is True
    assert report["failed_gates"] == []
    assert report["requires_manual_promotion"] is True
    assert {gate["gate"] for gate in report["gates"]} >= {
        "model_loads",
        "regression_tests",
        "data_leakage_checks",
        "api_smoke_tests",
    }


def test_missing_production_baseline_blocks_promotion():
    report = evaluate_gates({"metrics": _metrics()}, None)

    assert report["passed"] is False
    assert "production_baseline_available" in report["failed_gates"]


def test_f1_regression_beyond_policy_blocks_promotion():
    report = evaluate_gates(
        {"metrics": _metrics(f1=0.90, samples=5)}, {"metrics": _metrics(f1=0.98)}
    )

    assert report["passed"] is False
    assert "no_regression_f1" in report["failed_gates"]
    assert report["comparisons"]["f1"]["delta"] == -0.08


def test_false_positive_regression_blocks_promotion():
    report = evaluate_gates(
        {"metrics": _metrics(false_positive_rate=0.12)},
        {"metrics": _metrics(false_positive_rate=0.0)},
    )

    assert "no_regression_false_positive_rate" in report["failed_gates"]


def test_policy_can_be_tightened():
    report = evaluate_gates(
        {"metrics": _metrics(f1=0.97)},
        {"metrics": _metrics(f1=1.0)},
        policy={"f1_delta_min": -0.01},
    )

    assert report["passed"] is False
    assert report["policy"]["f1_delta_min"] == -0.01
    assert report["policy"]["roc_auc_delta_min"] == DEFAULT_POLICY["roc_auc_delta_min"]


def test_failed_external_checks_are_reported():
    report = evaluate_gates(
        {"metrics": _metrics()},
        {"metrics": _metrics()},
        checks={
            "regression_tests": False,
            "regression_tests_detail": "7 failed",
            "data_leakage_checks": False,
        },
    )

    assert set(report["failed_gates"]) == {"regression_tests", "data_leakage_checks"}
    failed = {gate["gate"]: gate for gate in report["gates"] if not gate["passed"]}
    assert failed["regression_tests"]["detail"] == "7 failed"


def test_comparison_markdown_mentions_manual_promotion():
    report = evaluate_gates({"metrics": _metrics()}, {"metrics": _metrics()})
    markdown = render_comparison_markdown(
        {"metrics": _metrics()},
        {"metrics": _metrics()},
        report["comparisons"],
        "v9",
        report,
    )

    assert "| f1 | 1.0000 | 1.0000 |" in markdown
    assert "tools/promote_model.py" in markdown
    assert "PASS" in markdown
