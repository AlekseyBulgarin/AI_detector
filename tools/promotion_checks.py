"""External checks required by the promotion gates.

These are deliberately separate from metric comparison: they prove the
candidate actually loads, that the test suite still passes, that no dataset
leakage slipped through, and that the HTTP API still works.
"""

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

HARD_LEAKAGE = ("DUPLICATE LEAKAGE",)
SOFT_LEAKAGE = ("HIGH LENGTH LEAKAGE", "SOURCE LEAKAGE", "TOPIC LEAKAGE")


def check_model_loads(version):
    from src.model_registry import get_candidate

    try:
        import joblib

        candidate = get_candidate(version)
        model = joblib.load(candidate["model_path"])
        ready = hasattr(model, "predict") and hasattr(model, "predict_proba")
        return ready, "candidate pipeline loaded" if ready else "missing predict API"
    except Exception as exc:  # noqa: BLE001 - report any load failure
        return False, f"model load failed: {exc}"


def check_regression_tests(python=None):
    command = [python or sys.executable, "-m", "pytest", "-q"]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"pytest could not run: {exc}"
    output = (completed.stdout or "").strip().splitlines()
    summary = output[-1] if output else ""
    return completed.returncode == 0, summary or "pytest produced no output"


def check_data_leakage(candidate_report, acknowledge_leakage=False):
    if not candidate_report:
        return False, "no evaluation report available for leakage checks"
    warnings = candidate_report.get("quality_warnings") or []
    split_summary = candidate_report.get("split_summary") or {}

    if split_summary.get("feedback_in_frozen_test"):
        return False, "feedback samples leaked into the frozen test set"

    hard = [item for item in warnings if any(key in item for key in HARD_LEAKAGE)]
    if hard:
        return False, "hard leakage detected: " + ", ".join(hard)

    soft = [item for item in warnings if any(key in item for key in SOFT_LEAKAGE)]
    if soft:
        if acknowledge_leakage:
            return True, "leakage acknowledged explicitly: " + ", ".join(soft)
        return (
            False,
            "dataset leakage requires explicit acknowledgement via "
            "--acknowledge-leakage: " + ", ".join(soft),
        )
    return True, "no dataset leakage warnings"


def check_api_smoke_tests():
    try:
        from app import app
    except Exception as exc:  # noqa: BLE001
        return False, f"application import failed: {exc}"

    client = app.test_client()
    health = client.get("/health")
    if health.status_code not in (200, 503):
        return False, f"/health returned {health.status_code}"

    probe = client.post(
        "/api/check",
        json={
            "text": "Это достаточно длинный текст для проверки работоспособности API перед продвижением модели."
        },
    )
    if probe.status_code != 200:
        return False, f"/api/check returned {probe.status_code}"
    payload = probe.get_json() or {}
    if "analysis_id" not in payload or payload.get("probability") is None:
        return False, "/api/check returned an incomplete payload"
    return True, "health and prediction smoke checks passed"


def build_checks(version, candidate_report=None, acknowledge_leakage=False, run_tests=True):
    if candidate_report is None:
        from src.evaluation import load_evaluation

        try:
            candidate_report = load_evaluation(version)
        except FileNotFoundError:
            candidate_report = None

    loaded, load_detail = check_model_loads(version)
    leakage_ok, leakage_detail = check_data_leakage(
        candidate_report, acknowledge_leakage=acknowledge_leakage
    )
    if run_tests:
        tests_ok, tests_detail = check_regression_tests()
    else:
        tests_ok, tests_detail = False, "regression tests were skipped"
    api_ok, api_detail = check_api_smoke_tests()

    return {
        "model_loads": loaded,
        "model_loads_detail": load_detail,
        "regression_tests": tests_ok,
        "regression_tests_detail": tests_detail,
        "data_leakage_checks": leakage_ok,
        "data_leakage_checks_detail": leakage_detail,
        "api_smoke_tests": api_ok,
        "api_smoke_tests_detail": api_detail,
        "_raw": json.dumps(
            {
                "load": load_detail,
                "tests": tests_detail,
                "leakage": leakage_detail,
                "api": api_detail,
            },
            ensure_ascii=False,
        ),
    }
