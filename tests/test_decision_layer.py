"""Confidence-aware decision layer contract."""

import json

from src import decision


def _thresholds(low=0.35, high=0.65):
    return {"low": low, "high": high, "source": "test", "alpha": 0.05, "samples": 10}


def test_classify_separates_human_uncertain_and_ai_bands():
    thresholds = _thresholds()

    human = decision.classify(5.0, thresholds)
    uncertain = decision.classify(50.0, thresholds)
    ai = decision.classify(95.0, thresholds)

    assert human["band"] == decision.LIKELY_HUMAN
    assert uncertain["band"] == decision.UNCERTAIN
    assert ai["band"] == decision.AI_INDICATORS
    assert human["risk"] == "low"
    assert uncertain["risk"] == "medium"
    assert ai["risk"] == "high"


def test_classify_never_claims_proof_of_authorship():
    for probability in (1.0, 50.0, 99.0):
        result = decision.classify(probability, _thresholds())
        assert "не доказательств" in result["limitation"]
        assert "аналитический сигнал" in result["warning"]


def test_classify_confidence_grows_with_distance_from_thresholds():
    thresholds = _thresholds()

    near = decision.classify(40.0, thresholds)
    far = decision.classify(95.0, thresholds)

    assert near["band"] == decision.UNCERTAIN
    assert near["confidence"] == "low"
    assert far["confidence"] == "high"
    assert far["confidence_margin"] > near["confidence_margin"]


def test_load_thresholds_falls_back_when_file_is_missing(tmp_path):
    thresholds = decision.load_thresholds(tmp_path / "absent.json")

    assert thresholds["low"] == 0.35
    assert thresholds["high"] == 0.65
    assert thresholds["source"] == "fallback_not_calibrated"


def test_load_thresholds_repairs_inverted_range(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"low": 0.9, "high": 0.4, "source": "validation_roc", "samples": 12}),
        encoding="utf-8",
    )

    thresholds = decision.load_thresholds(path)

    assert thresholds["low"] == 0.9
    assert thresholds["high"] == 0.9
    assert thresholds["source"] == "validation_roc"


def test_load_thresholds_survives_corrupt_file(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text("{not json", encoding="utf-8")

    assert decision.load_thresholds(path)["source"] == "fallback_not_calibrated"
