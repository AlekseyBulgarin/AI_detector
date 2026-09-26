"""Confidence-aware decision layer.

The model is not forced into a binary human/AI verdict. An uncertainty region
separates the two claims, and the thresholds are loaded from evaluation output
rather than guessed. The wording is evidence-oriented and never presented as
proof of authorship.
"""

import json
from pathlib import Path

from src.config import THRESHOLDS_PATH


DEFAULT_THRESHOLDS = {
    "low": 0.35,
    "high": 0.65,
    "source": "fallback_not_calibrated",
    "alpha": None,
    "samples": 0,
}

LIKELY_HUMAN = "likely_human"
UNCERTAIN = "uncertain"
AI_INDICATORS = "ai_indicators"

LABELS = {
    LIKELY_HUMAN: {
        "code": LIKELY_HUMAN,
        "title": "Похоже на человеческий текст",
        "short": "Likely Human",
        "description": "Модель не обнаружила устойчивых признаков сгенерированного текста.",
        "risk": "low",
    },
    UNCERTAIN: {
        "code": UNCERTAIN,
        "title": "Недостаточно данных для вывода",
        "short": "Uncertain",
        "description": "Сигналы смешаны: текст находится в зоне неопределённости.",
        "risk": "medium",
    },
    AI_INDICATORS: {
        "code": AI_INDICATORS,
        "title": "Обнаружены признаки AI-текста",
        "short": "AI Indicators Detected",
        "description": "Модель обнаружила закономерности, связанные с текстом, созданным ИИ.",
        "risk": "high",
    },
}

LIMITATIONS = (
    "Результат является аналитическим сигналом, а не доказательством использования ИИ. "
    "Отредактированные, короткие и смешанные тексты определяются хуже."
)


def load_thresholds(path=None):
    source = Path(path or THRESHOLDS_PATH)
    if not source.exists():
        return dict(DEFAULT_THRESHOLDS)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_THRESHOLDS)
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(
        {
            "low": float(payload.get("low", DEFAULT_THRESHOLDS["low"])),
            "high": float(payload.get("high", DEFAULT_THRESHOLDS["high"])),
            "source": payload.get("source", "unknown"),
            "alpha": payload.get("alpha"),
            "samples": payload.get("samples", 0),
        }
    )
    if thresholds["low"] > thresholds["high"]:
        thresholds["high"] = thresholds["low"]
    return thresholds


def _confidence(margin):
    if margin >= 0.25:
        return "high"
    if margin >= 0.1:
        return "medium"
    return "low"


def classify(probability, thresholds=None):
    """Map an AI probability to a three-way decision with a confidence level."""
    thresholds = thresholds or load_thresholds()
    value = max(0.0, min(100.0, float(probability)))
    low = thresholds["low"] * 100
    high = thresholds["high"] * 100

    if value <= low:
        band = LIKELY_HUMAN
        margin = (low - value) / 100
    elif value >= high:
        band = AI_INDICATORS
        margin = (value - high) / 100
    else:
        band = UNCERTAIN
        margin = min(value - low, high - value) / 100

    return {
        "band": band,
        "title": LABELS[band]["title"],
        "short": LABELS[band]["short"],
        "description": LABELS[band]["description"],
        "risk": LABELS[band]["risk"],
        "confidence": _confidence(margin),
        "confidence_margin": round(margin, 4),
        "probability": round(value, 1),
        "thresholds": {
            "low": round(low, 1),
            "high": round(high, 1),
            "source": thresholds.get("source", "unknown"),
        },
        "limitation": LIMITATIONS,
        "warning": (
            "Используйте результат как аналитический сигнал, а не как "
            "доказательство использования ИИ."
        ),
    }
