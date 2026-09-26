"""Hard-case mining: rank reviewed feedback by how valuable it is to a trainer.

High-value mistakes are the examples that expose real model weakness, so they
must surface first in the human review queue.
"""

CATEGORIES = (
    "false_positive_candidate",
    "false_negative_candidate",
    "low_confidence",
    "high_disagreement",
    "ai_assisted",
    "edge_case",
)

AI_ASSISTED = "ai_assisted"
UNCERTAIN = "unsure"
DISAGREEMENT_DISTANCE = 20.0
LOW_CONFIDENCE_BAND = 15.0


def _probabilistic_label(probability):
    return "ai" if probability >= 50 else "human"


def classify_case(prediction_label, probability, user_label, user_confidence=None):
    """Return ``(categories, primary_category)`` for one feedback record."""
    probability = float(probability)
    categories = []

    if user_label == AI_ASSISTED:
        categories.append(AI_ASSISTED)

    implied = _probabilistic_label(probability)
    is_fp = prediction_label == "ai" and user_label == "human"
    is_fn = prediction_label == "human" and user_label == "ai"

    if is_fp:
        categories.append("false_positive_candidate")
    if is_fn:
        categories.append("false_negative_candidate")

    distance = abs(probability - 50.0)
    if distance <= LOW_CONFIDENCE_BAND:
        categories.append("low_confidence")

    if user_label in {"human", "ai"}:
        disagreement = abs(probability - (100.0 if user_label == "ai" else 0.0))
        if disagreement >= DISAGREEMENT_DISTANCE and implied != user_label:
            categories.append("high_disagreement")

    if user_label == UNCERTAIN:
        categories.append("edge_case")

    if not categories:
        categories.append("edge_case")

    primary = categories[0]
    return categories, primary


def priority_score(
    categories,
    probability,
    user_label,
    user_confidence=None,
    prediction_label=None,
):
    """Compute a 0-100 priority score; larger means review first."""
    probability = float(probability)
    score = 0.0

    if "false_positive_candidate" in categories:
        # Human text called AI is the costliest error for a teacher.
        score += 45.0 + (probability / 100.0) * 30.0
    if "false_negative_candidate" in categories:
        # AI text called human is the main missed-detection failure.
        score += 40.0 + ((100.0 - probability) / 100.0) * 30.0
    if "high_disagreement" in categories:
        score += 20.0
    if "low_confidence" in categories:
        score += 12.0
    if AI_ASSISTED in categories:
        score += 15.0
    if "edge_case" in categories:
        score += 8.0
    if user_confidence == "certain":
        score += 10.0

    return round(min(100.0, score), 2)


def score_feedback(prediction_label, probability, user_label, user_confidence=None):
    """Convenience wrapper returning the full mining annotation."""
    categories, primary = classify_case(
        prediction_label, probability, user_label, user_confidence
    )
    score = priority_score(
        categories, probability, user_label, user_confidence, prediction_label
    )
    return {
        "categories": categories,
        "priority_category": primary,
        "priority_score": score,
    }
