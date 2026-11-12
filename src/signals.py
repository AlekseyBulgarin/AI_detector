"""Fast, explainable text-risk signals without a trained classifier.

The module deliberately does not attempt to identify an author.  It reports
observable writing patterns that can be useful during a teacher's review.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import mean, pstdev


MIN_CHARACTERS = 600
MIN_WORDS = 100
MAX_LIST_RATIO = 0.45

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_TRANSITIONS_RE = re.compile(
    r"\b(?:во-первых|во-вторых|в-третьих|таким образом|следовательно|"
    r"кроме того|в заключение|подводя итог|важно отметить|стоит отметить|"
    r"необходимо отметить|в то же время|итак|более того)\b",
    re.IGNORECASE,
)
_GENERIC_RE = re.compile(
    r"\b(?:играет важную роль|имеет большое значение|является неотъемлемой частью|"
    r"нельзя недооценивать|следует подчеркнуть|открывает новые возможности)\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_RE.split(text) if len(_words(part)) >= 3]


def _signal(code: str, title: str, detail: str, weight: int, group: str) -> dict:
    return {
        "code": code,
        "title": title,
        "detail": detail,
        "weight": weight,
        "group": group,
    }


def analyze(text: str) -> dict:
    """Return a conservative, deterministic review signal for *text*.

    A score is an index of detected patterns, not an AI probability.  It is
    intentionally capped unless signals from at least two groups agree.
    """
    clean = " ".join(text.split())
    words = _words(clean)
    sentences = _sentences(clean)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    word_count = len(words)
    character_count = len(clean)

    stats = {
        "characters": character_count,
        "words": word_count,
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
    }
    if word_count < MIN_WORDS or character_count < MIN_CHARACTERS:
        return {
            "status": "insufficient",
            "score": 0,
            "signals": [],
            "stats": stats,
            "reason": (
                f"Для содержательной оценки нужно не менее {MIN_WORDS} слов и "
                f"{MIN_CHARACTERS} символов; получено {word_count} слов и "
                f"{character_count} символов."
            ),
        }

    signals: list[dict] = []
    transition_count = len(_TRANSITIONS_RE.findall(clean))
    generic_count = len(_GENERIC_RE.findall(clean))
    per_100_words = (transition_count + generic_count) * 100 / word_count
    if per_100_words >= 2.2:
        signals.append(_signal(
            "formulaic_phrasing", "Повторяющиеся шаблонные обороты",
            f"Найдено {transition_count + generic_count} типичных связок и общих формулировок.",
            25, "phrasing",
        ))

    sentence_lengths = [len(_words(sentence)) for sentence in sentences]
    if len(sentence_lengths) >= 6 and mean(sentence_lengths) > 0:
        variation = pstdev(sentence_lengths) / mean(sentence_lengths)
        stats["sentence_variation"] = round(variation, 3)
        if variation < 0.32:
            signals.append(_signal(
                "uniform_sentences", "Необычно ровный ритм предложений",
                "Длины предложений отличаются мало; это лишь повод посмотреть на стиль внимательнее.",
                20, "rhythm",
            ))

    normalized_sentences = [re.sub(r"\W+", " ", item.lower()).strip() for item in sentences]
    repeats = sum(count - 1 for count in Counter(normalized_sentences).values() if count > 1)
    if repeats:
        signals.append(_signal(
            "repeated_sentences", "Повторяющиеся фразы",
            f"Обнаружено повторов предложений: {repeats}.", 25, "repetition",
        ))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    list_lines = sum(bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", line)) for line in lines)
    if len(lines) >= 4 and list_lines / len(lines) > MAX_LIST_RATIO:
        signals.append(_signal(
            "list_heavy", "Преобладает шаблонная списочная структура",
            "Большая часть текста оформлена как список; происхождение такого текста по стилю определить нельзя.",
            0, "format",
        ))

    # Deliberately require corroboration: individual stylistic quirks are normal.
    positive = [item for item in signals if item["weight"] > 0]
    groups = {item["group"] for item in positive}
    raw_score = sum(item["weight"] for item in positive)
    if len(groups) < 2:
        score = min(raw_score, 34)
    else:
        score = min(raw_score, 100)
    status = "review" if score >= 55 else "mixed"
    if any(item["code"] == "list_heavy" for item in signals):
        status = "insufficient"
        score = 0

    return {
        "status": status,
        "score": score,
        "signals": signals,
        "stats": stats,
        "reason": None,
    }
