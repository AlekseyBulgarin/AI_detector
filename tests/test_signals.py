from src.signals import MIN_WORDS, analyze


def test_short_text_is_not_scored_as_authorship_signal():
    result = analyze("Короткий фрагмент текста, которого недостаточно для вывода.")

    assert result["status"] == "insufficient"
    assert result["score"] == 0
    assert "не менее" in result["reason"]


def test_single_group_of_signals_cannot_produce_high_review_score():
    text = " ".join(
        "Таким образом, важно отметить значение этой темы " + "слово " * size + "."
        for size in range(2, 18)
    )

    result = analyze(text)

    assert result["score"] < 55
    assert result["status"] == "mixed"


def test_correlated_signals_are_explained_for_review():
    sentences = [
        "Таким образом важно отметить значение вопроса для общества сегодня."
        for _ in range(12)
    ]
    result = analyze(" ".join(sentences))

    assert result["status"] == "review"
    assert result["score"] >= 55
    assert len(result["signals"]) >= 2
    assert all("title" in signal and "detail" in signal for signal in result["signals"])


def test_list_heavy_text_returns_insufficient_result():
    text = "\n".join(f"- Пункт номер {number} содержит достаточно слов для проверки." for number in range(1, 35))

    result = analyze(text)

    assert result["status"] == "insufficient"
    assert result["score"] == 0
