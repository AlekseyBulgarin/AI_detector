from src.features import extract_features
from src.ml_pipeline import build_models


def test_feature_extraction_returns_five_numeric_values():
    features = extract_features("Это достаточно длинный текст для проверки признаков.")
    assert len(features) == 5
    assert all(isinstance(value, (int, float)) for value in features)


def test_all_phase_two_models_are_pipelines():
    models = build_models()
    assert set(models) == {"baseline", "word_tfidf", "char_tfidf", "combined"}
    assert all(hasattr(model, "fit") for model in models.values())
