from flask import Flask, render_template, request, jsonify
import os
import joblib

from src.features import extract_features


app = Flask(__name__)


MODEL_PATH = "models/model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Модель загружена")
else:
    print("⚠️ Модель не найдена. Сначала запусти train_model.py")


def predict_probability(text):
    """Возвращает вероятность того, что текст написан ИИ (0-100)"""
    if model is None:
        return 50

    features = extract_features(text)
    prob = model.predict_proba([features])[0][1]
    return round(prob * 100, 1)


@app.route("/")
def index():
    return render_template("index.html", user_text=None, probability=None, error=None)


@app.route("/check", methods=["POST"])
def check():
    text = request.form.get("user_text", "")

    if len(text.strip()) < 20:
        return render_template(
            "index.html",
            user_text=text,
            probability=None,
            error="❌ Текст слишком короткий (минимум 20 символов)",
        )

    probability = predict_probability(text)

    return render_template(
        "index.html",
        user_text=text,
        probability=probability,
        error=None,
    )


@app.route("/api/check", methods=["POST"])
def api_check():
    """API endpoint для проверки через JSON (для расширения)"""
    data = request.get_json()
    text = data.get("text", "")

    if len(text.strip()) < 20:
        return jsonify({"error": "Текст слишком короткий", "probability": None})

    probability = predict_probability(text)
    return jsonify({"probability": probability})


if __name__ == "__main__":
    app.run(debug=True)
