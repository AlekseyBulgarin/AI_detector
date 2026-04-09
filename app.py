from flask import Flask, render_template, request, jsonify
import os
import joblib
import re
import numpy as np
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)


MODEL_PATH = 'models/model.pkl'
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Модель загружена")
else:
    print("⚠️ Модель не найдена. Сначала запусти train_model.py")


STOP_WORDS = set(stopwords.words('russian'))

def extract_features(text):
    """Извлекает 5 признаков из текста"""
    if not text or len(text.strip()) < 20:
        return [0, 0, 0, 0, 0]
    

    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    words = re.findall(r'\w+', text.lower())
    
    if len(sentences) == 0 or len(words) == 0:
        return [0, 0, 0, 0, 0]
    
    # Признак 1: средняя длина предложения (в словах)
    avg_sent_len = len(words) / len(sentences)
    
    # Признак 2: доля уникальных слов
    unique_ratio = len(set(words)) / len(words)
    
    # Признак 3: частота стоп-слов
    stopword_count = sum(1 for w in words if w in STOP_WORDS)
    stopword_ratio = stopword_count / len(words)
    
    # Признак 4: разнообразие знаков препинания (длинное тире, многоточие)
    special_punct = re.findall(r'[—–…]', text)
    punct_ratio = len(special_punct) / len(text) if len(text) > 0 else 0
    
    # Признак 5: вариативность длины слов (стандартное отклонение)
    word_lengths = [len(w) for w in words]
    std_word_len = np.std(word_lengths) if word_lengths else 0
    
    return [avg_sent_len, unique_ratio, stopword_ratio, punct_ratio, std_word_len]

def predict_probability(text):
    """Возвращает вероятность того, что текст написан ИИ (0-100)"""
    if model is None:
        return 50  # Заглушка, если модель не обучена
    
    features = extract_features(text)
    prob = model.predict_proba([features])[0][1]  # вероятность класса 1 (AI)
    return round(prob * 100, 1)

@app.route('/')
def index():
    return render_template('index.html', user_text=None, probability=None, error=None)

@app.route('/check', methods=['POST'])
def check():
    text = request.form.get('user_text', '')
    
    # Проверка на пустой или слишком короткий текст
    if len(text.strip()) < 20:
        return render_template('index.html', 
                              user_text=text, 
                              probability=None, 
                              error="❌ Текст слишком короткий (минимум 20 символов)")
    
    # Получаем вероятность
    probability = predict_probability(text)
    
    return render_template('index.html', 
                          user_text=text, 
                          probability=probability, 
                          error=None)

@app.route('/api/check', methods=['POST'])
def api_check():
    """API endpoint для проверки через JSON (для расширения)"""
    data = request.get_json()
    text = data.get('text', '')
    
    if len(text.strip()) < 20:
        return jsonify({'error': 'Текст слишком короткий', 'probability': None})
    
    probability = predict_probability(text)
    return jsonify({'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)