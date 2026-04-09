import os
import joblib
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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
    
    avg_sent_len = len(words) / len(sentences)
    unique_ratio = len(set(words)) / len(words)
    stopword_count = sum(1 for w in words if w in STOP_WORDS)
    stopword_ratio = stopword_count / len(words)
    special_punct = re.findall(r'[—–…]', text)
    punct_ratio = len(special_punct) / len(text) if len(text) > 0 else 0
    word_lengths = [len(w) for w in words]
    std_word_len = np.std(word_lengths) if word_lengths else 0
    
    return [avg_sent_len, unique_ratio, stopword_ratio, punct_ratio, std_word_len]

# Загрузка данных
X, Y = [], []

# Тексты людей (класс 0)
human_path = 'data/raw/human/'
if os.path.exists(human_path):
    for filename in os.listdir(human_path):
        if filename.endswith('.txt'):
            with open(os.path.join(human_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                X.append(extract_features(text))
                Y.append(0)
    print(f"✅ Загружено {len(os.listdir(human_path))} текстов от людей")
else:
    print(f"⚠️ Папка {human_path} не найдена")

# Тексты ИИ (класс 1)
ai_path = 'data/raw/ai/'
if os.path.exists(ai_path):
    for filename in os.listdir(ai_path):
        if filename.endswith('.txt'):
            with open(os.path.join(ai_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                X.append(extract_features(text))
                Y.append(1)
    print(f"✅ Загружено {len(os.listdir(ai_path))} текстов от ИИ")

if len(X) == 0:
    print("❌ Нет данных для обучения! Положи тексты в папки data/raw/human/ и data/raw/ai/")
    exit()

# Обучение модели
X = np.array(X)
Y = np.array(Y)

model = LogisticRegression()
model.fit(X, Y)

# Оценка точности
y_pred = model.predict(X)
accuracy = accuracy_score(Y, y_pred)
print(f"📊 Точность на обучающей выборке: {accuracy * 100:.1f}%")

# Сохранение модели
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.pkl')
print("✅ Модель сохранена в models/model.pkl")