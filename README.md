# AI Detector

> Система анализа текста для определения признаков использования генеративного искусственного интеллекта.

AI Detector — веб-приложение для анализа учебных текстов и оценки вероятности того, что текст был создан или существенно сгенерирован с помощью AI.

Проект ориентирован в первую очередь на преподавателей и образовательные организации.

> **Важно:** результат AI Detector является аналитическим сигналом, а не доказательством использования искусственного интеллекта.

---

## Возможности

### Анализ текста

Пользователь вставляет текст, после чего система:

- выполняет предварительную обработку;
- извлекает текстовые признаки;
- передаёт их ML-модели;
- рассчитывает вероятность AI-generated текста;
- показывает результат и степень уверенности.

### История анализов

Система подготовлена для хранения информации о выполненных проверках:

- дата анализа;
- фрагмент текста;
- результат;
- вероятность AI;
- версия модели.

### Human Feedback

После анализа пользователь может сообщить системе реальное происхождение текста:

- Human
- AI
- AI-assisted
- Unsure

Feedback не должен автоматически изменять production-модель.

Архитектура предполагает следующий цикл:

```text
Prediction
    ↓
User Feedback
    ↓
Validation
    ↓
Human Review
    ↓
Approved Dataset
    ↓
Candidate Model
    ↓
Evaluation
    ↓
Manual Promotion
    ↓
Production Model
```

Это позволяет использовать реальные примеры для дальнейшего улучшения детектора без неконтролируемого online-learning.

---

# Интерфейс

Приложение развивается в формате минималистичного SaaS-интерфейса.

Основные разделы:

### Analyzer

Основной инструмент проверки текста.

### History

История предыдущих анализов и их результатов.

### Settings

Настройки приложения:

- Light / Dark / System theme;
- параметры интерфейса;
- управление анимациями;
- настройки хранения истории.

### Documentation

Документация по работе с сервисом:

- как пользоваться детектором;
- как интерпретировать результат;
- ограничения AI detection;
- FAQ.

### About

Информация о проекте, технологиях и контактах.

---

# Как работает AI Detector

Упрощённая архитектура:

```text
                         AI Detector

User
 │
 ▼
Web Interface
 │
 ▼
Flask API
 │
 ├───────────────┐
 │               │
 ▼               ▼
Text Pipeline   Feedback System
 │               │
 ▼               ▼
TF-IDF          Database
 │
 ▼
ML Model
 │
 ▼
Prediction
 │
 ▼
Probability + Explanation
```

Production-модель загружается при запуске приложения и повторно используется для запросов.

---

# Machine Learning

Текущая ML-архитектура основана на классическом машинном обучении.

Используются:

- Scikit-learn
- Logistic Regression
- Word-level TF-IDF
- Character-level TF-IDF
- handcrafted linguistic features

В процессе разработки сравнивались несколько вариантов:

```text
Handcrafted features
Word TF-IDF
Character TF-IDF
Combined pipeline
```

В качестве основной была выбрана combined-модель.

---

## Текстовые признаки

Помимо TF-IDF, система может анализировать статистические характеристики текста, например:

- среднюю длину предложения;
- разнообразие словаря;
- распределение длины слов;
- использование пунктуации;
- некоторые лингвистические характеристики текста.

Эти признаки комбинируются с TF-IDF представлением.

---

# Dataset

В проекте реализован собственный pipeline работы с данными.

Он поддерживает:

- UTF-8 validation;
- duplicate detection;
- normalized duplicate detection;
- metadata generation;
- deterministic dataset loading;
- dataset quality analysis.

Пример структуры:

```text
data/
├── raw/
│   ├── human/
│   └── ai/
│
├── interim/
├── processed/
└── metadata/
```

Для анализа качества данных используется:

```bash
python tools/dataset_analyzer.py
```

---

# Evaluation

Для сравнения моделей используются:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Confusion Matrix

Последние экспериментальные результаты одной из версий модели:

| Metric | Result |
|---|---:|
| Accuracy | 0.857 |
| Precision | 0.800 |
| Recall | 1.000 |
| F1 | 0.889 |
| ROC-AUC | 0.667 |

Cross-validation F1:

```text
0.846 ± 0.109
```

## Важное ограничение

Эти результаты были получены на очень небольшом экспериментальном наборе данных.

Поэтому они **не должны интерпретироваться как подтверждённая точность системы на реальных учебных работах**.

Основные текущие ограничения:

- небольшой dataset;
- возможный length bias;
- ограниченное разнообразие авторов и тематик;
- небольшая test выборка;
- AI-текст после сильного редактирования человеком сложнее классифицировать;
- human и AI-assisted тексты могут иметь смешанные характеристики.

Расширение и балансировка датасета являются одним из основных направлений развития проекта.

---

# Feedback Learning Loop

Одна из ключевых задач проекта — построение системы улучшения модели на основе проверенного feedback.

```text
Production Model
      │
      ▼
Prediction
      │
      ▼
Teacher Feedback
      │
      ▼
Pending Feedback
      │
      ▼
Validation
      │
      ▼
Human Review
      │
      ▼
Approved Samples
      │
      ▼
Dataset
      │
      ▼
Candidate Model
      │
      ▼
Evaluation
      │
      ▼
Production
```

Принцип:

> Raw user feedback никогда не должен автоматически попадать в production training dataset.

Это защищает систему от:

- ошибочной разметки;
- случайных ответов;
- дубликатов;
- poisoning;
- неконтролируемого ухудшения модели.

---

# API

## Проверка текста

```http
POST /api/check
```

Пример запроса:

```json
{
  "text": "Текст для анализа..."
}
```

API возвращает результат анализа и вероятность модели.

---

## Feedback

```http
POST /api/feedback
```

Используется для отправки пользовательской оценки результата.

Feedback сохраняется отдельно и может использоваться для дальнейшей проверки и развития dataset pipeline.

---

## Health Check

```http
GET /health
```

Используется для проверки состояния приложения.

---

# Структура проекта

```text
AI_detector/
│
├── app.py
├── train_model.py
├── requirements.txt
├── render.yaml
│
├── src/
│   ├── features.py
│   ├── ml_pipeline.py
│   ├── database.py
│   ├── feedback.py
│   └── utils.py
│
├── tools/
│   ├── dataset_analyzer.py
│   └── dataset_loader.py
│
├── models/
│   ├── model.pkl
│   └── metadata.json
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── metadata/
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   └── js/
│
├── tests/
│   ├── test_api.py
│   ├── test_dataset.py
│   └── test_features.py
│
├── reports/
│
├── README.md
├── CHANGELOG.md
└── PROJECT_STATUS.md
```

---

# Локальный запуск

## 1. Клонирование

```bash
git clone https://github.com/AlekseyBulgarin/AI_detector.git
cd AI_detector
```

## 2. Создание виртуального окружения

Windows:

```powershell
py -3.11 -m venv venv
```

Активация:

```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

## 4. Запуск

```bash
python app.py
```

После запуска приложение будет доступно локально через Flask.

---

# Обучение модели

Для повторного обучения:

```bash
python train_model.py
```

Pipeline:

```text
Dataset
   ↓
Validation
   ↓
Feature Extraction
   ↓
Model Training
   ↓
Evaluation
   ↓
Model Artifact
```

Модель сохраняется в:

```text
models/model.pkl
```

Метаданные:

```text
models/metadata.json
```

---

# Тестирование

Запуск тестов:

```bash
python -m pytest -q
```

Тестируются основные компоненты:

- API;
- dataset pipeline;
- feature extraction;
- feedback;
- ML integration.

---

# Production

Приложение может запускаться через Gunicorn:

```bash
gunicorn app:app
```

Production-архитектура:

```text
Internet
   ↓
Reverse Proxy
   ↓
Gunicorn
   ↓
Flask
   ↓
ML Pipeline
   ↓
Database
```

Проект изначально был подготовлен для Render, однако архитектура не привязана исключительно к этой платформе и может быть перенесена на VPS/Cloud infrastructure.

---

# Technology Stack

### Backend

- Python 3.11
- Flask
- Gunicorn

### Machine Learning

- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- NLTK

### Data

- SQLite
- Pandas

### Frontend

- HTML
- CSS
- JavaScript
- Jinja
- Lucide Icons

### Infrastructure

- Git
- GitHub
- Render / Linux deployment

---

# Roadmap

### ML

- [x] Baseline classifier
- [x] Word TF-IDF
- [x] Character TF-IDF
- [x] Combined model
- [x] Dataset analyzer
- [x] Duplicate detection
- [x] Model metadata
- [ ] Larger balanced dataset
- [ ] Frozen evaluation dataset
- [ ] Hard-case evaluation
- [ ] Transformer benchmark
- [ ] Calibration / uncertainty detection

### Feedback

- [x] Feedback API
- [x] Feedback database
- [ ] Advanced feedback validation
- [ ] Human review queue
- [ ] Approved feedback dataset
- [ ] Dataset versioning
- [ ] Candidate model registry
- [ ] Production model promotion
- [ ] Rollback

### Product

- [x] Web interface
- [x] AI text analysis
- [x] Feedback UI
- [ ] User accounts
- [ ] Persistent user history
- [ ] Admin panel
- [ ] PostgreSQL
- [ ] Analytics dashboard
- [ ] Export results

### Infrastructure

- [x] Production configuration
- [x] Health endpoint
- [x] Gunicorn support
- [x] Render deployment support
- [ ] VPS / Cloud migration
- [ ] PostgreSQL deployment
- [ ] Monitoring
- [ ] CI/CD
- [ ] Automated backups

---

# Principles

AI Detector follows several important principles.

### 1. Detection is probabilistic

AI detection cannot reliably prove authorship.

The output should be treated as an analytical signal.

### 2. Human-in-the-loop

Feedback should be reviewed before being incorporated into training data.

### 3. Evaluation before deployment

A new model should outperform or meaningfully improve upon the current production model before promotion.

### 4. Privacy

Educational texts may contain sensitive information.

Future development should minimize unnecessary storage of raw student content and separate prediction feedback from consent to use text for training.

### 5. Reproducibility

Datasets, model versions and evaluation results should be versioned and reproducible.

---

# Project Status

AI Detector is currently an **experimental educational ML project under active development**.

The current system already provides:

- text analysis;
- probability estimation;
- combined ML pipeline;
- dataset validation;
- feedback collection;
- model metadata;
- automated tests;
- production deployment support.

The next major objective is to expand the dataset and build a complete:

```text
Feedback
→ Review
→ Dataset
→ Training
→ Evaluation
→ Model Promotion
```

pipeline.

---

## Disclaimer

AI-generated text detection has fundamental limitations.

The result produced by this application should **not** be used as the sole basis for accusing a student of academic misconduct or applying disciplinary measures.

Human review and additional context should always be considered.

---

# AI Detector

**Detect patterns. Review evidence. Make informed decisions.**
