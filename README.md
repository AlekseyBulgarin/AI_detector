# AI Text Detector

https://ai-detector-br89.onrender.com


Веб-приложение для определения текстов, сгенерированных искусственным интеллектом (ChatGPT и др.). Проект выполнен для 9 класса.

---

## Цель

Разработать классификатор на основе машинного обучения, отличающий человеческие тексты от сгенерированных ИИ.

## Интерфейс

Текущая версия включает teacher-oriented dashboard: анализатор, локальную историю, настройки темы и плотности, документацию, сведения о проекте и адаптивную навигацию для мобильных устройств. История и настройки сохраняются только в `localStorage` текущего браузера; результаты анализа и обратная связь продолжают использовать существующие Flask API.

## Производительность

Приложение загружает модель один раз при старте процесса. Для повторных текстов используется SHA-256-кэш с учетом версии модели, а SQLite настроен на WAL и индексированный поиск. Локальный cold start после оптимизации составляет около 1,9 секунды, обычное предсказание — единицы миллисекунд. Подробные ограничения и методика измерений находятся в `reports/performance_audit.md` и `reports/performance_before_after.md`.

## Цикл обратной связи

Ответы пользователей сохраняются как `pending`. Администратор проверяет их через `/admin/feedback` и переводит в `approved`, `rejected` или `duplicate`. Только одобренные записи с включенным согласием на обучение экспортируются в `data/raw/feedback/`; автоматического обучения по pending-данным нет. Для production задайте секрет `ADMIN_TOKEN`.

---

## Как работает

1. Пользователь вводит текст в форму
2. Программа подготавливает текст несколькими способами:
   - средняя длина предложения
   - доля уникальных слов
   - частота стоп-слов
   - количество специальных знаков препинания
   - вариативность длины слов
3. Во время обучения сравниваются четыре модели: базовые признаки, word TF-IDF, character TF-IDF и объединённая модель
4. Результат отображается на экране
5. Пользователь может оставить обратную связь, которая сохраняется в статусе `pending`

---

## Технологии

- Python 3.11
- Flask (веб-фреймворк)
- scikit-learn (машинное обучение)
- NLTK (обработка текста)
- HTML, CSS, JavaScript

---

## Результаты

- Сырые данные: 38 текстов (19 человек + 19 ИИ)
- После фильтрации нормализованных дубликатов: 35 текстов
- Отчёт качества: `reports/dataset_quality.json`
- Сравнение моделей: `reports/model_comparison.md`
- Метаданные модели: `models/metadata.json`

Метрики Phase 2 нестабильны из-за маленького датасета. Они предназначены для сравнения текущих кандидатов, а не для доказательства авторства текста.

---

## Запуск проекта

1. Клонировать репозиторий
   git clone https://github.com/AlekseyBulgarin/AI_detector.git
   cd AI_detector

2. Установить зависимости
   pip install -r requirements.txt

3. Проанализировать датасет и переобучить модель
   python tools/dataset_analyzer.py
   python train_model.py --model-version phase2-v1

   # Export approved, consented feedback after admin review, then train a new version
   python tools/export_feedback_dataset.py
   python train_model.py --include-feedback --model-version phase2-v2

   # Promote a reviewed version explicitly; training never promotes automatically
   python train_model.py --include-feedback --model-version phase2-v2 --promote

4. Запустить приложение
   python app.py

5. Открыть в браузере
   http://127.0.0.1:5000

## Deploy to Render

The repository includes `render.yaml` with:

```text
Build: pip install -r requirements.txt
Start: gunicorn app:app --workers 1 --threads 2 --timeout 120 --access-logfile - --error-logfile -
```

The trained artifact `models/model.pkl` must be committed to the repository. The
current feedback database uses SQLite at `data/feedback.db`; Render's default
filesystem is ephemeral, so production feedback should later be migrated to a
managed database through `DATABASE_URL`.



## Автор

Алексей Булгарин, 9 класс
GitHub: AlekseyBulgarin
