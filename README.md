# AI Text Detector

Веб-приложение для определения текстов, сгенерированных искусственным интеллектом (ChatGPT и др.). Проект выполнен для 9 класса.

---

## Цель

Разработать классификатор на основе машинного обучения, отличающий человеческие тексты от сгенерированных ИИ.

## Интерфейс

Текущая версия включает teacher-oriented dashboard: анализатор, локальную историю, настройки темы и плотности, документацию, сведения о проекте и адаптивную навигацию для мобильных устройств. История и настройки сохраняются только в `localStorage` текущего браузера; результаты анализа и обратная связь продолжают использовать существующие Flask API.

## Производительность

Приложение загружает модель один раз при старте процесса. Для повторных текстов используется SHA-256-кэш с учетом версии модели, а SQLite настроен на WAL и индексированный поиск. Локальный cold start после оптимизации составляет около 1,9 секунды, обычное предсказание — единицы миллисекунд. Подробные ограничения и методика измерений находятся в `reports/performance_audit.md` и `reports/performance_before_after.md`.

## Цикл обратной связи

Ответы пользователей сохраняются как `pending`. Администратор проверяет их через `/admin/feedback` и переводит в `approved`, `rejected`, `needs_review` или `duplicate`. Только одобренные записи с включенным согласием на обучение экспортируются в `data/feedback/approved/`; автоматического обучения по pending-данным нет. Для production задайте секрет `ADMIN_TOKEN` — без него очередь проверки отключена (`403`). Подробности: [`docs/feedback-system.md`](docs/feedback-system.md).

## Платформа улучшения модели

Обратная связь никогда не переобучает модель автоматически. Полный цикл выглядит так:

```text
tools/dataset_analyzer.py            отчёт качества датасета
tools/export_feedback_dataset.py     экспорт одобренных и согласованных примеров
tools/build_dataset_version.py       замороженный снапшот data/datasets/<version>/
train_model.py --dataset <v>         кандидат в models/candidates/<version>/
tools/evaluate_candidate.py          метрики на frozen test
tools/compare_models.py              сравнение с production + gating
tools/promote_model.py               явное повышение версии (только человек)
tools/rollback_model.py              откат к архивной версии
```

Ключевые правила:

- пороги решения калибруются на validation-выборке, а не на frozen test;
- сплит строится по группам (`author`/`prompt`/`generation_session`/`topic`), чтобы связанные документы не попали одновременно в train и в тест;
- примеры из обратной связи никогда не попадают в frozen test;
- повышение версии требует, чтобы прошли все gating-правила, включая pytest и smoke-тесты API;
- предупреждения о длине и топиках (`HIGH LENGTH LEAKAGE`, `TOPIC LEAKAGE`) блокируют повышение, пока их не подтвердят явно флагом `--acknowledge-leakage`.

Документация: [`docs/dataset-pipeline.md`](docs/dataset-pipeline.md), [`docs/model-lifecycle.md`](docs/model-lifecycle.md), [`docs/privacy.md`](docs/privacy.md).

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
4. Результат отображается на экране: вероятность, трехпозиционное решение (`похоже на человеческий текст` / `недостаточно данных` / `признаки AI`), пороговые зоны и фактическая задержка
5. Пользователь может оставить обратную связь с меткой, уверенностью и согласием на обучение; запись сохраняется в статусе `pending` и попадает в очередь ручной проверки

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
- Замороженный тест: 5 текстов
- Отчёт качества: `reports/dataset_quality.json`
- Сравнение моделей: `reports/model_comparison.md`, `reports/model_comparison_v3.md`
- Метаданные модели: `models/metadata.json`

Метрики Phase 2 нестабильны из-за маленького датасета. Длина человеческих и AI-текстов заметно различается (медиана ≈ 33 против ≈ 69 слов), поэтому в манифесте датасета зафиксированы предупреждения `HIGH LENGTH LEAKAGE` и `TOPIC LEAKAGE`. Метрики предназначены для сравнения текущих кандидатов, а не для доказательства авторства текста.

---

## Запуск проекта

1. Клонировать репозиторий
   git clone https://github.com/AlekseyBulgarin/AI_detector.git
   cd AI_detector

2. Установить зависимости
   pip install -r requirements.txt

3. Проанализировать датасет и переобучить модель
   python tools/dataset_analyzer.py

   # Замороженный снапшот датасета и кандидат (кандидат никогда не становится production сам)
   python tools/build_dataset_version.py --version v2 --no-feedback
   python train_model.py --dataset v2 --model-version v3

   # Экспорт одобренной обратной связи (только approved + согласие + human/ai)
   python tools/export_feedback_dataset.py
   python train_model.py --dataset v2 --model-version v4

   # Оценка, gating и только затем явное повышение версии
   python tools/evaluate_candidate.py --version v3
   python tools/compare_models.py --version v3
   python tools/promote_model.py --version v3 --dry-run
   python tools/promote_model.py --version v3 --acknowledge-leakage

4. Запустить приложение
   python app.py

5. Открыть в браузере
   http://127.0.0.1:5000

6. Запустить тесты
   python -m pytest -q

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
