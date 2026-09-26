import json
import sqlite3
from pathlib import Path

from src.config import sqlite_path_from_url


DATABASE_PATH = sqlite_path_from_url()
VALID_FEEDBACK_STATUSES = {
    "pending",
    "approved",
    "rejected",
    "duplicate",
    "invalid",
    "needs_review",
}
REVIEWABLE_STATUSES = VALID_FEEDBACK_STATUSES - {"pending"}
VALID_LABELS = {"human", "ai", "ai_assisted", "unsure"}
TRAINABLE_LABELS = {"human", "ai"}
VALID_CONFIDENCE = {"certain", "not_sure"}

ANALYSIS_REQUIRED_COLUMNS = {
    "id",
    "text_hash",
    "prediction",
    "probability",
    "model_version",
    "created_at",
    "text_content",
    "text_length",
    "word_count",
    "dataset_version",
    "feature_version",
    "latency_ms",
    "store_text",
    "normalized_hash",
}

FEEDBACK_REQUIRED_COLUMNS = {
    "id",
    "analysis_id",
    "text_hash",
    "normalized_hash",
    "text_content",
    "prediction_label",
    "prediction_probability",
    "model_version",
    "user_label",
    "label",
    "status",
    "allow_training",
    "user_confidence",
    "validation_flags",
    "validation_result",
    "priority_category",
    "hard_case_categories",
    "priority_score",
    "duplicate_of",
    "reviewer_note",
    "created_at",
    "reviewed_at",
}


class UnsupportedDatabaseError(RuntimeError):
    """Raised until a PostgreSQL backend is added."""


class AnalysisNotFoundError(LookupError):
    """Raised when feedback references an unknown analysis."""


def get_connection(database_path=None):
    database_path = database_path or DATABASE_PATH
    if database_path is None:
        raise UnsupportedDatabaseError(
            "PostgreSQL DATABASE_URL is reserved for a future database adapter"
        )
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _create_feedback_table(connection, table_name="feedback"):
    connection.execute(
        f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            normalized_hash TEXT NOT NULL DEFAULT '',
            text_content TEXT,
            prediction_label TEXT NOT NULL,
            prediction_probability REAL NOT NULL,
            model_version TEXT NOT NULL,
            user_label TEXT NOT NULL
                CHECK(user_label IN ('human', 'ai', 'ai_assisted', 'unsure')),
            label TEXT NOT NULL CHECK(label IN ('human', 'ai', 'ai_assisted', 'unsure')),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected', 'duplicate',
                                 'invalid', 'needs_review')),
            allow_training INTEGER NOT NULL DEFAULT 0 CHECK(allow_training IN (0, 1)),
            user_confidence TEXT CHECK(user_confidence IN ('certain', 'not_sure')
                                       OR user_confidence IS NULL),
            validation_flags TEXT NOT NULL DEFAULT '[]',
            validation_result TEXT NOT NULL DEFAULT 'accepted',
            priority_category TEXT NOT NULL DEFAULT 'edge_case',
            hard_case_categories TEXT NOT NULL DEFAULT '[]',
            priority_score REAL NOT NULL DEFAULT 0,
            duplicate_of INTEGER,
            reviewer_note TEXT,
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            FOREIGN KEY(analysis_id) REFERENCES analysis(id)
        )
        """
    )


def _feedback_schema_is_current(connection):
    table = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'feedback'"
    ).fetchone()
    if table is None:
        return False
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
    }
    if not FEEDBACK_REQUIRED_COLUMNS.issubset(columns):
        return False
    schema = table[0] or ""
    for token in ("'ai_assisted'", "'needs_review'", "'invalid'"):
        if token not in schema:
            return False
    return True


def _legacy_column(columns, name, fallback):
    """Reference a legacy column when it exists, otherwise use a literal."""
    return f"f.{name}" if name in columns else fallback


def _migrate_feedback_table(connection):
    if not _feedback_schema_is_current(connection):
        table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'feedback'"
        ).fetchone()
        if table is None:
            _create_feedback_table(connection)
        else:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(feedback)").fetchall()
            }
            connection.execute("ALTER TABLE feedback RENAME TO feedback_legacy")
            connection.execute("DROP INDEX IF EXISTS idx_feedback_text_status")
            connection.execute("DROP INDEX IF EXISTS idx_feedback_status_created")
            connection.execute("DROP INDEX IF EXISTS idx_feedback_priority")
            _create_feedback_table(connection)
            if {"id", "analysis_id", "label"}.issubset(columns):
                legacy_label = _legacy_column(columns, "label", "'unsure'")
                legacy_status = _legacy_column(columns, "status", "'pending'")
                legacy_created = _legacy_column(columns, "created_at", "''")
                legacy_consent = _legacy_column(columns, "allow_training", "0")
                connection.execute(
                    f"""
                    INSERT INTO feedback
                        (id, analysis_id, text_hash, normalized_hash, text_content,
                         prediction_label, prediction_probability, model_version,
                         user_label, label, status, allow_training,
                         validation_flags, validation_result,
                         priority_category, hard_case_categories, priority_score,
                         created_at)
                    SELECT f.id,
                           f.analysis_id,
                           COALESCE(a.text_hash, ''),
                           '',
                           a.text_content,
                           COALESCE(a.prediction, 'human'),
                           COALESCE(a.probability, 0),
                           COALESCE(a.model_version, 'legacy'),
                           CASE WHEN {legacy_label} IN ('human', 'ai', 'ai_assisted', 'unsure')
                                THEN {legacy_label} ELSE 'unsure' END,
                           CASE WHEN {legacy_label} IN ('human', 'ai', 'ai_assisted', 'unsure')
                                THEN {legacy_label} ELSE 'unsure' END,
                           CASE WHEN {legacy_status} IN ('pending', 'approved', 'rejected',
                                                         'duplicate', 'invalid', 'needs_review')
                                THEN {legacy_status} ELSE 'pending' END,
                           COALESCE({legacy_consent}, 0),
                           '[]',
                           'accepted',
                           'edge_case',
                           '[]',
                           0,
                           {legacy_created}
                    FROM feedback_legacy AS f
                    LEFT JOIN analysis AS a ON a.id = f.analysis_id
                    """
                )
            connection.execute("DROP TABLE feedback_legacy")
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_text_status ON feedback (text_hash, status)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_status_created ON feedback (status, created_at)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_priority ON feedback (priority_score DESC)"
    )


def _ensure_analysis_columns(connection):
    existing = {
        row[1] for row in connection.execute("PRAGMA table_info(analysis)").fetchall()
    }
    additions = {
        "text_content": "TEXT",
        "text_length": "INTEGER",
        "word_count": "INTEGER",
        "dataset_version": "TEXT",
        "feature_version": "TEXT",
        "latency_ms": "REAL",
        "store_text": "INTEGER NOT NULL DEFAULT 1",
        "normalized_hash": "TEXT NOT NULL DEFAULT ''",
    }
    for column, declaration in additions.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE analysis ADD COLUMN {column} {declaration}")


def initialize_database(database_path=None):
    database_path = database_path or DATABASE_PATH
    if database_path is None:
        raise UnsupportedDatabaseError(
            "PostgreSQL DATABASE_URL is reserved for a future database adapter"
        )
    if database_path != ":memory:":
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    with get_connection(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis (
                id TEXT PRIMARY KEY,
                text_hash TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                text_content TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_analysis_text_model
                ON analysis (text_hash, model_version);
            """
        )
        _ensure_analysis_columns(connection)
        _migrate_feedback_table(connection)
        if database_path != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")


def insert_analysis(
    analysis_id,
    text_hash,
    prediction,
    probability,
    model_version,
    created_at,
    text_content=None,
    normalized_hash="",
    text_length=0,
    word_count=0,
    dataset_version=None,
    feature_version=None,
    latency_ms=None,
    store_text=True,
):
    retained_text = text_content if store_text else None
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO analysis
                (id, text_hash, normalized_hash, prediction, probability,
                 model_version, created_at, text_content, text_length, word_count,
                 dataset_version, feature_version, latency_ms, store_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                text_hash,
                normalized_hash,
                prediction,
                probability,
                model_version,
                created_at,
                retained_text,
                text_length,
                word_count,
                dataset_version,
                feature_version,
                latency_ms,
                int(bool(store_text)),
            ),
        )


def get_analysis(analysis_id, database_path=None):
    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, text_hash, normalized_hash, text_content, prediction,
                   probability, model_version, dataset_version, feature_version,
                   text_length, word_count, latency_ms, store_text, created_at
            FROM analysis WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()
    return dict(row) if row else None


def find_feedback_for_analysis(analysis_id, text_hash=None, database_path=None):
    with get_connection(database_path) as connection:
        if text_hash:
            row = connection.execute(
                """
                SELECT id, analysis_id, status, user_label, allow_training
                FROM feedback
                WHERE analysis_id = ? OR text_hash = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (analysis_id, text_hash),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT id, analysis_id, status, user_label, allow_training
                FROM feedback WHERE analysis_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (analysis_id,),
            ).fetchone()
    return dict(row) if row else None


def find_feedback_for_analysis_exact(analysis_id, database_path=None):
    """Return a feedback row only when the same analysis was already reviewed."""
    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, analysis_id, status, user_label, allow_training
            FROM feedback WHERE analysis_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (analysis_id,),
        ).fetchone()
    return dict(row) if row else None


def find_feedback_by_hash(text_hash, exclude_analysis_id=None, database_path=None):
    with get_connection(database_path) as connection:
        query = "SELECT id, analysis_id, status, user_label FROM feedback WHERE text_hash = ?"
        params = [text_hash]
        if exclude_analysis_id:
            query += " AND analysis_id <> ?"
            params.append(exclude_analysis_id)
        query += " ORDER BY created_at DESC LIMIT 1"
        row = connection.execute(query, params).fetchone()
    return dict(row) if row else None


def list_recent_feedback_texts(limit=500, database_path=None):
    """Recent retained feedback texts used for near-duplicate detection."""
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, analysis_id, text_content FROM feedback
            WHERE text_content IS NOT NULL AND trim(text_content) <> ''
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def insert_feedback(
    analysis_id,
    text_hash,
    text_content,
    prediction_label,
    prediction_probability,
    model_version,
    user_label,
    allow_training,
    created_at,
    normalized_hash="",
    user_confidence=None,
    validation_flags=None,
    validation_result="accepted",
    priority_category="edge_case",
    hard_case_categories=None,
    priority_score=0.0,
    status="pending",
    duplicate_of=None,
):
    if user_label not in VALID_LABELS:
        raise ValueError("Invalid feedback label")
    if status not in VALID_FEEDBACK_STATUSES:
        raise ValueError("Invalid feedback status")
    if user_confidence is not None and user_confidence not in VALID_CONFIDENCE:
        raise ValueError("Invalid feedback confidence")
    with get_connection() as connection:
        if connection.execute(
            "SELECT 1 FROM analysis WHERE id = ?", (analysis_id,)
        ).fetchone() is None:
            raise AnalysisNotFoundError(analysis_id)
        cursor = connection.execute(
            """
            INSERT INTO feedback
                (analysis_id, text_hash, normalized_hash, text_content,
                 prediction_label, prediction_probability, model_version,
                 user_label, label, status, allow_training, user_confidence,
                 validation_flags, validation_result, priority_category,
                 hard_case_categories, priority_score, duplicate_of, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                text_hash,
                normalized_hash,
                text_content,
                prediction_label,
                prediction_probability,
                model_version,
                user_label,
                user_label,
                status,
                int(allow_training),
                user_confidence,
                json.dumps(validation_flags or []),
                validation_result,
                priority_category,
                json.dumps(hard_case_categories or [priority_category]),
                float(priority_score),
                duplicate_of,
                created_at,
            ),
        )
        return cursor.lastrowid


def find_analysis_by_hash(text_hash, model_version):
    """Return the latest matching analysis for a model version, if available."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, prediction, probability, model_version, text_content,
                   dataset_version, feature_version, created_at
            FROM analysis
            WHERE text_hash = ? AND model_version = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (text_hash, model_version),
        ).fetchone()
    return dict(row) if row else None


def _decode_flags(row):
    item = dict(row)
    for key in ("validation_flags", "hard_case_categories"):
        raw = item.get(key) or "[]"
        try:
            item[key] = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            item[key] = []
    return item


FEEDBACK_COLUMNS = """
    id, analysis_id, text_hash, normalized_hash, text_content, prediction_label,
    prediction_probability, model_version, user_label, status, allow_training,
    user_confidence, validation_flags, validation_result, priority_category,
    hard_case_categories, priority_score, duplicate_of, reviewer_note,
    created_at, reviewed_at
"""


def list_feedback(
    status=None,
    database_path=None,
    priority_category=None,
    min_priority=None,
    category=None,
):
    query = f"SELECT {FEEDBACK_COLUMNS} FROM feedback"
    clauses = []
    params = []
    if status and status != "all":
        clauses.append("status = ?")
        params.append(status)
    if priority_category:
        clauses.append("priority_category = ?")
        params.append(priority_category)
    if category:
        clauses.append(
            "(priority_category = ? OR hard_case_categories LIKE ?)"
        )
        params.extend([category, f'%"{category}"%'])
    if min_priority is not None:
        clauses.append("priority_score >= ?")
        params.append(min_priority)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY priority_score DESC, created_at DESC"
    with get_connection(database_path) as connection:
        return [
            _decode_flags(row)
            for row in connection.execute(query, params).fetchall()
        ]


def get_feedback(feedback_id, database_path=None):
    with get_connection(database_path) as connection:
        row = connection.execute(
            f"SELECT {FEEDBACK_COLUMNS} FROM feedback WHERE id = ?",
            (feedback_id,),
        ).fetchone()
    return _decode_flags(row) if row else None


def update_feedback_status(
    feedback_id,
    status,
    reviewed_at,
    database_path=None,
    reviewer_note=None,
):
    if status not in REVIEWABLE_STATUSES:
        raise ValueError("Invalid feedback review status")
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE feedback
            SET status = ?, reviewed_at = ?,
                reviewer_note = COALESCE(?, reviewer_note)
            WHERE id = ?
            """,
            (status, reviewed_at, reviewer_note, feedback_id),
        )
    if cursor.rowcount == 0:
        raise LookupError(feedback_id)


def list_approved_feedback(database_path=None):
    with get_connection(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT {FEEDBACK_COLUMNS}
            FROM feedback
            WHERE status = 'approved'
              AND allow_training = 1
              AND label IN ('human', 'ai')
              AND text_content IS NOT NULL
              AND length(trim(text_content)) >= 20
            ORDER BY id
            """
        ).fetchall()
    return [_decode_flags(row) for row in rows]
