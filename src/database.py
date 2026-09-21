import sqlite3
from pathlib import Path

from src.config import sqlite_path_from_url


DATABASE_PATH = sqlite_path_from_url()
VALID_FEEDBACK_STATUSES = {"pending", "approved", "rejected", "duplicate"}


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
            text_content TEXT,
            prediction_label TEXT NOT NULL,
            prediction_probability REAL NOT NULL,
            model_version TEXT NOT NULL,
            user_label TEXT NOT NULL CHECK(user_label IN ('human', 'ai', 'unsure')),
            label TEXT NOT NULL CHECK(label IN ('human', 'ai', 'unsure')),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected', 'duplicate')),
            allow_training INTEGER NOT NULL DEFAULT 0 CHECK(allow_training IN (0, 1)),
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            FOREIGN KEY(analysis_id) REFERENCES analysis(id)
        )
        """
    )


def _migrate_feedback_table(connection):
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
        schema = table[0] or ""
        required = {
            "text_hash",
            "text_content",
            "prediction_label",
            "prediction_probability",
            "model_version",
            "user_label",
            "reviewed_at",
            "allow_training",
        }
        if not required.issubset(columns) or "'duplicate'" not in schema:
            connection.execute("ALTER TABLE feedback RENAME TO feedback_legacy")
            _create_feedback_table(connection)
            if "analysis_id" in columns and "label" in columns:
                connection.execute(
                    """
                    INSERT INTO feedback
                        (id, analysis_id, text_hash, text_content,
                         prediction_label, prediction_probability, model_version,
                         user_label, label, status, created_at)
                    SELECT f.id,
                           f.analysis_id,
                           COALESCE(a.text_hash, ''),
                           a.text_content,
                           COALESCE(a.prediction, 'human'),
                           COALESCE(a.probability, 0),
                           COALESCE(a.model_version, 'legacy'),
                           f.label,
                           f.label,
                           CASE WHEN f.status IN ('pending', 'approved', 'rejected', 'duplicate')
                                THEN f.status ELSE 'pending' END,
                           f.created_at
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
        analysis_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(analysis)").fetchall()
        }
        if "text_content" not in analysis_columns:
            connection.execute("ALTER TABLE analysis ADD COLUMN text_content TEXT")
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
):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO analysis
                (id, text_hash, prediction, probability, model_version, created_at, text_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                text_hash,
                prediction,
                probability,
                model_version,
                created_at,
                text_content,
            ),
        )


def get_analysis(analysis_id, database_path=None):
    with get_connection(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, text_hash, text_content, prediction, probability,
                   model_version, created_at
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
):
    with get_connection() as connection:
        if connection.execute("SELECT 1 FROM analysis WHERE id = ?", (analysis_id,)).fetchone() is None:
            raise AnalysisNotFoundError(analysis_id)
        cursor = connection.execute(
            """
            INSERT INTO feedback
                (analysis_id, text_hash, text_content, prediction_label,
                 prediction_probability, model_version, user_label, label,
                 allow_training, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                analysis_id,
                text_hash,
                text_content,
                prediction_label,
                prediction_probability,
                model_version,
                user_label,
                user_label,
                int(allow_training),
                created_at,
            ),
        )
        return cursor.lastrowid


def find_analysis_by_hash(text_hash, model_version):
    """Return the latest matching analysis for a model version, if available."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, prediction, probability, model_version, text_content, created_at
            FROM analysis
            WHERE text_hash = ? AND model_version = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (text_hash, model_version),
        ).fetchone()
    return dict(row) if row else None


def list_feedback(status=None, database_path=None):
    query = """
        SELECT id, analysis_id, text_hash, text_content, prediction_label,
               prediction_probability, model_version, user_label, status,
               allow_training, created_at, reviewed_at
        FROM feedback
    """
    params = ()
    if status:
        query += " WHERE status = ?"
        params = (status,)
    query += " ORDER BY created_at DESC"
    with get_connection(database_path) as connection:
        return [dict(row) for row in connection.execute(query, params).fetchall()]


def update_feedback_status(feedback_id, status, reviewed_at, database_path=None):
    if status not in VALID_FEEDBACK_STATUSES - {"pending"}:
        raise ValueError("Invalid feedback review status")
    with get_connection(database_path) as connection:
        cursor = connection.execute(
            "UPDATE feedback SET status = ?, reviewed_at = ? WHERE id = ?",
            (status, reviewed_at, feedback_id),
        )
    if cursor.rowcount == 0:
        raise LookupError(feedback_id)


def list_approved_feedback(database_path=None):
    with get_connection(database_path) as connection:
        rows = connection.execute(
            """
            SELECT id, text_hash, text_content, prediction_label,
                   prediction_probability, model_version, user_label, created_at
            FROM feedback
            WHERE status = 'approved'
              AND allow_training = 1
              AND user_label IN ('human', 'ai')
              AND text_content IS NOT NULL
              AND length(trim(text_content)) >= 20
            ORDER BY id
            """
        ).fetchall()
    return [dict(row) for row in rows]
