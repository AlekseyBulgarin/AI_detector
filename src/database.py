import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "data" / "feedback.db"


def get_connection(database_path=DATABASE_PATH):
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path=DATABASE_PATH):
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
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id TEXT NOT NULL,
                label TEXT NOT NULL CHECK(label IN ('human', 'ai', 'unsure')),
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending', 'approved', 'rejected')),
                FOREIGN KEY(analysis_id) REFERENCES analysis(id)
            );
            """
        )


def insert_analysis(analysis_id, text_hash, prediction, probability, model_version, created_at):
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO analysis
                (id, text_hash, prediction, probability, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (analysis_id, text_hash, prediction, probability, model_version, created_at),
        )


def insert_feedback(analysis_id, label, created_at):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO feedback (analysis_id, label, created_at, status)
            VALUES (?, ?, ?, 'pending')
            """,
            (analysis_id, label, created_at),
        )
        return cursor.lastrowid
