import sqlite3
from .settings import DB_PATH, RESULTS_DIR

def init_db():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
            mode TEXT NOT NULL, model_class TEXT, confidence REAL, category TEXT,
            trust TEXT, action TEXT, retried INTEGER, latency_ms REAL, note TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

def save_item(item):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO items
        (filename, mode, model_class, confidence, category, trust, action, retried, latency_ms, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (item["filename"], item["mode"], item.get("model_class"), item["confidence"],
         item.get("category"), item["trust"], item["action"], int(item["retried"]),
         item["latency_ms"], item.get("note")))
