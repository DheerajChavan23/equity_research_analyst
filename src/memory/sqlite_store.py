import os
import sqlite3
import json
from typing import Optional, Dict, Any, List

class LocalMemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "data/state.db")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Episodic / Semantic Memory (long-term preferences, user constraints)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT UNIQUE,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Scratchpad store (working memory per research run)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scratchpads (
                    session_id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save_scratchpad(self, session_id: str, ticker: str, state: Dict[str, Any]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scratchpads (session_id, ticker, state_json, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=CURRENT_TIMESTAMP
            """, (session_id, ticker, json.dumps(state)))
            conn.commit()

    def load_scratchpad(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT state_json FROM scratchpads WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None


    def get_all_memories(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """Retrieves persistent memories, optionally filtered by category."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if category:
                cursor.execute("SELECT category, key, value FROM memories WHERE category = ?", (category,))
            else:
                cursor.execute("SELECT category, key, value FROM memories")
            rows = cursor.fetchall()
            return [{"category": r[0], "key": r[1], "value": r[2]} for r in rows]