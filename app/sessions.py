"""Persistent local shopping sessions with optimistic revision checks."""

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3

from app.retrieval import ROOT


class SessionConflict(ValueError):
    pass


def empty_state():
    return {"requirements": None, "last_candidates": [], "focus_product": None,
            "search_options": {"price_ceiling": None, "excluded_product_ids": []}}


class SessionStore:
    def __init__(self, path=None):
        self.path = Path(path or ROOT / "data/processed/shopping_sessions.sqlite")

    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                state_json TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_turns (
                session_id TEXT NOT NULL, revision INTEGER NOT NULL, query TEXT NOT NULL,
                result_json TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY (session_id, revision)
            );
        """)
        return connection

    @staticmethod
    def validate_id(session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[\w.-]{1,64}", session_id):
            raise ValueError("会话 ID 应为 1～64 位字母、数字、中文、点、横线或下划线。")

    def load(self, session_id, history_limit=6):
        self.validate_id(session_id)
        if type(history_limit) is not int or not 1 <= history_limit <= 100:
            raise ValueError("历史条数必须为 1～100。")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT revision, state_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            turns = connection.execute(
                "SELECT revision, query, result_json FROM session_turns WHERE session_id=? "
                "ORDER BY revision DESC LIMIT ?", (session_id, history_limit)).fetchall()
        return {
            "session_id": session_id, "revision": row[0] if row else 0,
            "state": json.loads(row[1]) if row else empty_state(),
            "turns": [{"revision": turn[0], "query": turn[1], "result": json.loads(turn[2])}
                      for turn in reversed(turns)],
        }

    def save(self, session_id, expected_revision, state, query, result):
        self.validate_id(session_id)
        state_json = json.dumps(state, ensure_ascii=False)
        result_json = json.dumps(result, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT revision FROM sessions WHERE session_id=?", (session_id,)).fetchone()
                if (row[0] if row else 0) != expected_revision:
                    raise SessionConflict("同一会话已被另一轮更新，本轮未覆盖状态，请基于最新会话重试。")
                revision = expected_revision + 1
                connection.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
                                   (session_id, revision, state_json, now))
                connection.execute("INSERT INTO session_turns VALUES (?, ?, ?, ?, ?)",
                                   (session_id, revision, query, result_json, now))
        return revision
