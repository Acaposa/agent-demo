# ============================================================
# agent/session_store.py —— 会话持久化（SQLite）
# ------------------------------------------------------------
# 把 AgentSession 的 summary + messages 存进 SQLite，
# 支持按 session_id 恢复，实现跨进程/跨会话记忆。
# ============================================================

import json
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime

from agent.config import BASE_DIR

logger = logging.getLogger("agent")

SESSIONS_DB = BASE_DIR / "sessions.db"


@contextmanager
def _conn():
    """打开一个连接，退出时提交并关闭。

    注意：sqlite3.Connection 本身的 `with` 只提交事务、不关闭连接，
    长期运行会有连接泄漏，这里显式包一层保证资源释放。
    """
    conn = sqlite3.connect(SESSIONS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            summary    TEXT,
            messages   TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_session(session_id: str, summary: str | None, messages: list[dict]) -> None:
    if not session_id:
        return
    try:
        with _conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, summary, messages, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    summary or "",
                    json.dumps(messages, ensure_ascii=False),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
    except Exception as e:
        logger.warning(f"保存 session 失败：{e}")


def load_session(session_id: str) -> tuple[str | None, list[dict]] | None:
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT summary, messages FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        summary, messages_json = row
        return (summary or None, json.loads(messages_json))
    except Exception as e:
        logger.warning(f"读取 session 失败：{e}")
        return None


def list_sessions() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT session_id, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [{"session_id": r[0], "updated_at": r[1]} for r in rows]


def delete_session(session_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))