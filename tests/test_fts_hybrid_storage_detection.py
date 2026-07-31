from __future__ import annotations

import sqlite3
from pathlib import Path

from hermes_state import SessionDB


HYBRID_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id',
    tokenize='trigram'
);
CREATE VIEW messages_fts_trigram_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';
"""


def _create_hybrid_db(path: Path) -> tuple[int, int]:
    db = SessionDB(db_path=path)
    try:
        db.create_session(session_id="s1", source="test", model="test")
        tool_id = db.append_message(
            "s1",
            role="tool",
            content="toolpayloaduniquetoken",
            tool_name="terminal",
        )
        user_id = db.append_message(
            "s1",
            role="user",
            content="userpayloaduniquetoken",
        )
    finally:
        db.close()

    con = sqlite3.connect(path)
    try:
        for trigger in (
            "messages_fts_trigram_insert",
            "messages_fts_trigram_delete",
            "messages_fts_trigram_update",
        ):
            con.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        con.execute("DROP TABLE messages_fts_trigram")
        con.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
        con.executescript(HYBRID_TRIGRAM_SQL)
        con.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"
        )
        con.execute(
            "INSERT INTO state_meta(key,value) VALUES('fts_storage_version','1') "
            "ON CONFLICT(key) DO UPDATE SET value='1'"
        )
        con.commit()
    finally:
        con.close()
    return int(tool_id), int(user_id)


def test_hybrid_trigram_layout_is_detected_and_repaired(tmp_path: Path):
    path = tmp_path / "state.db"
    tool_id, user_id = _create_hybrid_db(path)

    db = SessionDB(db_path=path)
    try:
        assert db._conn is not None
        assert db.fts_optimize_available() is True
        result = db.optimize_fts_storage(vacuum=False)
        assert result["ok"] is True
        assert db.fts_optimize_available() is False

        trigram_sql = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='messages_fts_trigram'"
        ).fetchone()[0]
        normalized = "".join(str(trigram_sql).lower().split()).replace('"', "'")
        assert "content='messages_fts_trigram_src'" in normalized

        trigram_rowids = {
            int(row[0])
            for row in db._conn.execute(
                "SELECT id FROM messages_fts_trigram_docsize ORDER BY id"
            )
        }
        assert user_id in trigram_rowids
        assert tool_id not in trigram_rowids
        assert db.search_messages("toolpayloaduniquetoken", role_filter=["tool"])
        assert db.search_messages("userpayloaduniquetoken", role_filter=["user"])

        db._conn.execute(
            "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
        )
        db._conn.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram, rank) "
            "VALUES('integrity-check', 1)"
        )
    finally:
        db.close()
