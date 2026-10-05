"""Removing a visible message must inspect its identity peers, not its whole session.

Large compacted histories made each DELETE in replace_messages scan the session
again through SQLite's MIN(rowid) optimization, holding the global writer for minutes.
"""

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("action", ["delete", "archive"])
def test_peer_reordering_work_is_bounded_after_startup_heal(tmp_path, action):
    path = tmp_path / "state.db"
    db = SessionDB(path)
    targets = {}
    for sid, count in (("small", 200), ("large", 2_000)):
        db.create_session(sid, "test")

        def seed(conn):
            first = conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, display_identity) "
                "VALUES (?, 'assistant', 'shared', 1, 'shared')", (sid,),
            ).lastrowid
            conn.executemany(
                "INSERT INTO messages (session_id, role, content, timestamp, active, compacted, "
                "display_identity) VALUES (?, 'assistant', 'history', 1, 0, 1, ?)",
                [(sid, f"unrelated-{i}") for i in range(count)],
            )
            last = conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, display_identity) "
                "VALUES (?, 'assistant', 'shared', 1, 'shared')", (sid,),
            ).lastrowid
            return first, last

        targets[sid] = db._execute_write(seed)

    # Reproduce an existing installation with the earlier trigger bodies. The
    # ordinary reopen must upgrade them without a data/index rebuild.
    for name in ("messages_display_identity_delete", "messages_display_visibility_update"):
        sql = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,),
        ).fetchone()[0]
        db._conn.execute(f"DROP TRIGGER {name}")
        db._conn.execute(sql.replace(" INDEXED BY idx_messages_display_identity", ""))
    db.close()
    db = SessionDB(path)
    steps = {}
    try:
        for sid, (first, last) in targets.items():
            callbacks = 0

            def progress():
                nonlocal callbacks
                callbacks += 1
                return 0

            sql = ("DELETE FROM messages WHERE id = ?" if action == "delete"
                   else "UPDATE messages SET active = 0 WHERE id = ?")
            db._conn.set_progress_handler(progress, 10)
            try:
                db._execute_write(lambda conn: conn.execute(sql, (first,)))
            finally:
                db._conn.set_progress_handler(None, 0)
            steps[sid] = callbacks
            # The remaining generation becomes its own earliest visible row.
            assert db._read_one("SELECT display_order FROM messages WHERE id=?", (last,))[0] == last
        assert steps["large"] < steps["small"] * 3, steps
    finally:
        db.close()
