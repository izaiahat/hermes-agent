"""F11: dispose a dead direct owner's retained claim, never its UNKNOWN outcome."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def interrupted(tmp_path, monkeypatch):
    from cron import executions, jobs

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    # The child really claims, starts and exits without finishing. No payload is run.
    child = subprocess.run([sys.executable, "-c", """
import json
from cron import executions, jobs
j = jobs.create_job(prompt='fixture only', schedule='every 24h')
e = executions.create_execution(j['id'], source='direct')
assert jobs.claim_job_for_fire(j['id'], manual=True)
assert executions.mark_execution_running(e['id'])
print(json.dumps({'job_id': j['id'], 'execution_id': e['id']}))
"""], capture_output=True, text=True, check=True, env=os.environ.copy())
    ids = json.loads(child.stdout)
    with jobs.use_cron_store(home):
        assert executions.recover_interrupted_executions() == 1
        jobs.pause_job(ids["job_id"], reason="interrupted effect: do not replay")
        sibling = jobs.create_job(schedule="every 12h", prompt="untouched sibling", paused=True)
        rows = jobs._peek_jobs_unlocked()
        target = next(j for j in rows if j["id"] == ids["job_id"])
        ledger = executions.get_execution(ids["execution_id"])
        assert ledger["status"] == "unknown"
        # Existing APIs cannot dispose this terminal row/recurring claim.
        assert executions.recover_interrupted_executions() == 0
        assert executions.finish_execution(ledger["id"], success=True) is None
        assert jobs.clear_run_claim(target["id"]) is False
        lock = tmp_path / "payload.lock"
        lock.touch()
        evidence = tmp_path / "reconciliation.json"
        evidence.write_text(json.dumps({**ids, "no_active_work": True,
                                      "finding": "fixture: no payload was launched"}))
        now = datetime.now(timezone.utc)
        request = {
            "home": str(home.resolve()), "job": target, "execution": ledger,
            "reason": "Reconciled interrupted effect; preserve UNKNOWN and pause",
            "created_at": now.isoformat(), "expires_at": (now + timedelta(minutes=10)).isoformat(),
            "evidence": [{"path": str(evidence), "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}],
            "work_locks": [{"path": str(lock), "device": lock.stat().st_dev, "inode": lock.stat().st_ino}],
        }
        yield request, rows, sibling


@pytest.mark.linux_only
def test_exact_disposition_preserves_unknown_pause_and_siblings(interrupted):
    from cron import executions, jobs
    from cron.claim_disposition import dispose_claim, inspect_disposition

    request, before, _ = interrupted
    original = copy.deepcopy(request)
    assert dispose_claim(request, apply=False)["status"] == "ready"
    assert jobs._peek_jobs_unlocked() == before
    result = dispose_claim(request, apply=True)
    assert result["status"] == "disposed"
    expected = copy.deepcopy(before)
    next(j for j in expected if j["id"] == request["job"]["id"])["fire_claim"] = None
    assert jobs._peek_jobs_unlocked() == expected
    assert executions.get_execution(request["execution"]["id"]) == request["execution"]
    assert request == original  # Native saver never receives the caller's comparison oracle.
    assert inspect_disposition(request["execution"]["id"])["status"] == "applied"
    with pytest.raises(ValueError, match="intent exists"):
        dispose_claim(request, apply=True)
    assert jobs._peek_jobs_unlocked() == expected


@pytest.mark.linux_only
@pytest.mark.parametrize("change", ["evidence", "job", "ledger", "work_lock"])
def test_intent_does_not_authorize_moved_preimages(interrupted, monkeypatch, change):
    from cron import jobs
    from cron import claim_disposition as disposition
    from contextlib import contextmanager

    request, before, _ = interrupted
    original_append = disposition._append
    original_ledger = disposition._ledger
    active_conn = []

    @contextmanager
    def track_ledger(home):
        with original_ledger(home) as conn:
            active_conn.append(conn)
            yield conn

    def append_then_move(path, event, **kwargs):
        original_append(path, event, **kwargs)
        if not kwargs.get("first"):
            return
        if change == "work_lock":
            lock = Path(request["work_locks"][0]["path"])
            lock.unlink()
            lock.touch()
            return
        if change == "evidence":
            Path(request["evidence"][0]["path"]).write_text("moved after admission")
        elif change == "job":
            changed = copy.deepcopy(before)
            changed[0]["paused_reason"] = "concurrent replacement"
            jobs.save_jobs(changed)
        else:
            # Fault injection in the connection holding the ledger write exclusion.
            active_conn[0].execute("UPDATE executions SET error='moved' WHERE id=?",
                                   (request["execution"]["id"],))

    monkeypatch.setattr(disposition, "_ledger", track_ledger)
    monkeypatch.setattr(disposition, "_append", append_then_move)
    with pytest.raises(ValueError, match="moved"):
        disposition.dispose_claim(request, apply=True)
    assert next(j for j in jobs._peek_jobs_unlocked() if j["id"] == request["job"]["id"])["fire_claim"] == request["job"]["fire_claim"]
    with pytest.raises(ValueError, match="intent exists"):
        disposition.dispose_claim(request, apply=True)


@pytest.mark.linux_only
@pytest.mark.parametrize("case", [
    "wrong_job", "wrong_execution", "moved_claim", "enabled", "run_claim", "missing_start",
    "foreign_owner", "outside_execution", "non_unknown", "newer", "active", "live_pid",
    "pid_probe_error", "inprocess", "expired", "evidence", "no_evidence", "no_work_locks",
    "work_lock_replaced", "work_busy", "jobs_busy", "fire_busy", "ledger_busy", "profile",
])
def test_refusal_never_changes_job_or_execution(interrupted, monkeypatch, case):
    from cron import executions, jobs, scheduler
    from cron import claim_disposition as disposition
    from contextlib import ExitStack
    import fcntl
    import sqlite3

    request, _, _ = interrupted
    with ExitStack() as stack:
        if case == "wrong_job":
            request["job"]["id"] = "wrong"
        elif case == "wrong_execution":
            request["execution"]["id"] = "0" * 32
        elif case == "moved_claim":
            request["job"]["fire_claim"]["by"] += "moved"
        elif case in {"enabled", "run_claim", "missing_start", "foreign_owner", "outside_execution", "non_unknown", "live_pid"}:
            rows = jobs._peek_jobs_unlocked()
            job = next(j for j in rows if j["id"] == request["job"]["id"])
            if case == "enabled":
                job["enabled"] = True
            elif case == "run_claim":
                job["run_claim"] = {"by": "other", "at": request["created_at"]}
            elif case == "foreign_owner":
                job["fire_claim"]["by"] = "other-host:" + job["fire_claim"]["by"].split(":", 1)[1]
            elif case == "outside_execution":
                job["fire_claim"]["at"] = request["created_at"]
            else:
                with executions._transaction() as conn:
                    if case == "missing_start":
                        conn.execute("UPDATE executions SET process_started_at=NULL")
                    elif case == "non_unknown":
                        conn.execute("UPDATE executions SET status='completed'")
                    else:
                        conn.execute("UPDATE executions SET pid=?", (os.getpid(),))
                        # Use a wrong start-time deliberately: a recycled/live PID is never cleared.
                        conn.execute("UPDATE executions SET process_started_at=1")
                        parts = job["fire_claim"]["by"].split(":")
                        job["fire_claim"]["by"] = f"{parts[0]}:{os.getpid()}:{parts[2]}"
                request["execution"] = executions.get_execution(request["execution"]["id"])
            jobs.save_jobs(rows)
            request["job"] = job
        elif case in {"newer", "active"}:
            added = executions.create_execution(request["job"]["id"], source="direct")
            if case == "newer":
                executions.finish_execution(added["id"], success=True)
        elif case == "pid_probe_error":
            def denied(*args):
                raise PermissionError("probe denied")
            monkeypatch.setattr(disposition.os, "kill", denied)
        elif case == "inprocess":
            monkeypatch.setattr(scheduler, "is_job_running", lambda *a, **k: True)
        elif case == "expired":
            request["expires_at"] = request["created_at"]
        elif case == "evidence":
            Path(request["evidence"][0]["path"]).write_text("moved")
        elif case == "no_evidence":
            request["evidence"] = []
        elif case == "no_work_locks":
            request["work_locks"] = []
        elif case == "work_lock_replaced":
            request["work_locks"][0]["inode"] += 1
        elif case in {"work_busy", "jobs_busy", "fire_busy"}:
            monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.01)
            if case == "work_busy":
                path = Path(request["work_locks"][0]["path"])
            elif case == "jobs_busy":
                path = jobs._jobs_lock_file()
            else:
                import uuid
                key = f"{Path(request['home']) / 'cron'}::{request['job']['id']}"
                path = Path(request["home"]) / "cron" / f".fire-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex}.lock"
            fd = os.open(path, os.O_RDONLY)
            stack.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif case == "ledger_busy":
            conn = sqlite3.connect(Path(request["home"]) / "cron" / "executions.db")
            stack.callback(conn.close)
            conn.execute("BEGIN IMMEDIATE")
        elif case == "profile":
            request["home"] += "-wrong"
        jobs_path = jobs._current_cron_store().jobs_file
        before_bytes = jobs_path.read_bytes()
        # Read-only connection works even when the negative control holds the ledger writer lock.
        conn = sqlite3.connect(Path(disposition._home()) / "cron" / "executions.db")
        before_ledger = conn.execute("SELECT * FROM executions ORDER BY id").fetchall()
        with pytest.raises((ValueError, RuntimeError, OSError, sqlite3.Error)):
            disposition.dispose_claim(request, apply=True)
        assert jobs_path.read_bytes() == before_bytes
        assert conn.execute("SELECT * FROM executions ORDER BY id").fetchall() == before_ledger
        conn.close()
        assert not (disposition._home() / "cron" / "claim-dispositions").exists()


@pytest.mark.linux_only
@pytest.mark.parametrize("failure", ["before_save", "after_save", "readback", "verification_append"])
def test_uncertain_save_has_durable_intent_and_cannot_replay(interrupted, monkeypatch, failure):
    from cron import executions, jobs
    from cron import claim_disposition as disposition

    request, before, _ = interrupted
    save, append = jobs.save_jobs, disposition._append
    calls = []

    def uncertain_save(rows):
        path = disposition._journal_path(disposition._home(), request["execution"]["id"])
        assert json.loads(path.read_text().splitlines()[0])["event"] == "intent"
        calls.append(1)
        if failure == "before_save":
            raise OSError("before save")
        save(rows)
        # Mutation after the save must not change the readback oracle.
        rows[0]["unexpected_mutation"] = True
        if failure == "after_save":
            raise OSError("lost save acknowledgement")
        if failure == "readback":
            moved = jobs._peek_jobs_unlocked()
            moved[0]["paused_reason"] = "moved after save"
            save(moved)

    def uncertain_append(path, event, **kwargs):
        if not kwargs.get("first") and failure == "verification_append":
            raise OSError("lost verification append")
        return append(path, event, **kwargs)

    monkeypatch.setattr(jobs, "save_jobs", uncertain_save)
    monkeypatch.setattr(disposition, "_append", uncertain_append)
    with pytest.raises((OSError, ValueError)):
        disposition.dispose_claim(request, apply=True)
    expected = {"before_save": "not_applied", "readback": "uncertain"}.get(failure, "applied")
    assert disposition.inspect_disposition(request["execution"]["id"])["status"] == expected
    assert executions.get_execution(request["execution"]["id"]) == request["execution"]
    with pytest.raises(ValueError, match="intent exists"):
        disposition.dispose_claim(request, apply=True)
    assert len(calls) == 1


@pytest.mark.linux_only
def test_cli_preview_apply_inspect_and_profile_isolation(interrupted, tmp_path, monkeypatch):
    from cron import jobs
    from cron import claim_disposition as disposition

    request, before, _ = interrupted
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request))
    command = [sys.executable, "-m", "cron.claim_disposition"]

    def cli(*args):
        result = subprocess.run(command + list(args), capture_output=True, text=True, check=True,
                                env=os.environ.copy())
        return json.loads(result.stdout)

    assert cli("--request", str(request_file))["status"] == "ready"
    assert jobs._peek_jobs_unlocked() == before
    other = tmp_path / "other-home"
    with monkeypatch.context() as scoped, jobs.use_cron_store(other):
        scoped.setenv("HERMES_HOME", str(other))
        with pytest.raises(ValueError, match="profile mismatch"):
            disposition.dispose_claim(request, apply=True)
        assert not (other / "cron").exists()
    assert cli("--request", str(request_file), "--apply")["status"] == "disposed"
    assert cli("--inspect", request["execution"]["id"])["status"] == "applied"
    assert disposition.inspect_disposition(request["execution"]["id"])["status"] == "applied"


@pytest.mark.linux_only
@pytest.mark.parametrize("landed", [False, True])
def test_pending_intent_blocks_attempts_and_forced_claims(interrupted, monkeypatch, landed):
    from cron import executions, jobs
    from cron import claim_disposition as disposition
    request, _, _ = interrupted
    save = jobs.save_jobs

    def uncertain(rows):
        if landed:
            save(rows)
        raise OSError("uncertain save")

    monkeypatch.setattr(jobs, "save_jobs", uncertain)
    with pytest.raises(OSError):
        disposition.dispose_claim(request, apply=True)
    monkeypatch.setattr(jobs, "save_jobs", save)
    with pytest.raises(ValueError, match="pending disposition"):
        executions.create_execution(request["job"]["id"], source="direct")
    with pytest.raises(ValueError, match="pending disposition"):
        jobs.claim_job_for_fire(request["job"]["id"], force=True)
    assert executions.list_executions(job_id=request["job"]["id"]) == [request["execution"]]


@pytest.mark.linux_only
@pytest.mark.parametrize("committed", [False, True])
def test_disposition_unknown_is_exempt_from_terminal_pruning(interrupted, monkeypatch, committed):
    from cron import executions, jobs
    from cron import claim_disposition as disposition
    request, _, _ = interrupted
    if committed:
        disposition.dispose_claim(request, apply=True)
    else:
        with monkeypatch.context() as faults:
            def fail_save(rows):
                raise OSError("save unavailable")
            faults.setattr(jobs, "save_jobs", fail_save)
            with pytest.raises(OSError):
                disposition.dispose_claim(request, apply=True)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    other = executions.create_execution("unrelated", source="direct")
    executions.finish_execution(other["id"], success=True)
    assert executions.get_execution(other["id"]) is None
    assert executions.get_execution(request["execution"]["id"]) == request["execution"]


@pytest.mark.linux_only
@pytest.mark.parametrize("malformation", ["wrapper", "duplicate_key"])
def test_disposition_refuses_ambiguous_registry(interrupted, malformation):
    from cron import jobs
    from cron.claim_disposition import dispose_claim
    request, _, _ = interrupted
    path = jobs._current_cron_store().jobs_file
    if malformation == "wrapper":
        store = json.loads(path.read_text())
        store["unknown_future_metadata"] = {"preserve": True}
        path.write_text(json.dumps(store))
    else:
        text = path.read_text()
        path.write_text(text.replace('"jobs":', '"jobs": [], "jobs":', 1))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        dispose_claim(request, apply=True)
    assert path.read_bytes() == before


@pytest.mark.linux_only
def test_prefire_attempt_queued_at_intent_is_refused_after_uncertain_save(interrupted, monkeypatch):
    from cron import executions, jobs
    from cron import claim_disposition as disposition
    import threading

    request, _, _ = interrupted
    began = threading.Event()
    outcome = []
    original_append = disposition._append

    def contender():
        began.set()
        try:
            executions.create_execution(request["job"]["id"], source="builtin")
        except ValueError as error:
            outcome.append(str(error))
        else:
            outcome.append("unexpectedly admitted")

    thread = threading.Thread(target=contender)

    def append_then_contend(path, event, **kwargs):
        original_append(path, event, **kwargs)
        if kwargs.get("first"):
            thread.start()
            assert began.wait(5)

    def fail_save(rows):
        raise OSError("no acknowledgement")

    monkeypatch.setattr(disposition, "_append", append_then_contend)
    monkeypatch.setattr(jobs, "save_jobs", fail_save)
    try:
        with pytest.raises(OSError):
            disposition.dispose_claim(request, apply=True)
    finally:
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(outcome) == 1 and "pending disposition" in outcome[0]
    assert executions.list_executions(job_id=request["job"]["id"]) == [request["execution"]]


@pytest.mark.linux_only
@pytest.mark.parametrize("failure", [False, None, "denied", "nested"])
def test_strict_registry_lock_cannot_degrade(interrupted, monkeypatch, failure):
    from cron import jobs
    def unavailable(*args):
        if failure == "denied":
            raise PermissionError("lock inaccessible")
        return False if failure == "nested" else failure
    monkeypatch.setattr(jobs, "_acquire_flock", unavailable)
    if failure == "nested":
        with jobs._jobs_lock():
            with pytest.raises(RuntimeError, match="cross-process lock required"):
                with jobs._jobs_lock(require_cross_process=True):
                    pytest.fail("entered degraded nested lock")
    else:
        with pytest.raises(RuntimeError, match="cross-process lock required"):
            with jobs._jobs_lock(require_cross_process=True):
                pytest.fail("entered degraded lock")
