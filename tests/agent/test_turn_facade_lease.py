"""Unit tests for agent.turn_facade_lease (admission + lease bracket)."""
import sqlite3
import threading
from types import SimpleNamespace

import pytest

import agent.turn_facade_lease as lease_module

from agent.turn_facade_lease import (
    LEASE_TTL_SECONDS,
    DurableTurnLease,
    admit_durable_turn_lease,
)


class _Db:
    def __init__(self, exists=True, acquired=True):
        self.exists = exists
        self.acquired = acquired
        self.events = []

    def get_session(self, session_id):
        return {"id": session_id} if self.exists else None

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("acquire", session_id, holder))
        return self.acquired

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.events.append(("release", session_id, holder))


def _agent(db, **overrides):
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        _persist_disabled=False,
        _interrupt_requested=False,
        _interrupt_message=None,
        _execution_thread_id=None,
        _session_turn_lease_refresh_interval=60.0,
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._emit_warning = agent.statuses.append
    agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = lambda: threading.Lock()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _admit(agent, history=None):
    return admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "cli"},
        conversation_history=history,
    )


def test_no_lease_without_durable_row_or_when_persist_disabled():
    seed = [{"role": "user", "content": "hi"}]
    admission = _admit(_agent(_Db(exists=False)), seed)
    assert admission.lease is None and admission.early_result is None
    assert admission.conversation_history is seed

    db = _Db()
    admission = _admit(_agent(db, _persist_disabled=True), seed)
    assert admission.lease is None and db.events == []


def test_admission_sets_holder_attrs_and_release_clears_them(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    db = _Db()
    agent = _agent(db)
    admission = _admit(agent)
    lease = admission.lease
    assert isinstance(lease, DurableTurnLease)
    assert agent._session_db_created is True
    assert agent._active_session_turn_lease_holder == lease.holder
    assert agent._active_session_turn_lease_ttl_seconds == LEASE_TTL_SECONDS
    assert lease.holder.startswith("pid=") and ":platform=cli" in lease.holder
    assert lease.watchdog is None and lease.timer_handles == []
    assert lease.is_turn_active() is False

    lease.stop_refresher()
    lease.join_threads()
    lease.clear_interrupt()
    lease.release()
    assert db.events == [("acquire", "s1", lease.holder), ("release", "s1", lease.holder)]
    assert agent._active_session_turn_lease_holder is None
    assert agent._active_session_turn_lease_ttl_seconds is None


def test_timeout_and_interrupt_early_results():
    agent = _agent(_Db(acquired=False))
    admission = _admit(agent, [{"role": "user", "content": "x"}])
    assert admission.lease is None
    assert admission.early_result["failed"] is True
    assert admission.early_result["error"] == "session_turn_lease_timeout:s1"
    # Stamped so UI descriptors show "session busy" instead of code="unknown".
    assert admission.early_result["failure_reason"] == "session_busy"
    assert admission.early_result["failure_retryable"] is True
    assert admission.early_result["messages"] == [{"role": "user", "content": "x"}]

    agent = _agent(_Db(acquired=False), _interrupt_requested=True, _interrupt_message="stop")
    agent.clear_interrupt = lambda: None
    admission = _admit(agent)
    assert admission.early_result["interrupted"] is True
    assert admission.early_result["interrupt_message"] == "stop"


def test_interrupt_turn_only_while_active():
    agent = _agent(_Db())
    calls = []
    agent.interrupt = lambda msg, **kw: calls.append(msg)
    lease = DurableTurnLease(agent, agent._session_db, "s1", "h")
    lease._interrupt_turn("lost")  # inactive: ignored
    assert calls == [] and lease.interrupt_message is None
    lease.turn_active = True
    lease._interrupt_turn("lost")
    assert calls == ["lost"] and lease.interrupt_message == "lost"
    lease.deactivate_after_liveness_abort()
    assert lease.stop.is_set() and lease.is_turn_active() is False


@pytest.mark.parametrize("error", ["database is locked", "database is busy"])
def test_transient_refresh_contention_does_not_interrupt(monkeypatch, error):
    clock = [100.0]
    monkeypatch.setattr(lease_module.time, "monotonic", lambda: clock[0])
    db = _Db()
    def locked(*args, **kwargs):
        raise sqlite3.OperationalError(error)
    db.refresh_session_turn_lease = locked
    agent = _agent(db)
    calls = []
    agent.interrupt = lambda message, **kw: calls.append(message)
    lease = DurableTurnLease(agent, db, "s1", "h")
    lease.turn_active = True
    clock[0] += 80.0  # one write-patience wait and a delayed scheduler tick
    assert lease.refresh_tick() is None
    assert calls == [] and lease.interrupt_message is None
    assert not lease.stop.is_set()
    db.refresh_session_turn_lease = lambda *args, **kwargs: True
    clock[0] += 60.0
    assert lease.refresh_tick() is None  # the next tick really retries
    assert calls == []


def test_repeated_contention_near_ttl_interrupts_before_expiry(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(lease_module.time, "monotonic", lambda: clock[0])
    db = _Db()
    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    db.refresh_session_turn_lease = locked
    agent = _agent(db)
    calls = []
    agent.interrupt = lambda message, **kw: calls.append((message, kw))
    lease = DurableTurnLease(agent, db, "s1", "h")
    lease.turn_active = True
    for elapsed in (80, 160, 239):
        clock[0] = 100.0 + elapsed
        assert lease.refresh_tick() is None
    assert calls == []
    clock[0] = 340.0  # TTL 300s, 60s safety margin
    assert lease.refresh_tick() is False
    assert len(calls) == 1 and calls[0][1]["hard_cancel"] is True
    assert calls[0][1]["tool_reason"] == "session turn lease lost"


def test_successful_refresh_resets_contention_deadline(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(lease_module.time, "monotonic", lambda: clock[0])
    db = _Db()
    agent = _agent(db)
    calls = []
    agent.interrupt = lambda message, **kw: calls.append(message)
    lease = DurableTurnLease(agent, db, "s1", "h")
    lease.turn_active = True
    clock[0] = 300.0
    assert lease.refresh_tick() is None
    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")
    db.refresh_session_turn_lease = locked
    clock[0] = 400.0
    assert lease.refresh_tick() is None
    assert calls == []


def test_holder_fenced_refresh_miss_still_interrupts():
    db = _Db()
    db.refresh_session_turn_lease = lambda *args, **kwargs: False
    agent = _agent(db)
    calls = []
    agent.interrupt = lambda message, **kw: calls.append((message, kw))
    lease = DurableTurnLease(agent, db, "s1", "h")
    lease.turn_active = True
    assert lease.refresh_tick() is False
    assert len(calls) == 1 and calls[0][1]["tool_reason"] == "session turn lease lost"


def test_non_contention_refresh_error_still_interrupts():
    db = _Db()
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")
    db.refresh_session_turn_lease = broken
    agent = _agent(db)
    calls = []
    agent.interrupt = lambda message, **kw: calls.append(message)
    lease = DurableTurnLease(agent, db, "s1", "h")
    lease.turn_active = True
    assert lease.refresh_tick() is False
    assert len(calls) == 1
