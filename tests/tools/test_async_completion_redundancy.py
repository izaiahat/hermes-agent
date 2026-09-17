"""Regression tests for exact-once async completion delivery."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

import tools.async_delegation as ad
from agent.tool_executor import _record_async_artifact_access
from tools.process_registry import ProcessRegistry


@pytest.fixture(autouse=True)
def _isolated_observations(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    ad._artifact_observations.clear()
    yield
    ad._artifact_observations.clear()


def _event(path, *, status="completed", batch_id="deleg_test", session_key="session-a"):
    result = {
        "status": status,
        "summary": f"Wrote {path}",
        "tool_trace": [{
            "tool": "write_file",
            "status": "ok",
            "input_summary": {"targets": {"path": str(path)}},
        }],
    }
    return {
        "type": "async_delegation",
        "delegation_id": batch_id,
        "batch_id": batch_id,
        "session_key": session_key,
        "status": status,
        "dispatched_at": 50.0,
        "result": result,
        "results": [result],
    }


def _full_read_result(content="1|done"):
    return json.dumps({"content": content, "total_lines": 1, "truncated": False})


def test_consumed_artifact_completion_is_acknowledged_exactly_once(tmp_path):
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("done", encoding="utf-8")
    os.utime(artifact, (100.0, 100.0))
    event = _event(artifact)

    ad._persist_dispatch(
        {
            "delegation_id": event["delegation_id"],
            "session_key": event["session_key"],
            "dispatched_at": event["dispatched_at"],
            "started_at": event["dispatched_at"],
            "status": "running",
            "task_count": 1,
            "completed_count": 0,
            "results": [],
        }
    )
    ad._persist_completion(event, event["result"])
    ad.record_parent_artifact_access(
        session_keys=[event["session_key"]],
        tool_name="read_file",
        args={"path": str(artifact)},
        result=_full_read_result(),
        observed_at=200.0,
    )

    assert ad.consume_redundant_completion(
        event, consumer="test", session_keys=[event["session_key"]]
    )
    row = ad.get_durable_delegation(event["delegation_id"])
    assert row is not None
    assert row["delivery_state"] == "delivered"
    assert row["delivery_attempts"] == 1

    # A duplicate callback copy remains suppressible but cannot re-claim delivery.
    assert ad.consume_redundant_completion(
        event, consumer="second", session_keys=[event["session_key"]]
    )
    row = ad.get_durable_delegation(event["delegation_id"])
    assert row["delivery_attempts"] == 1


def test_unseen_artifact_preserves_completion(tmp_path):
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("done", encoding="utf-8")
    event = _event(artifact)

    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None
    assert not ad.consume_redundant_completion(
        event, consumer="test", session_keys=["session-a"]
    )


def test_failure_is_never_suppressed_even_when_artifact_was_read(tmp_path):
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("partial", encoding="utf-8")
    event = _event(artifact, status="failed")
    ad.record_parent_artifact_access(
        session_keys=["session-a"], tool_name="read_file",
        args={"path": str(artifact)},
        result=_full_read_result("1|partial"),
    )

    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None


def test_partial_batch_access_preserves_material_unseen_result(tmp_path):
    first = tmp_path / "ONE.md"
    second = tmp_path / "TWO.md"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    event = _event(first)
    event["results"].append(
        {
            "status": "completed",
            "summary": f"Wrote {second}",
            "tool_trace": [{
                "tool": "write_file", "status": "ok",
                "input_summary": {"targets": {"path": str(second)}},
            }],
        }
    )
    ad.record_parent_artifact_access(
        session_keys=["session-a"], tool_name="read_file",
        args={"path": str(first)},
        result=_full_read_result("1|one"),
    )

    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None


def test_observation_must_be_same_session_after_dispatch_and_after_latest_write(tmp_path):
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("done", encoding="utf-8")
    os.utime(artifact, (100.0, 100.0))
    event = _event(artifact)

    ad.record_parent_artifact_access(
        session_keys=["other-session"], tool_name="read_file",
        args={"path": str(artifact)}, result=_full_read_result(), observed_at=200.0,
    )
    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None

    ad.record_parent_artifact_access(
        session_keys=["session-a"], tool_name="read_file",
        args={"path": str(artifact)}, result=_full_read_result(), observed_at=40.0,
    )
    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None

    ad.record_parent_artifact_access(
        session_keys=["session-a"], tool_name="read_file",
        args={"path": str(artifact)}, result=_full_read_result(), observed_at=90.0,
    )
    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None

    ad.record_parent_artifact_access(
        session_keys=["session-a"], tool_name="read_file",
        args={"path": str(artifact)}, result=_full_read_result(), observed_at=200.0,
    )
    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) == "artifact-consumed"


@pytest.mark.parametrize(
    ("args_extra", "result"),
    [
        pytest.param(
            {"limit": 1},
            json.dumps({"content": "1|preview", "total_lines": 2, "truncated": True}),
            id="partial-limit-read",
        ),
        pytest.param(
            {"offset": 2},
            json.dumps({"content": "2|tail", "total_lines": 2, "truncated": False}),
            id="non-start-offset",
        ),
        pytest.param(
            {},
            json.dumps({
                "content": "1|clipped",
                "total_lines": 1,
                "truncated": True,
                "truncated_by": "bytes",
            }),
            id="character-budget-truncation",
        ),
        pytest.param(
            {},
            "<persisted-output>\nPreview only\n</persisted-output>",
            id="spillover-preview",
        ),
        pytest.param(
            {},
            json.dumps({"total_lines": 1, "truncated": False}),
            id="body-missing",
        ),
    ],
)
def test_incomplete_parent_read_never_suppresses_completion(
    tmp_path, args_extra, result
):
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("whole body", encoding="utf-8")
    event = _event(artifact)

    ad.record_parent_artifact_access(
        session_keys=["session-a"],
        tool_name="read_file",
        args={"path": str(artifact), **args_extra},
        result=result,
        observed_at=200.0,
    )

    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None
    assert ad._artifact_observations == {}


def test_patch_only_never_qualifies_and_invalidates_a_prior_full_read(tmp_path):
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("whole body", encoding="utf-8")
    os.utime(artifact, (100.0, 100.0))
    event = _event(artifact)

    ad.record_parent_artifact_access(
        session_keys=["session-a"],
        tool_name="patch",
        args={"path": str(artifact), "old_string": "body", "new_string": "result"},
        result="successful patch",
        observed_at=150.0,
    )
    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None

    ad.record_parent_artifact_access(
        session_keys=["session-a"],
        tool_name="read_file",
        args={"path": str(artifact)},
        result=_full_read_result("1|whole body"),
        observed_at=200.0,
    )
    assert ad.redundant_completion_reason(
        event, session_keys=["session-a"]
    ) == "artifact-consumed"

    ad.record_parent_artifact_access(
        session_keys=["session-a"],
        tool_name="patch",
        args={"path": artifact.name, "old_string": "body", "new_string": "result"},
        result="successful patch",
        observed_at=210.0,
    )
    assert ad.redundant_completion_reason(event, session_keys=["session-a"]) is None


def test_full_parent_write_establishes_complete_artifact_knowledge(tmp_path):
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("parent replacement", encoding="utf-8")
    os.utime(artifact, (100.0, 100.0))
    event = _event(artifact)

    ad.record_parent_artifact_access(
        session_keys=["session-a"],
        tool_name="write_file",
        args={"path": str(artifact), "content": "parent replacement"},
        result=json.dumps({"verified": True}),
        observed_at=200.0,
    )

    assert ad.redundant_completion_reason(
        event, session_keys=["session-a"]
    ) == "artifact-consumed"


def test_empty_success_is_suppressed_but_summary_only_success_is_preserved():
    empty = {
        "type": "async_delegation",
        "batch_id": "deleg_empty",
        "session_key": "session-a",
        "status": "completed",
        "result": {"summary": "", "results": [], "files_written": []},
        "results": [],
    }
    material = {
        **empty,
        "batch_id": "deleg_material",
        "results": [{"status": "completed", "summary": "new answer"}],
    }

    assert ad.redundant_completion_reason(empty) == "empty-success"
    assert ad.redundant_completion_reason(material) is None


def test_tool_executor_records_only_successful_parent_file_access(monkeypatch):
    import agent.delegation_context as delegation_context
    import tools.approval as approval

    calls = []
    monkeypatch.setattr(approval, "get_current_session_key", lambda default="": "route-a")
    monkeypatch.setattr(ad, "record_parent_artifact_access", lambda **kw: calls.append(kw))
    monkeypatch.setattr(delegation_context, "is_delegated_child_context", lambda: False)
    agent = SimpleNamespace(session_id="ui-a")

    _record_async_artifact_access(
        agent, "read_file", {"path": "/tmp/result.md"}, _full_read_result(),
        failed=False, blocked=False
    )
    _record_async_artifact_access(
        agent, "read_file", {"path": "/tmp/failed.md"}, _full_read_result(),
        failed=True, blocked=False
    )
    _record_async_artifact_access(
        agent, "read_file", {"path": "/tmp/blocked.md"}, _full_read_result(),
        failed=False, blocked=True
    )
    monkeypatch.setattr(delegation_context, "is_delegated_child_context", lambda: True)
    _record_async_artifact_access(
        agent, "read_file", {"path": "/tmp/child.md"}, _full_read_result(),
        failed=False, blocked=False
    )

    assert calls == [{
        "session_keys": ["route-a", "ui-a"],
        "tool_name": "read_file",
        "args": {"path": "/tmp/result.md"},
        "result": _full_read_result(),
    }]


def test_registry_drain_filters_consumed_event_but_keeps_unseen_result(tmp_path):
    registry = ProcessRegistry()
    consumed_path = tmp_path / "CONSUMED.md"
    unseen_path = tmp_path / "UNSEEN.md"
    consumed_path.write_text("consumed", encoding="utf-8")
    unseen_path.write_text("unseen", encoding="utf-8")
    consumed = _event(consumed_path, batch_id="deleg_consumed_drain")
    unseen = _event(unseen_path, batch_id="deleg_unseen_drain")
    ad.record_parent_artifact_access(
        session_keys=["session-a"],
        tool_name="read_file",
        args={"path": str(consumed_path)},
        result=_full_read_result("1|consumed"),
    )
    registry.completion_queue.put(consumed)
    registry.completion_queue.put(unseen)

    drained = registry.drain_notifications(
        session_key="session-a", owns_event=lambda _event: True
    )

    assert [event["delegation_id"] for event, _text in drained] == [
        "deleg_unseen_drain"
    ]
