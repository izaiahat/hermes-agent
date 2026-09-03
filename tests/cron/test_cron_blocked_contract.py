"""Focused contract tests for agent-authored [CRON_BLOCKED] reports."""
from __future__ import annotations

import cron.scheduler as scheduler


def _run(monkeypatch, *, final: str, no_agent: bool = False):
    calls: dict[str, list] = {"deliver": [], "mark": [], "finish": []}

    def fake_run_job(job, *, defer_agent_teardown=None, **_kwargs):
        return True, "raw output", final, None

    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args, **_kwargs: "/tmp/output")
    monkeypatch.setattr(
        scheduler, "_deliver_result",
        lambda _job, content, **_kwargs: calls["deliver"].append(content),
    )
    monkeypatch.setattr(
        scheduler, "mark_job_run",
        lambda *args, **kwargs: calls["mark"].append((args, kwargs)),
    )
    monkeypatch.setattr(
        scheduler, "finish_execution",
        lambda *args, **kwargs: calls["finish"].append((args, kwargs)),
    )
    monkeypatch.setattr(scheduler, "create_execution", lambda *_args, **_kwargs: {"id": "exec-1"})
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(scheduler, "_summarize_cron_failure_for_delivery", lambda *_args: "GENERIC FAILURE")

    ok = scheduler.run_one_job({
        "id": "blocked-job", "name": "domain job", "no_agent": no_agent,
        "deliver": "discord", "deliver_target": "channel-1",
    })
    return ok, calls


def test_leading_blocked_marker_delivers_domain_report_and_records_failed_blocked(monkeypatch):
    report = "BLOCKED_SCORER_INFRASTRUCTURE\n0 qualified; repair scorer; no-send"
    ok, calls = _run(monkeypatch, final=f"[CRON_BLOCKED]\n{report}")

    assert ok is False
    assert calls["deliver"] == [report]
    assert "[CRON_BLOCKED]" not in calls["deliver"][0]
    mark_args, mark_kwargs = calls["mark"][0]
    assert mark_args[1] is False
    assert mark_kwargs["status"] == "blocked"
    assert mark_args[2] == "Agent reported a blocked domain outcome"
    assert calls["finish"][0][1]["success"] is False
    assert calls["finish"][0][1]["error"] == "Agent reported a blocked domain outcome"


def test_mid_body_marker_does_not_trigger(monkeypatch):
    content = "Healthy domain report\nquoted token: [CRON_BLOCKED]\nall checks passed"
    ok, calls = _run(monkeypatch, final=content)

    assert ok is True
    assert calls["deliver"] == [content]
    assert calls["mark"][0][0][1] is True
    assert "status" not in calls["mark"][0][1]
    assert calls["finish"][0][1]["success"] is True


def test_marker_only_fails_closed_without_empty_delivery(monkeypatch):
    ok, calls = _run(monkeypatch, final="  [CRON_BLOCKED]  \n")

    assert ok is False
    assert calls["deliver"] == []
    assert calls["mark"][0][0][1] is False
    assert calls["mark"][0][1]["status"] == "blocked"
    assert "without a domain report" in calls["mark"][0][0][2]
    assert calls["finish"][0][1]["success"] is False


def test_no_agent_output_does_not_use_agent_blocked_contract(monkeypatch):
    content = "[CRON_BLOCKED]\nscript-owned text"
    ok, calls = _run(monkeypatch, final=content, no_agent=True)

    assert ok is True
    assert calls["deliver"] == [content]
    assert calls["mark"][0][0][1] is True


def test_healthy_output_unchanged(monkeypatch):
    content = "Healthy current-generation report"
    ok, calls = _run(monkeypatch, final=content)

    assert ok is True
    assert calls["deliver"] == [content]
    assert calls["finish"][0][1]["success"] is True
