"""Regression for GLP-346: an async batch whose children route to different models must report each child's
model in the completion header, not the batch default from ``creds["model"]``."""

import time
from types import SimpleNamespace

import tools.async_delegation as async_delegation
from tools.delegate_tool_dispatch import _Batch, _dispatch_unit
from tools.process_registry_notifications import format_process_notification


def _batch(children, batch_model):
    tasks = [{"goal": f"goal {i}"} for i in range(len(children))]
    return _Batch(
        task_list=tasks, children=[(i, tasks[i], c) for i, c in enumerate(children)], parent_agent=None,
        creds={"model": batch_model}, context=None, top_role="leaf", max_children=3, live_deleg_id="deleg_mix",
        live_writers=[], live_paths=[], origin_wake_sid="s", origin_ui_session_id="", origin_owner_transport=None,
        origin_owner_session_record=None, origin_session_history_delivery=False, overall_start=time.time(),
    )


def test_mixed_model_batch_header_shows_routed_models(monkeypatch):
    captured = {}
    monkeypatch.setattr(async_delegation, "dispatch_async_delegation_batch",
                        lambda **kw: captured.update(kw) or {"status": "dispatched"})
    astra, unrouted = SimpleNamespace(model="gpt-6-astra-900k"), SimpleNamespace(model=None)
    _dispatch_unit(_batch([astra, unrouted], "gpt-6.1-sol"), "deleg_mix", None, {})

    header_model = captured["model"]
    assert "gpt-6-astra-900k" in header_model  # the routed child's real model
    assert "gpt-6.1-sol" in header_model  # unrouted child falls back to the batch model

    evt = {
        "type": "async_delegation", "delegation_id": "deleg_mix", "is_batch": True, "status": "completed",
        "goals": ["goal 0", "goal 1"], "role": "leaf", "model": header_model, "completed_at": time.time(),
        "results": [
            {"task_index": 0, "status": "completed", "summary": "a", "model": astra.model},
            {"task_index": 1, "status": "completed", "summary": "b", "model": "gpt-6.1-sol"},
        ],
    }
    text = format_process_notification(evt)
    header_line = next(line for line in text.splitlines() if "Model:" in line)
    assert "gpt-6-astra-900k" in header_line
    task1 = next(line for line in text.splitlines() if "TASK 1/2" in line)
    assert "model=gpt-6-astra-900k" in task1
