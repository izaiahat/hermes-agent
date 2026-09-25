"""Focused, offline delegation routing and fleet capacity regression tests."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import delegation_admission as admission
from tools import delegate_tool as delegate
from tools.delegate_tool_child_run import _route_receipt
from tools.delegate_tool_dispatch import _Batch


def test_mixed_task_routes_and_effort_receipts(monkeypatch):
    parent = SimpleNamespace(model="gpt-6-sol", provider="openai-codex", reasoning_config={"enabled": True, "effort": "medium"})
    calls = []

    def official_resolve(requested, target_model):
        calls.append((requested, target_model))
        return {"model": target_model or "claude-opus-5-5", "provider": requested,
                "base_url": "https://api.anthropic.com" if requested == "anthropic" else "https://chatgpt.com/backend-api/codex",
                "api_key": "secret-not-in-receipt", "api_mode": "anthropic_messages" if requested == "anthropic" else "codex_responses"}

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=official_resolve):
        tasks = [
            {"goal": "A", "model": "gpt-6-sol", "reasoning_effort": "low"},
            {"goal": "B", "provider": "anthropic", "model": "claude-opus-5-5", "reasoning_effort": "high"},
        ]
        default = {"model": "gpt-6-sol", "provider": "openai-codex", "base_url": None, "api_key": None,
                   "api_mode": None, "request_overrides": None}
        delegate._resolve_task_routes(tasks, {"provider": "openai-codex", "base_url": "https://old", "api_key": "old"}, parent, default)
    assert calls == [("openai-codex", "gpt-6-sol"), ("anthropic", "claude-opus-5-5")]
    assert tasks[1]["_resolved_route"]["base_url"] == "https://api.anthropic.com"
    assert tasks[1]["_routing_cfg"].get("api_key") is None
    children = []

    def build(**kw):
        child = SimpleNamespace(model=kw["model"], provider=kw["override_provider"],
                                reasoning_config={"enabled": True, "effort": kw["task_reasoning_effort"]})
        children.append(child)
        return child

    with patch.object(delegate, "_build_child_preserving_parent_tools", side_effect=build):
        built, err = delegate._build_children(tasks, [None, None], default, top_role="leaf", max_iterations=1,
            parent_agent=parent, routing_cfg={}, live_deleg_id=None, live_writers=[])
    assert err is None and len(built) == 2
    assert [_route_receipt(c)["effective"]["reasoning_effort"] for c in children] == ["low", "high"]
    assert _route_receipt(children[1])["requested"]["provider"] == "anthropic"
    assert "secret" not in json.dumps([_route_receipt(c) for c in children])


def _reserve_in_process(path, barrier, finished, queue):
    from tools import delegation_admission as gate
    gate._budget_path = lambda: Path(path)
    gate.host_child_limit = lambda: 2
    barrier.wait()
    leases, active, limit = gate.try_reserve_host_children(2)
    queue.put((leases is not None, active, limit))
    queue.close()
    queue.join_thread()
    finished.wait(15)
    if leases:
        # Exit without releasing: next caller must reclaim stale PID reservations.
        os._exit(0)


def test_cross_parent_atomic_capacity_and_stale_reclaim(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    monkeypatch.setattr(admission, "_budget_path", lambda: path)
    monkeypatch.setattr(admission, "host_child_limit", lambda: 2)
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(2)
    finished = ctx.Event()
    q = ctx.Queue()
    workers = [ctx.Process(target=_reserve_in_process, args=(str(path), barrier, finished, q)) for _ in range(2)]
    for worker in workers:
        worker.start()
    outcomes = [q.get(timeout=15) for _ in workers]
    finished.set()
    for worker in workers:
        worker.join(timeout=15)
    assert sorted(outcome[0] for outcome in outcomes) == [False, True]
    # The winning process has exited. A new parent reclaims both stale slots.
    leases, active, limit = admission.try_reserve_host_children(2)
    assert leases is not None and (active, limit) == (0, 2)
    for lease in leases:
        lease.release()
        lease.release()
    assert json.loads(path.read_text()) == []


def test_rolling_slots_and_leaf_depth(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "_budget_path", lambda: tmp_path / "budget.json")
    monkeypatch.setattr(admission, "host_child_limit", lambda: 2)
    first, _, _ = admission.try_reserve_host_children(2)
    assert first is not None
    denied, active, limit = admission.try_reserve_host_children(1)
    assert denied is None and (active, limit) == (2, 2)
    first[0].release()
    replacement, active, limit = admission.try_reserve_host_children(1)
    assert replacement is not None and (active, limit) == (1, 2)
    first[1].release()
    replacement[0].release()
    from tools.delegate_tool_config import _get_max_spawn_depth
    with patch("tools.delegate_tool._load_config", return_value={"max_spawn_depth": 99}):
        assert _get_max_spawn_depth() == 1
    leaf = SimpleNamespace(_delegate_depth=1)
    with patch.object(delegate, "is_spawn_paused", return_value=False):
        assert "depth limit" in delegate.delegate_task(goal="no recursion", parent_agent=leaf)


def test_finished_child_releases_slot_before_sibling_finishes(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "_budget_path", lambda: tmp_path / "budget.json")
    monkeypatch.setattr(admission, "host_child_limit", lambda: 2)
    leases, _, _ = admission.try_reserve_host_children(2)
    assert leases is not None
    child = SimpleNamespace(_delegate_capacity_lease=leases[0])
    parent = SimpleNamespace()
    batch = _Batch([], [], parent, {}, None, "leaf", 2, None, [], [], "", "", None, None, False, 0)
    with patch.object(delegate, "_run_single_child", return_value={"status": "completed"}):
        assert batch.run_child(0, {"goal": "done"}, child)["status"] == "completed"
    rolling, active, limit = admission.try_reserve_host_children(1)
    assert rolling is not None and (active, limit) == (1, 2)
    rolling[0].release()
    leases[1].release()


def test_failed_rolling_batch_does_not_close_existing_siblings():
    from unittest.mock import Mock
    existing = SimpleNamespace(close=Mock())
    partial = SimpleNamespace(close=Mock())
    parent = SimpleNamespace(_active_children=[existing, partial])
    delegate._release_partial_children(parent, [(0, {}, partial)])
    assert parent._active_children == [existing]
    existing.close.assert_not_called()
    partial.close.assert_called_once()


def test_timed_out_worker_keeps_slot_until_actual_exit(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from tools.delegate_tool_child_run import _defer_close_after_timeout
    monkeypatch.setattr(admission, "_budget_path", lambda: tmp_path / "budget.json")
    monkeypatch.setattr(admission, "host_child_limit", lambda: 1)
    leases, _, _ = admission.try_reserve_host_children(1)
    assert leases is not None
    child = SimpleNamespace(_delegate_capacity_lease=leases[0], close=lambda: None)
    future = Future()
    batch = _Batch([], [], SimpleNamespace(), {}, None, "leaf", 1, None, [], [], "", "", None, None, False, 0)
    def timed_out(**kwargs):
        _defer_close_after_timeout(child, future)
        return {"status": "timeout"}
    with patch.object(delegate, "_run_single_child", side_effect=timed_out):
        batch.run_child(0, {"goal": "timeout"}, child)
    assert admission.try_reserve_host_children(1)[0] is None
    future.set_result(None)
    next_leases, _, _ = admission.try_reserve_host_children(1)
    assert next_leases is not None
    next_leases[0].release()
