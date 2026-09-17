"""The capacity admission gate that makes delegation width 8 safe.

Operator decision 2026-09-16 raised the per-call child width from 5 to 8
(receipt ops/linear/approvals/OPERATOR-DECISION-20260916-delegation-width-8.json)
on the condition that the host itself be measured before every spawn. These
tests drive the gate against a /proc-shaped directory through HERMES_PROC_ROOT —
a test seam, never a bypass — and cover the delegate_task wiring, which must
fail CLOSED when the gate module cannot be imported.
"""
from __future__ import annotations

import builtins
import threading
from unittest.mock import MagicMock

import pytest

from tools import delegation_admission as admission

GIB_KB = 1024 * 1024


def write_proc(root, *, mem_kb=32 * GIB_KB, psi=0.0, load=1.0, swap=(0, 0, 0)):
    (root / "meminfo").write_text(
        f"MemTotal:       65000000 kB\nMemAvailable:   {mem_kb} kB\n", encoding="utf-8"
    )
    pressure = root / "pressure"
    pressure.mkdir(exist_ok=True)
    (pressure / "memory").write_text(
        "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
        f"full avg10={psi:.2f} avg60=0.00 avg300=0.00 total=0\n",
        encoding="utf-8",
    )
    (root / "loadavg").write_text(f"{load:.2f} 1.00 1.00 1/100 1\n", encoding="utf-8")
    (root / "vmstat").write_text(f"pswpin {swap[0]}\npswpout 0\n", encoding="utf-8")
    return root


@pytest.fixture
def proc_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PROC_ROOT", str(tmp_path))
    admission._CACHE.update(at=0.0, result=None)
    yield tmp_path
    admission._CACHE.update(at=0.0, result=None)


def verdict(**kwargs):
    return admission.admission_problem(sample_seconds=0, sleep=lambda _s: None, **kwargs)


@pytest.mark.real_capacity_admission
class TestThresholds:
    def test_a_healthy_host_admits(self, proc_root):
        write_proc(proc_root)
        assert verdict() is None

    def test_low_memory_refuses_and_names_the_measure(self, proc_root):
        write_proc(proc_root, mem_kb=4 * GIB_KB)
        assert "MemAvailable" in verdict()

    def test_memory_pressure_refuses(self, proc_root):
        write_proc(proc_root, psi=2.5)
        assert "PSI" in verdict()

    def test_high_load_refuses(self, proc_root):
        write_proc(proc_root, load=12.0)
        assert "load" in verdict()

    def test_unreadable_memavailable_refuses_rather_than_guessing(self, proc_root):
        write_proc(proc_root)
        (proc_root / "meminfo").unlink()
        assert "unreadable" in verdict()

    def test_one_quiet_sample_is_housekeeping_not_thrash(self, proc_root):
        """WORKER-STANDARD 7 refuses only when swap moves in BOTH samples."""
        write_proc(proc_root)
        reads = iter([10, 11, 11])

        def moving_once():
            return next(reads)

        original = admission.swap_io
        admission.swap_io = moving_once
        try:
            assert verdict() is None
        finally:
            admission.swap_io = original

    def test_swap_moving_in_both_samples_refuses(self, proc_root):
        write_proc(proc_root)
        reads = iter([10, 11, 13])
        original = admission.swap_io
        admission.swap_io = lambda: next(reads)
        try:
            assert "swap moving in both" in verdict()
        finally:
            admission.swap_io = original

    def test_the_verdict_is_cached_so_one_batch_samples_once(self, proc_root):
        write_proc(proc_root)
        calls = []
        original = admission._measure
        admission._measure = lambda *a, **k: calls.append(1) or None
        try:
            admission.admission_problem(sleep=lambda _s: None)
            admission.admission_problem(sleep=lambda _s: None)
        finally:
            admission._measure = original
        assert len(calls) == 1


def _mock_parent():
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "test-key"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


@pytest.mark.real_capacity_admission
class TestDelegateWiring:
    GOAL = "Refactor the login handler to use the new session helper"

    def test_a_refusal_blocks_the_spawn_with_the_measured_reason(self, proc_root):
        from tools.delegate_tool import delegate_task

        write_proc(proc_root, mem_kb=1 * GIB_KB)
        admission._CACHE.update(at=0.0, result=None)
        result = str(delegate_task(goal=self.GOAL, parent_agent=_mock_parent()))
        assert "capacity admission gate" in result
        assert "MemAvailable" in result

    def test_a_missing_gate_module_fails_closed(self, proc_root, monkeypatch):
        from tools import delegate_tool

        write_proc(proc_root)
        real_import = builtins.__import__

        def refuse_admission(name, globals=None, locals=None, fromlist=(), level=0):
            if name.endswith("delegation_admission"):
                raise ImportError("gate module removed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", refuse_admission)
        result = str(delegate_tool.delegate_task(goal=self.GOAL, parent_agent=_mock_parent()))
        assert "is missing" in result
