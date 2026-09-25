"""The capacity admission gate for the 12-wide / 16-host-slot policy.

Operator decision 2026-09-23 permits more parallel native children while keeping
atomic host reservations and measured pressure refusals. These tests use a
/proc-shaped directory through HERMES_PROC_ROOT (test seam, never bypass).
"""
from __future__ import annotations

import builtins
import threading
from unittest.mock import MagicMock

import pytest

from tools import delegation_admission as admission

GIB_KB = 1024 * 1024


def write_proc(root, *, mem_kb=32 * GIB_KB, swap_free_kb=2 * GIB_KB,
               psi=0.0, load=1.0, swap=(0, 0, 0)):
    (root / "meminfo").write_text(
        f"MemTotal:       65000000 kB\nMemAvailable:   {mem_kb} kB\n"
        f"SwapFree:      {swap_free_kb} kB\n", encoding="utf-8"
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
        write_proc(proc_root, mem_kb=4 * GIB_KB - 1)
        assert "MemAvailable" in (verdict() or "")

    @pytest.mark.parametrize("free_kb,refused", [(GIB_KB - 1, True), (GIB_KB, False), (2 * GIB_KB, False)])
    def test_swap_free_floor_is_independent_of_quiet_swap_io(self, proc_root, free_kb, refused):
        write_proc(proc_root, swap_free_kb=free_kb)
        result = verdict()
        assert (result is not None) is refused
        if refused:
            assert result is not None and 'SwapFree' in result and '1 GiB' in result

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

    @pytest.mark.parametrize("pages_per_second,refused", [(0, False), (255, False), (1023, False), (1024, True), (2048, True)])
    def test_only_sustained_material_swap_refuses(self, proc_root, monkeypatch, pages_per_second, refused):
        write_proc(proc_root)
        monkeypatch.setattr(admission.os, 'sysconf', lambda _key: 4096)
        reads = iter([10, 10 + pages_per_second * 5, 10 + pages_per_second * 10])
        original = admission.swap_io
        admission.swap_io = lambda: next(reads)
        try:
            result = admission.admission_problem(sample_seconds=5, sleep=lambda _s: None)
            if refused:
                assert result is not None and "sustained swap" in result
            else:
                assert result is None
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

    @pytest.mark.parametrize("psi,refused", [(1.99, False), (2.00, True)])
    def test_psi_boundary(self, proc_root, psi, refused):
        write_proc(proc_root, psi=psi)
        assert (verdict() is not None) is refused

    @pytest.mark.parametrize("load,refused", [(11.99, False), (12.00, True)])
    def test_eight_core_load_boundary(self, proc_root, monkeypatch, load, refused):
        write_proc(proc_root, load=load)
        monkeypatch.setattr(admission, "MAX_LOAD_1M", 12.0)
        assert (verdict() is not None) is refused

    def test_four_gib_floor_and_four_child_available_reserve(self, proc_root, tmp_path, monkeypatch):
        monkeypatch.setattr(admission, "_budget_path", lambda: tmp_path / "budget.json")
        monkeypatch.setattr(admission, "_start_tick", lambda _pid: "test")
        monkeypatch.setattr(admission.os, "sched_getaffinity", lambda _pid: set(range(8)))
        write_proc(proc_root, mem_kb=4 * GIB_KB - 1)
        assert "4 GiB" in (verdict() or "")
        admission._CACHE.update(at=0.0, result=None)
        write_proc(proc_root, mem_kb=4 * GIB_KB)
        assert verdict() is None
        # Four-child batch requires 4 GiB + 4 * 768 MiB available.
        write_proc(proc_root, mem_kb=7 * GIB_KB - 1)
        assert admission.try_reserve_host_children(4)[0] is None
        write_proc(proc_root, mem_kb=7 * GIB_KB)
        assert admission.host_child_limit() == 4
        leases, active, limit = admission.try_reserve_host_children(4)
        assert leases is not None and (active, limit) == (0, 4)
        assert admission.try_reserve_host_children(1)[0] is None
        for lease in leases:
            lease.release()
        assert admission.try_reserve_host_children(5)[0] is None

    def test_host_memory_sizing_and_unreadable_total_fail_closed(self, proc_root, monkeypatch):
        monkeypatch.setattr(admission.os, "sched_getaffinity", lambda _pid: set(range(8)))
        write_proc(proc_root)
        (proc_root / "meminfo").write_text(
            f"MemTotal: {31 * GIB_KB} kB\nMemAvailable: {19 * GIB_KB} kB\n"
        )
        assert admission.host_child_limit() == 4
        (proc_root / "meminfo").write_text(f"MemAvailable: {19 * GIB_KB} kB\n")
        with pytest.raises(RuntimeError, match="MemTotal unreadable"):
            admission.host_child_limit()


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
