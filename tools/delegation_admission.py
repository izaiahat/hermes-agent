"""Measured capacity admission for child spawns (operator safety gate).

Refuse a spawn when the host cannot carry it: MemAvailable >= 4 GiB plus 768 MiB/new child,
SwapFree >= 1 GiB (if reported), memory PSI full avg10 < 2%, swap below 4 MiB/s
in at least one of two five-second samples, one-minute load below 1.5 times the logical CPU count.
The operator authorized tolerating minor swap traffic; occupied cold swap alone is not pressure.
After a prior swap-critical incident, both measured headroom and atomic host-wide
slots bound aggressive multi-TUI fan-out. `HERMES_PROC_ROOT` is a TEST SEAM
(a directory shaped like /proc) — a path, never a disable flag. There is no
bypass flag. A missing /proc file is an unknown measurement (skipped), except
MemAvailable, which must be readable.
"""
from __future__ import annotations

import os
import time
import json
import fcntl
import uuid
from pathlib import Path

MIN_MEM_AVAILABLE_KB = 4 * 1024 * 1024        # 4 GiB
MIN_SWAP_FREE_KB = 1024 * 1024               # 1 GiB, independent of swap I/O
MAX_PSI_FULL_AVG10 = 2.0                       # percent
MAX_SWAP_BYTES_PER_SECOND = 4 * 1024 * 1024  # sustained 4 MiB/s in both samples
# Remote-model children share a process, but tools can still create pressure.
# Leave up to 1.5 runnable lanes per logical CPU before refusing new work.
MAX_LOAD_1M = 1.5 * (max(1, len(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else max(1, os.cpu_count() or 1))


def _proc() -> str:
    return os.environ.get("HERMES_PROC_ROOT") or "/proc"


def _read(name: str) -> str | None:
    try:
        with open(os.path.join(_proc(), name), "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def mem_available_kb() -> int | None:
    text = _read("meminfo")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return None


def swap_free_kb() -> int | None:
    text = _read("meminfo")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("SwapFree:"):
            return int(line.split()[1])
    return None


def psi_full_avg10() -> float | None:
    text = _read("pressure/memory")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("full "):
            for tok in line.split():
                if tok.startswith("avg10="):
                    return float(tok[6:])
    return None


def swap_io() -> int | None:
    text = _read("vmstat")
    if text is None:
        return None
    total = 0
    for line in text.splitlines():
        if line.startswith(("pswpin ", "pswpout ")):
            total += int(line.split()[1])
    return total


def load_1m() -> float | None:
    text = _read("loadavg")
    if text is None:
        try:
            return os.getloadavg()[0]
        except OSError:
            return None
    return float(text.split()[0])


_CACHE: dict = {"at": 0.0, "result": None}
CACHE_SECONDS = 30.0


def admission_problem(*, sample_seconds: float = 5.0, sleep=time.sleep, now=time.time) -> str | None:
    """None when the host can carry another child; else the measured reason.

    Small swap transfers are permitted. Refuse only sustained material throughput in both samples;
    low available RAM and PSI remain independent vetoes. CACHE_SECONDS shares a sample within a batch.
    """
    t = now()
    if _CACHE["result"] is not None and t - _CACHE["at"] < CACHE_SECONDS:
        return _CACHE["result"] or None
    result = _measure(sample_seconds, sleep)
    _CACHE.update(at=t, result=result if result is not None else "")
    return result


def _measure(sample_seconds: float, sleep) -> str | None:
    mem = mem_available_kb()
    if mem is None:
        return "MemAvailable unreadable; refusing rather than guessing"
    if mem < MIN_MEM_AVAILABLE_KB:
        return f"MemAvailable {mem // 1024} MiB < 4 GiB"
    free_swap = swap_free_kb()
    if free_swap is not None and free_swap < MIN_SWAP_FREE_KB:
        return f"SwapFree {free_swap // 1024} MiB < 1 GiB"
    psi = psi_full_avg10()
    if psi is not None and psi >= MAX_PSI_FULL_AVG10:
        return f"memory PSI full avg10 {psi}% >= 2%"
    s0 = swap_io()
    if s0 is not None:
        sleep(sample_seconds)
        s1 = swap_io()
        page_bytes = os.sysconf("SC_PAGE_SIZE")
        sample_limit = MAX_SWAP_BYTES_PER_SECOND * max(sample_seconds, 0.001)
        if s1 is not None and (s1 - s0) * page_bytes >= sample_limit:
            sleep(sample_seconds)
            s2 = swap_io()
            if s2 is not None and (s2 - s1) * page_bytes >= sample_limit:
                return f"sustained swap >= 4 MiB/s in both {sample_seconds:g}s samples ({s1 - s0} then {s2 - s1} pages)"
    load = load_1m()
    if load is not None and load >= MAX_LOAD_1M:
        return f"one-minute load {load:.2f} >= {MAX_LOAD_1M}"
    return None


# Every TUI process takes the same flock. Keep the file inode (never unlink it).
# A PID plus kernel start tick makes leases reclaimable after crashes and PID reuse.
def _budget_path() -> Path:
    # All named profiles and external one-shots share the physical host pool.
    # A profile-scoped Hermes home here would multiply the four slots.
    return Path.home() / ".hermes" / "cache" / "delegation-host-budget.json"


def _start_tick(pid: int) -> str | None:
    try:
        return Path(f"{_proc()}/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, TypeError):
        return None


class HostLease:
    def __init__(self, token: str):
        self.token = token

    def release(self) -> None:
        token, self.token = self.token, ""
        if token:
            _host_budget_edit(release=token)


def _host_budget_edit(*, count: int = 0, limit: int = 0, release: str = "") -> tuple[list[HostLease] | None, int]:
    path = _budget_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read()
        try:
            rows = json.loads(raw) if raw else []
            if not isinstance(rows, list):
                raise ValueError("invalid host budget")
        except ValueError as exc:
            raise RuntimeError("host delegation budget corrupt; refusing reservation") from exc
        rows = [r for r in rows if isinstance(r, dict) and r.get("pid") is not None
                and _start_tick(r["pid"]) == str(r.get("start"))]
        if release:
            rows = [r for r in rows if r.get("token") != release]
        active = len(rows)
        if count and active + count > limit:
            leases = None
        else:
            tick = _start_tick(os.getpid())
            if count and tick is None:
                raise RuntimeError("cannot establish host reservation owner")
            tokens = [uuid.uuid4().hex for _ in range(count)]
            rows.extend({"pid": os.getpid(), "start": tick, "token": t} for t in tokens)
            leases = [HostLease(t) for t in tokens]
        fh.seek(0)
        fh.truncate()
        json.dump(rows, fh)
        fh.flush()
        os.fsync(fh.fileno())
        return leases, active


def host_child_limit() -> int:
    """CPU lanes (up to twice affinity CPUs), bounded by 1.5 GiB/child after a 4 GiB reserve."""
    text = _read("meminfo") or ""
    total = next((int(line.split()[1]) for line in text.splitlines() if line.startswith("MemTotal:")), 0)
    if not total:
        raise RuntimeError("MemTotal unreadable; refusing host reservation")
    cpu_lanes = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(4, 2 * cpu_lanes, (total - MIN_MEM_AVAILABLE_KB) // (1536 * 1024)))


def try_reserve_host_children(count: int) -> tuple[list[HostLease] | None, int, int]:
    """Atomic cross-process reservation; retain 4 GiB plus 768 MiB per new child."""
    if count < 1:
        raise ValueError("count must be positive")
    mem = mem_available_kb()
    if mem is None or mem - MIN_MEM_AVAILABLE_KB < count * 768 * 1024:
        return None, 0, 0
    limit = host_child_limit()
    leases, active = _host_budget_edit(count=count, limit=limit)
    return leases, active, limit
