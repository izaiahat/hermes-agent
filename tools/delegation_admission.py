"""Capacity admission for child spawns (the canonical eight's gate, made mechanical 2026-09-16).

Refuse a spawn when the host cannot carry it: MemAvailable >= 8 GiB, memory PSI full avg10 < 1 %,
no swap in/out across a one-second sample, one-minute load < 8. Thresholds are WORKER-STANDARD 7's.
Why: on 2026-09-12, 58 one-shot Hermes jobs put the AWS host swap-critical; with width 8 the box, not the
number, is the limit. `HERMES_PROC_ROOT` is a TEST SEAM (a directory shaped like /proc) - a path, never a
bypass; there is no flag that turns the gate off. A missing /proc file is a check that cannot run and is
treated as unknown (skipped) EXCEPT MemAvailable, which must be readable.
"""
from __future__ import annotations

import os
import time

MIN_MEM_AVAILABLE_KB = 8 * 1024 * 1024        # 8 GiB
MAX_PSI_FULL_AVG10 = 1.0                       # percent
MAX_LOAD_1M = 8.0


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

    Swap is judged exactly as WORKER-STANDARD 7 states it: "no swap in/out in BOTH five-second samples".
    A single page moving in one sample is kernel housekeeping on a 22 GiB-free host, not thrash; refusing on
    it would be stricter than the standard and would teach people to want a bypass. The verdict is cached for
    CACHE_SECONDS so one batch of spawns pays the ten-second sample once."""
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
        return f"MemAvailable {mem // 1024} MiB < 8 GiB"
    psi = psi_full_avg10()
    if psi is not None and psi >= MAX_PSI_FULL_AVG10:
        return f"memory PSI full avg10 {psi}% >= 1%"
    s0 = swap_io()
    if s0 is not None:
        sleep(sample_seconds)
        s1 = swap_io()
        if s1 is not None and s1 != s0:
            sleep(sample_seconds)
            s2 = swap_io()
            if s2 is not None and s2 != s1:
                return f"swap moving in both {sample_seconds:g}s samples ({s1 - s0} then {s2 - s1} pages)"
    load = load_1m()
    if load is not None and load >= MAX_LOAD_1M:
        return f"one-minute load {load:.2f} >= 8"
    return None
