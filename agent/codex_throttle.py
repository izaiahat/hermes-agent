"""Cross-process concurrency + rate gate for OpenAI Codex requests.

The ChatGPT / Codex *subscription* backend (``https://chatgpt.com/backend-api/codex``)
enforces a tight **per-account** burst / concurrency limit.  When it is exceeded it
returns ``HTTP 429 {"detail": "Rate limit exceeded"}`` with no ``Retry-After``
header — i.e. it is a *burst* signal, not quota exhaustion.

Hermes can drive that one account from several places **at once**:

  * the main gateway agent process,
  * one or more TUI dashboards (e.g. the Hermes Desktop dashboard on its own port)
    and the ``tui_gateway.slash_worker`` processes they spawn,
  * sub-agent / delegation fan-out (``delegation.max_concurrent_children`` is 10).

Because those are *separate processes* (and, for sub-agents, separate threads) that
all share one Codex account, an in-process ``asyncio``/``threading`` semaphore cannot
bound the real concurrency the backend sees.  This module provides a gate backed by
``fcntl.flock`` on files under the Hermes **root** directory, so it serializes Codex
requests across every Hermes process *and* thread on the box.  A small JSON state
file carries a global cooldown that every caller honors after a 429, collapsing N
independent 1-second retry storms into a single coordinated backoff.

Design goals:
  * **Fail open.**  Any unexpected error degrades the gate to a no-op; it must never
    wedge or crash the request path.
  * **Crash safe.**  ``flock`` is released automatically by the kernel when the fd is
    closed or the process dies, so a killed worker can never strand a slot.
  * **Reentrant.**  A thread already holding the gate proceeds without re-locking, so
    an accidental nested Codex call can never self-deadlock.

All knobs are environment-tunable (read once at import; env is set at process launch):

  HERMES_CODEX_GATE_DISABLED                  set truthy to disable the gate entirely
  HERMES_CODEX_MAX_CONCURRENCY                max in-flight Codex requests box-wide (default 1)
  HERMES_CODEX_MIN_REQUEST_INTERVAL_SECONDS   min spacing between request *starts* (default 0)
  HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS   max wait for a slot before degrading (default 900)
  HERMES_CODEX_RATE_LIMIT_COOLDOWN_SECONDS    global pause after a 429 (default 8)
  HERMES_CODEX_RATE_LIMIT_COOLDOWN_MAX_SECONDS cap on the cooldown / Retry-After (default 60)
"""
from __future__ import annotations

import errno
import json
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import fcntl  # POSIX only; the Hermes agent host is Linux.

    _HAVE_FCNTL = True
except Exception:  # pragma: no cover - non-POSIX
    _HAVE_FCNTL = False


# ── env helpers ──────────────────────────────────────────────────────────────


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# ── config (read once) ───────────────────────────────────────────────────────

_DISABLED = _env_bool("HERMES_CODEX_GATE_DISABLED", False) or not _HAVE_FCNTL
_MAX_CONCURRENCY = max(1, _env_int("HERMES_CODEX_MAX_CONCURRENCY", 1))
_MIN_INTERVAL = max(0.0, _env_float("HERMES_CODEX_MIN_REQUEST_INTERVAL_SECONDS", 0.0))
_ACQUIRE_TIMEOUT = max(0.0, _env_float("HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS", 900.0))
_COOLDOWN_BASE = max(0.0, _env_float("HERMES_CODEX_RATE_LIMIT_COOLDOWN_SECONDS", 15.0))
_COOLDOWN_MAX = max(_COOLDOWN_BASE, _env_float("HERMES_CODEX_RATE_LIMIT_COOLDOWN_MAX_SECONDS", 90.0))
# Consecutive 429s within this window escalate the cooldown; a quiet gap resets it.
_COOLDOWN_RESET_WINDOW = max(1.0, _env_float("HERMES_CODEX_COOLDOWN_RESET_WINDOW_SECONDS", 120.0))
# Honor a backend Retry-After only up to this cap (the Codex backend often sends a
# misleading Retry-After: 1 while it keeps 429ing, so values <= the floor are ignored).
_RETRY_AFTER_HONOR_MAX = max(
    _COOLDOWN_MAX, _env_float("HERMES_CODEX_RETRY_AFTER_MAX_SECONDS", 300.0)
)

_STATE_NAME = "state.json"
_STATE_LOCK_NAME = "state.lock"

# Per-thread reentrancy depth so a nested Codex call on the same worker thread
# never blocks on a slot the thread already owns.
_local = threading.local()


def is_enabled() -> bool:
    return not _DISABLED


def _gate_dir() -> Optional[Path]:
    """Resolve the shared gate directory under the Hermes *root* (never per-profile).

    Keyed on the root so every profile / dashboard / slash_worker on this box shares
    one gate — they all draw on the same Codex account.  Returns ``None`` if no
    writable location can be resolved (gate then degrades to a no-op).
    """
    candidates = []
    try:  # local import to avoid an import cycle at module load
        from hermes_constants import get_default_hermes_root  # type: ignore

        candidates.append(Path(get_default_hermes_root()))
    except Exception:
        pass
    env_home = os.getenv("HERMES_HOME")
    if env_home:
        candidates.append(Path(env_home))
    candidates.append(Path(os.path.expanduser("~/.hermes")))
    for base in candidates:
        try:
            d = base / "codex_gate"
            d.mkdir(parents=True, exist_ok=True)
            return d
        except Exception:
            continue
    return None


# ── shared state (cooldown / pacing) ─────────────────────────────────────────


def _read_cooldown_until(gate_dir: Path) -> float:
    try:
        with open(gate_dir / _STATE_NAME, "r") as fh:
            data = json.load(fh)
        return float(data.get("cooldown_until", 0.0) or 0.0)
    except Exception:
        return 0.0


def _with_state(gate_dir: Path, mutator) -> Optional[dict]:
    """Open ``state.lock`` (exclusive flock), read ``state.json``, run
    ``mutator(data)`` (mutating ``data`` in place and returning truthy if it
    changed anything), and atomically write it back when changed.  Returns the
    (post-mutation) ``data`` dict, or ``None`` on any failure (gate fails open).
    """
    if not _HAVE_FCNTL:
        return None
    state_path = gate_dir / _STATE_NAME
    try:
        lock_fd = os.open(str(gate_dir / _STATE_LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
    except Exception:
        return None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data = {}
        try:
            with open(state_path, "r") as fh:
                data = json.load(fh) or {}
        except Exception:
            data = {}
        try:
            changed = mutator(data)
        except Exception:
            changed = False
        if changed:
            tmp = f"{state_path}.tmp.{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, state_path)
        return data
    except Exception:
        return None
    finally:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(lock_fd)
        except Exception:
            pass


def _compute_cooldown(retry_after: Optional[float], fail_count: int) -> float:
    """Cooldown seconds for the Nth consecutive 429.

    Honor a *meaningful* backend ``Retry-After`` (one larger than our floor); ignore
    a too-small value (the Codex backend keeps sending ``Retry-After: 1`` while it
    continues to 429).  Otherwise escalate exponentially from the floor up to the cap
    so a hammered account is given progressively more room to recover.
    """
    if retry_after and retry_after > _COOLDOWN_BASE:
        return min(float(retry_after), _RETRY_AFTER_HONOR_MAX)
    exponent = min(max(0, fail_count - 1), 6)
    return min(_COOLDOWN_BASE * (2 ** exponent), _COOLDOWN_MAX)


# ── the gate ─────────────────────────────────────────────────────────────────


class _CodexGate:
    """Context manager: acquire one Codex concurrency slot for the wrapped request."""

    def __init__(
        self,
        est_tokens: int = 0,
        interrupt_check: Optional[Callable[[], bool]] = None,
        touch: Optional[Callable[[str], None]] = None,
    ):
        self._slot_fd: Optional[int] = None
        self._reentrant = False
        self._counts_held = False  # did this instance bump _local.held?
        self._interrupt_check = interrupt_check
        self._touch = touch
        self._est_tokens = est_tokens

    # -- helpers --

    def _maybe_touch(self, msg: str) -> None:
        if self._touch is not None:
            try:
                self._touch(msg)
            except Exception:
                pass

    def _interrupted(self) -> bool:
        if self._interrupt_check is None:
            return False
        try:
            return bool(self._interrupt_check())
        except Exception:
            return False

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.time() + seconds
        while True:
            remaining = end - time.time()
            if remaining <= 0:
                return
            if self._interrupted():
                return
            time.sleep(min(0.1, remaining))

    def _try_acquire_slot(self, gate_dir: Path) -> bool:
        """Try to flock one of the N slot files (own fd per attempt)."""
        for i in range(_MAX_CONCURRENCY):
            slot_path = gate_dir / f"slot_{i}.lock"
            try:
                fd = os.open(str(slot_path), os.O_RDWR | os.O_CREAT, 0o600)
            except Exception:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                try:
                    os.close(fd)
                except Exception:
                    pass
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    continue  # slot busy → try the next one
                continue
            except Exception:
                try:
                    os.close(fd)
                except Exception:
                    pass
                continue
            self._slot_fd = fd
            return True
        return False

    def _enforce_min_interval(self, gate_dir: Path) -> None:
        if _MIN_INTERVAL <= 0:
            return
        try:
            with open(gate_dir / _STATE_NAME, "r") as fh:
                next_start = float((json.load(fh) or {}).get("next_start", 0.0) or 0.0)
        except Exception:
            next_start = 0.0
        wait = next_start - time.time()
        if wait > 0:
            self._sleep_interruptible(min(wait, _MIN_INTERVAL))

        def _set_next(data):
            data["next_start"] = time.time() + _MIN_INTERVAL
            return True

        _with_state(gate_dir, _set_next)

    # -- context manager protocol --

    def __enter__(self) -> "_CodexGate":
        if _DISABLED:
            return self
        # Reentrant: this thread already holds a slot → just count the depth.
        if getattr(_local, "held", 0) > 0:
            self._reentrant = True
            self._counts_held = True
            _local.held += 1
            return self
        gate_dir = _gate_dir()
        if gate_dir is None:
            return self  # fail open
        deadline = (time.time() + _ACQUIRE_TIMEOUT) if _ACQUIRE_TIMEOUT > 0 else None
        wait_ticks = 0
        while True:
            if self._interrupted():
                raise InterruptedError("Codex gate wait interrupted")
            # 1) Honor the global post-429 cooldown WITHOUT holding a slot.
            cooldown_until = _read_cooldown_until(gate_dir)
            now = time.time()
            if cooldown_until > now:
                self._maybe_touch(
                    f"codex rate-limit cooldown, {int(cooldown_until - now)}s remaining"
                )
                self._sleep_interruptible(min(cooldown_until - now, 1.0))
                continue
            # 2) Try to grab one of the N slots.
            if self._try_acquire_slot(gate_dir):
                self._enforce_min_interval(gate_dir)
                _local.held = getattr(_local, "held", 0) + 1
                self._counts_held = True
                return self
            # 3) All slots busy → bounded, jittered wait, then retry.
            if deadline is not None and time.time() > deadline:
                logger.warning(
                    "Codex gate acquire timed out after %.0fs (max_concurrency=%d); "
                    "proceeding without a slot to avoid blocking forever.",
                    _ACQUIRE_TIMEOUT,
                    _MAX_CONCURRENCY,
                )
                return self  # degrade: better an ungated request than a wedge
            wait_ticks += 1
            if wait_ticks % 60 == 0:
                self._maybe_touch("waiting for a codex concurrency slot")
            self._sleep_interruptible(0.05 + random.random() * 0.15)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._slot_fd is not None:
            try:
                fcntl.flock(self._slot_fd, fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                os.close(self._slot_fd)
            except Exception:
                pass
            self._slot_fd = None
        if self._counts_held:
            try:
                if getattr(_local, "held", 0) > 0:
                    _local.held -= 1
            except Exception:
                pass
            self._counts_held = False
        return False  # never suppress exceptions


def codex_request_gate(
    est_tokens: int = 0,
    interrupt_check: Optional[Callable[[], bool]] = None,
    touch: Optional[Callable[[str], None]] = None,
) -> _CodexGate:
    """Return a context manager that holds one box-wide Codex concurrency slot."""
    return _CodexGate(est_tokens=est_tokens, interrupt_check=interrupt_check, touch=touch)


# ── 429 cooldown signalling ──────────────────────────────────────────────────


def note_rate_limited(retry_after: Optional[float] = None) -> None:
    """Record a 429 and set an escalating global cooldown so every local Codex
    caller (other processes, sub-agent threads) backs off together instead of each
    independently hammering the backend on its own 1s retry timer.
    """
    if _DISABLED:
        return
    gate_dir = _gate_dir()
    if gate_dir is None:
        return

    def _mut(data):
        now = time.time()
        last = float(data.get("last_fail_ts", 0.0) or 0.0)
        count = int(data.get("fail_count", 0) or 0)
        if now - last > _COOLDOWN_RESET_WINDOW:
            count = 0  # quiet for a while → start the escalation over
        count += 1
        cooldown = _compute_cooldown(retry_after, count)
        data["fail_count"] = count
        data["last_fail_ts"] = now
        data["cooldown_until"] = max(float(data.get("cooldown_until", 0.0) or 0.0), now + cooldown)
        data["last_cooldown"] = cooldown
        return True

    data = _with_state(gate_dir, _mut)
    if data:
        logger.info(
            "Codex 429 #%s — global cooldown %.0fs; all local Codex callers will pause.",
            data.get("fail_count"),
            data.get("last_cooldown", _COOLDOWN_BASE),
        )


def note_success() -> None:
    """Record a successful Codex response, resetting the 429 escalation counter."""
    if _DISABLED:
        return
    gate_dir = _gate_dir()
    if gate_dir is None:
        return

    def _mut(data):
        if int(data.get("fail_count", 0) or 0) == 0:
            return False
        data["fail_count"] = 0
        return True

    _with_state(gate_dir, _mut)


def _extract_retry_after(err) -> Optional[float]:
    try:
        resp = getattr(err, "response", None)
        headers = getattr(resp, "headers", None)
        if headers is not None and hasattr(headers, "get"):
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                return float(raw)
    except Exception:
        pass
    return None


def note_rate_limited_from_error(err) -> bool:
    """If ``err`` is a Codex 429, set the global cooldown.  Returns True if it was."""
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(err, "status", None)
    if status != 429 and type(err).__name__ != "RateLimitError":
        return False
    note_rate_limited(_extract_retry_after(err))
    return True
