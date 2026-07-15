"""Cross-process concurrency + rate gate for OpenAI Codex requests.

The ChatGPT / Codex *subscription* backend (``https://chatgpt.com/backend-api/codex``)
enforces a tight **per-account** burst / concurrency limit.  When it is exceeded it
returns ``HTTP 429 {"detail": "Rate limit exceeded"}`` with no ``Retry-After``
header — i.e. it is a *burst* signal, not quota exhaustion.

Hermes can drive that one account from several places **at once**:

  * the main gateway agent process,
  * one or more TUI dashboards (e.g. the Hermes Desktop dashboard on its own port)
    and the ``tui_gateway.slash_worker`` processes they spawn,
  * sub-agent / delegation fan-out (hard-capped at five active descendants per
    process/tree).

Because those are *separate processes* (and, for sub-agents, separate threads) that
all share one Codex account, an in-process ``asyncio``/``threading`` semaphore cannot
bound the real concurrency the backend sees.  This module provides a gate backed by
``fcntl.flock`` on files under the Hermes **root** directory, so it bounds the number
of *simultaneous* Codex requests across every Hermes process *and* thread on the box
to an adaptive ceiling.  A small JSON state file carries (a) a global cooldown that
every caller honors after a 429/503, collapsing N independent retry storms into a
single coordinated backoff, and (b) the current AIMD concurrency permit (below).

Adaptive concurrency (AIMD, à la TCP congestion control): the gate starts at a
configured concurrency and, while the account stays healthy, *additively* probes one
slot higher every ``PROBE_SECONDS`` up to ``MAX_CONCURRENCY``.  On any 429 (per-account
burst limit) or 503/529 (backend overloaded) it *multiplicatively* halves the permit
down to ``MIN_CONCURRENCY`` and arms the cooldown — so the box self-tunes to whatever
the account currently tolerates instead of being pinned to one hand-picked number.
Set ``MAX_CONCURRENCY == MIN_CONCURRENCY`` to pin it (the old static-semaphore behavior).

Design goals:
  * **Fail closed on admission.** If the gate is enabled but cannot resolve its
    shared state directory or cannot acquire a slot before the bounded timeout,
    the request raises a retryable local error instead of bypassing the box-wide
    ceiling. Explicit ``HERMES_CODEX_GATE_DISABLED=1`` remains the operator-owned
    emergency bypass.
  * **Crash safe.**  ``flock`` is released automatically by the kernel when the fd is
    closed or the process dies, so a killed worker can never strand a slot.
  * **Reentrant.**  A thread already holding the gate proceeds without re-locking, so
    an accidental nested Codex call can never self-deadlock.

All knobs are environment-tunable through one authoritative root policy. Before
reading them, Hermes loads ``<Hermes root>/codex-throttle.env`` (or the explicit
test/recovery override ``HERMES_CODEX_THROTTLE_ENV_FILE``) and replaces inherited
``HERMES_CODEX_*`` values with the file's values. This keeps gateway, dashboard,
TUI, slash-worker, auxiliary, app-server, compaction, and delegated launch paths
on one policy even when a shell wrapper did not source the file:

  HERMES_CODEX_GATE_DISABLED                   set truthy to disable the gate entirely
  HERMES_CODEX_MAX_CONCURRENCY                 box-wide request ceiling (default and hard cap 5)
  HERMES_CODEX_MIN_CONCURRENCY                 floor the adaptive permit never drops below (default 1)
  HERMES_CODEX_CONCURRENCY_START               permit value on fresh state (default = MAX)
  HERMES_CODEX_ADAPTIVE_CONCURRENCY            enable the AIMD permit (default on; moot when MAX==MIN)
  HERMES_CODEX_CONCURRENCY_PROBE_SECONDS       healthy seconds between additive +1 probes (default 30)
  HERMES_CODEX_CONCURRENCY_BACKOFF_FACTOR      permit multiplier on a 429/503 (default 0.5)
  HERMES_CODEX_MIN_REQUEST_INTERVAL_SECONDS    min spacing between request *starts* (default 0)
  HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS    monotonic fail-closed wait including cooldown (default 900; 0 rejects)
  HERMES_CODEX_RATE_LIMIT_COOLDOWN_SECONDS     global pause after a 429 (default 15)
  HERMES_CODEX_OVERLOAD_COOLDOWN_SECONDS       gentler global pause after a 503/529 (default 5)
  HERMES_CODEX_RATE_LIMIT_COOLDOWN_MAX_SECONDS cap on the escalating cooldown (default 90)
"""

from __future__ import annotations

import errno
import json
import logging
import math
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


# ── shared throttle-file loading ─────────────────────────────────────────────


def _throttle_env_candidates() -> list[Path]:
    """Return the one canonical root-scoped throttle policy path.

    ``HERMES_CODEX_THROTTLE_ENV_FILE`` exists for isolated tests and recovery.
    Normal production processes resolve the policy from the Hermes root only.
    """
    explicit = str(os.getenv("HERMES_CODEX_THROTTLE_ENV_FILE") or "").strip()
    if explicit:
        return [Path(os.path.expanduser(explicit))]
    try:
        from hermes_constants import get_default_hermes_root  # type: ignore

        return [Path(get_default_hermes_root()) / "codex-throttle.env"]
    except Exception:
        return []


def _load_throttle_env() -> Optional[Path]:
    """Load authoritative ``HERMES_CODEX_*`` values from the shared policy.

    Parsing is intentionally conservative: ``KEY=VALUE`` and ``export KEY=VALUE``
    are accepted, optional matching quotes are removed, and no shell expansion or
    command execution occurs. When the canonical file exists its values override
    ordinary inherited environment values so every launch surface uses one policy.
    """
    for path in _throttle_env_candidates():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if (
                not key.startswith("HERMES_CODEX_")
                or not key.replace("_", "").isalnum()
            ):
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ[key] = value
        return path
    return None


_LOADED_THROTTLE_ENV_FILE = _load_throttle_env()


# ── config (read once) ───────────────────────────────────────────────────────

# Only the explicit operator switch disables admission. Missing POSIX locking is
# an unavailable-gate error, never an implicit bypass.
_DISABLED = _env_bool("HERMES_CODEX_GATE_DISABLED", False)
_MAX_CONCURRENCY = min(5, max(1, _env_int("HERMES_CODEX_MAX_CONCURRENCY", 5)))
# AIMD adaptive concurrency: the gate keeps a shared "permit" in [MIN, MAX] that grows
# additively while healthy and shrinks multiplicatively on a 429/503.  When MAX == MIN
# (e.g. both 1) the permit is fixed and the gate behaves like the old static semaphore.
_MIN_CONCURRENCY = min(
    _MAX_CONCURRENCY, max(1, _env_int("HERMES_CODEX_MIN_CONCURRENCY", 1))
)
_CONCURRENCY_START = min(
    _MAX_CONCURRENCY,
    max(_MIN_CONCURRENCY, _env_int("HERMES_CODEX_CONCURRENCY_START", _MAX_CONCURRENCY)),
)
_ADAPTIVE = _env_bool("HERMES_CODEX_ADAPTIVE_CONCURRENCY", True) and (
    _MAX_CONCURRENCY > _MIN_CONCURRENCY
)
# Additive increase: probe one extra slot back after this many healthy seconds.
_PROBE_INTERVAL = max(1.0, _env_float("HERMES_CODEX_CONCURRENCY_PROBE_SECONDS", 30.0))
# Multiplicative decrease: factor applied to the permit on a 429/503 (0 < f < 1; the
# drop is always at least one slot, down to the floor).
_BACKOFF_FACTOR = min(
    0.99, max(0.05, _env_float("HERMES_CODEX_CONCURRENCY_BACKOFF_FACTOR", 0.5))
)
_MIN_INTERVAL = max(0.0, _env_float("HERMES_CODEX_MIN_REQUEST_INTERVAL_SECONDS", 0.0))
_ACQUIRE_TIMEOUT = max(
    0.0, _env_float("HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS", 900.0)
)
_COOLDOWN_BASE = max(0.0, _env_float("HERMES_CODEX_RATE_LIMIT_COOLDOWN_SECONDS", 15.0))
# A 503/529 is a server-side blip (usually clears in a second or two), so it gets a
# gentler base cooldown than a 429 (a per-account burst limit that needs real room).
_OVERLOAD_COOLDOWN_BASE = max(
    0.0, _env_float("HERMES_CODEX_OVERLOAD_COOLDOWN_SECONDS", 5.0)
)
_COOLDOWN_MAX = max(
    _COOLDOWN_BASE, _env_float("HERMES_CODEX_RATE_LIMIT_COOLDOWN_MAX_SECONDS", 90.0)
)
# Consecutive 429s within this window escalate the cooldown; a quiet gap resets it.
_COOLDOWN_RESET_WINDOW = max(
    1.0, _env_float("HERMES_CODEX_COOLDOWN_RESET_WINDOW_SECONDS", 120.0)
)
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


class CodexGateAdmissionError(TimeoutError):
    """Retryable local rejection raised before a Codex request starts."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def is_enabled() -> bool:
    return not _DISABLED


def runtime_config() -> dict:
    """Return a secret-free snapshot of the effective gate policy."""
    gate_dir = _gate_dir() if not _DISABLED else None
    return {
        "enabled": not _DISABLED,
        "fcntl_available": _HAVE_FCNTL,
        "policy_file_authoritative": _LOADED_THROTTLE_ENV_FILE is not None,
        "loaded_env_file": (
            str(_LOADED_THROTTLE_ENV_FILE) if _LOADED_THROTTLE_ENV_FILE else None
        ),
        "gate_dir": str(gate_dir) if gate_dir else None,
        "max_concurrency": _MAX_CONCURRENCY,
        "min_concurrency": _MIN_CONCURRENCY,
        "concurrency_start": _CONCURRENCY_START,
        "adaptive": _ADAPTIVE,
        "acquire_timeout_seconds": _ACQUIRE_TIMEOUT,
        "current_permit": _read_permit(gate_dir) if gate_dir else None,
    }


def _gate_dir() -> Optional[Path]:
    """Resolve the one canonical gate directory under the Hermes root."""
    try:  # local import to avoid an import cycle at module load
        from hermes_constants import get_default_hermes_root  # type: ignore

        gate_dir = Path(get_default_hermes_root()) / "codex_gate"
        gate_dir.mkdir(parents=True, exist_ok=True)
        if not gate_dir.is_dir() or not os.access(gate_dir, os.W_OK | os.X_OK):
            return None
        return gate_dir
    except Exception:
        return None


# ── shared state (cooldown / pacing) ─────────────────────────────────────────


def _read_cooldown_until(gate_dir: Path) -> float:
    try:
        with open(gate_dir / _STATE_NAME, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return float(data.get("cooldown_until", 0.0) or 0.0)
    except Exception:
        return 0.0


def recommended_retry_delay() -> float:
    """Seconds until the shared Codex cooldown clears, for visible retry timers.

    The gate itself already honors the cooldown before acquiring a request slot; this
    helper lets the conversation retry loop sleep up front so the user-visible retry
    delay matches the real box-wide recovery window.  It is intentionally best-effort
    and fails open to 0.0 if the gate is disabled or state cannot be read.
    """
    if _DISABLED:
        return 0.0
    gate_dir = _gate_dir()
    if gate_dir is None:
        return 0.0
    return max(0.0, _read_cooldown_until(gate_dir) - time.time())


def _read_permit(gate_dir: Path) -> int:
    """Current AIMD concurrency permit (best-effort, no lock — read on the hot path).

    Falls back to the optimistic start value if state is missing/unreadable so a
    transient read glitch never *over*-throttles; the separately-honored cooldown
    still protects the backend in that window.
    """
    if not _ADAPTIVE:
        return _MAX_CONCURRENCY
    try:
        with open(gate_dir / _STATE_NAME, "r", encoding="utf-8") as fh:
            permit = int((json.load(fh) or {}).get("permit", _CONCURRENCY_START))
    except Exception:
        permit = _CONCURRENCY_START
    return max(_MIN_CONCURRENCY, min(_MAX_CONCURRENCY, permit))


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
        lock_fd = os.open(
            str(gate_dir / _STATE_LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600
        )
    except Exception:
        return None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data = {}
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception:
            data = {}
        try:
            changed = mutator(data)
        except Exception:
            changed = False
        if changed:
            tmp = f"{state_path}.tmp.{os.getpid()}"
            with open(tmp, "w", encoding="utf-8") as fh:
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


def _compute_cooldown(
    retry_after: Optional[float], fail_count: int, base: Optional[float] = None
) -> float:
    """Cooldown seconds for the Nth consecutive pushback.

    ``base`` is the floor for this error class (429 vs the gentler 503/529 base);
    defaults to the 429 base.  Honor a *meaningful* backend ``Retry-After`` (one
    larger than that floor); ignore a too-small value (the Codex backend keeps
    sending ``Retry-After: 1`` while it continues to 429).  Otherwise escalate
    exponentially from the floor up to the cap so a hammered account is given
    progressively more room to recover.
    """
    base = _COOLDOWN_BASE if base is None else base
    if retry_after and retry_after > base:
        return min(float(retry_after), _RETRY_AFTER_HONOR_MAX)
    exponent = min(max(0, fail_count - 1), 6)
    return min(base * (2**exponent), _COOLDOWN_MAX)


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

    def _raise_if_interrupted(self) -> None:
        if self._interrupted():
            raise CodexGateAdmissionError(
                "interrupted", "Codex gate wait interrupted before request admission"
            )

    def _raise_if_timed_out(self, deadline: float) -> None:
        if time.monotonic() >= deadline:
            message = (
                "Codex gate acquire timed out after "
                f"{_ACQUIRE_TIMEOUT:.0f}s (max_concurrency={_MAX_CONCURRENCY}); "
                "request blocked to preserve the global concurrency ceiling"
            )
            logger.error(message)
            raise CodexGateAdmissionError("timeout", message)

    def _sleep_interruptible(self, seconds: float, deadline: float) -> None:
        end = min(time.monotonic() + max(0.0, seconds), deadline)
        while True:
            self._raise_if_interrupted()
            self._raise_if_timed_out(deadline)
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def _try_acquire_slot(self, gate_dir: Path, limit: int) -> bool:
        """Try to flock one of the first ``limit`` slot files (own fd per attempt).

        ``limit`` is the current AIMD permit (1..MAX); only the low-indexed slots
        below it are eligible, so when the permit shrinks the high-index slots drain
        naturally as their already-in-flight requests finish.
        """
        for i in range(max(1, min(limit, _MAX_CONCURRENCY))):
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

    def _enforce_min_interval(self, gate_dir: Path, deadline: float) -> None:
        if _MIN_INTERVAL <= 0:
            return
        try:
            with open(gate_dir / _STATE_NAME, "r", encoding="utf-8") as fh:
                next_start = float((json.load(fh) or {}).get("next_start", 0.0) or 0.0)
        except Exception:
            next_start = 0.0
        wait = next_start - time.time()
        if wait > 0:
            self._sleep_interruptible(min(wait, _MIN_INTERVAL), deadline)

        def _set_next(data):
            data["next_start"] = time.time() + _MIN_INTERVAL
            return True

        _with_state(gate_dir, _set_next)

    # -- context manager protocol --

    def __enter__(self) -> "_CodexGate":
        if _DISABLED:
            return self
        if not _HAVE_FCNTL:
            raise CodexGateAdmissionError(
                "unavailable",
                "Codex concurrency gate requires POSIX fcntl locking; request blocked",
            )
        # Reentrant wrappers around the same synchronous request share one slot.
        if getattr(_local, "held", 0) > 0:
            self._reentrant = True
            self._counts_held = True
            _local.held += 1
            return self
        gate_dir = _gate_dir()
        if gate_dir is None:
            raise CodexGateAdmissionError(
                "unavailable",
                "Codex concurrency gate canonical directory is unavailable; "
                "request blocked to preserve the global ceiling",
            )
        deadline = time.monotonic() + _ACQUIRE_TIMEOUT
        wait_ticks = 0
        while True:
            self._raise_if_interrupted()
            self._raise_if_timed_out(deadline)
            # 1) Honor the global post-pushback cooldown WITHOUT holding a slot.
            cooldown_until = _read_cooldown_until(gate_dir)
            now_wall = time.time()
            if cooldown_until > now_wall:
                self._maybe_touch(
                    f"codex rate-limit cooldown, {int(cooldown_until - now_wall)}s remaining"
                )
                self._sleep_interruptible(min(cooldown_until - now_wall, 1.0), deadline)
                continue
            # 2) Try to grab one of the slots eligible under the current AIMD permit.
            permit = _read_permit(gate_dir)
            if self._try_acquire_slot(gate_dir, permit):
                try:
                    self._enforce_min_interval(gate_dir, deadline)
                except BaseException:
                    self.__exit__(None, None, None)
                    raise
                _local.held = getattr(_local, "held", 0) + 1
                self._counts_held = True
                return self
            # 3) All slots busy → bounded, jittered wait, then retry.
            wait_ticks += 1
            if wait_ticks % 60 == 0:
                self._maybe_touch("waiting for a codex concurrency slot")
            self._sleep_interruptible(0.05 + random.random() * 0.15, deadline)

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
    return _CodexGate(
        est_tokens=est_tokens, interrupt_check=interrupt_check, touch=touch
    )


# ── 429 cooldown signalling ──────────────────────────────────────────────────


def note_rate_limited(
    retry_after: Optional[float] = None, status: Optional[int] = None
) -> None:
    """Record a 429/503 and react on two axes: (a) set an escalating global cooldown so
    every local Codex caller (other processes, sub-agent threads) backs off together
    instead of each independently hammering the backend on its own retry timer, and
    (b) *multiplicatively* shrink the AIMD concurrency permit so the box eases off the
    gas rather than merely waiting and then re-bursting at the same width.

    ``status`` selects the cooldown floor: a 503/529 (server blip) gets the gentler
    overload base; everything else gets the 429 burst base.
    """
    if _DISABLED:
        return
    gate_dir = _gate_dir()
    if gate_dir is None:
        return
    base = _OVERLOAD_COOLDOWN_BASE if status in (503, 529) else _COOLDOWN_BASE

    def _mut(data):
        now = time.time()
        last = float(data.get("last_fail_ts", 0.0) or 0.0)
        count = int(data.get("fail_count", 0) or 0)
        if now - last > _COOLDOWN_RESET_WINDOW:
            count = 0  # quiet for a while → start the escalation over
        count += 1
        cooldown = _compute_cooldown(retry_after, count, base)
        data["fail_count"] = count
        data["last_fail_ts"] = now
        data["cooldown_until"] = max(
            float(data.get("cooldown_until", 0.0) or 0.0), now + cooldown
        )
        data["last_cooldown"] = cooldown
        # AIMD multiplicative decrease: ease off concurrency, guaranteeing at least a
        # one-slot drop, down to the floor.  Reset the probe clock so we don't grow the
        # permit back until we've been healthy for a full PROBE_INTERVAL again.
        if _ADAPTIVE:
            cur = max(
                _MIN_CONCURRENCY,
                min(_MAX_CONCURRENCY, int(data.get("permit", _CONCURRENCY_START))),
            )
            # Halve (rounding up), but always drop by at least one slot, never below the
            # floor.  Rounding up keeps a single transient blip from collapsing a small
            # ceiling straight to 1 (3→2, not 3→1); repeated pushback still walks it down
            # a slot at a time, and the escalating cooldown handles sustained trouble.
            shrunk = max(
                _MIN_CONCURRENCY, min(cur - 1, math.ceil(cur * _BACKOFF_FACTOR))
            )
            if shrunk < cur:
                data["permit"] = shrunk
                data["last_increase_ts"] = now
        return True

    data = _with_state(gate_dir, _mut)
    if data:
        logger.info(
            "Codex %s #%s — global cooldown %.0fs, concurrency permit now %s; "
            "all local Codex callers will back off together.",
            status or 429,
            data.get("fail_count"),
            data.get("last_cooldown", base),
            data.get("permit", _MAX_CONCURRENCY),
        )


def note_success() -> None:
    """Record a successful Codex response: reset the 429 escalation counter and, for
    AIMD, *additively* probe the concurrency permit one slot higher once we've been
    healthy (out of cooldown) for a full PROBE_INTERVAL.  Writes state only when
    something actually changes, so the per-response success path stays cheap under load.
    """
    if _DISABLED:
        return
    gate_dir = _gate_dir()
    if gate_dir is None:
        return

    def _mut(data):
        changed = False
        if int(data.get("fail_count", 0) or 0) != 0:
            data["fail_count"] = 0
            changed = True
        if _ADAPTIVE:
            now = time.time()
            cur = max(
                _MIN_CONCURRENCY,
                min(_MAX_CONCURRENCY, int(data.get("permit", _CONCURRENCY_START))),
            )
            cooldown_until = float(data.get("cooldown_until", 0.0) or 0.0)
            last_inc = float(data.get("last_increase_ts", 0.0) or 0.0)
            if (
                cur < _MAX_CONCURRENCY
                and now >= cooldown_until
                and (now - last_inc) >= _PROBE_INTERVAL
            ):
                data["permit"] = cur + 1
                data["last_increase_ts"] = now
                changed = True
        return changed

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


# HTTP statuses that mean "ease off the load": 429 = per-account burst/concurrency
# limit; 503/529 = backend overloaded.  Both arm the cooldown and shrink concurrency.
_BACKOFF_STATUSES = frozenset({429, 503, 529})


def _classify_backoff(err):
    """Decide whether ``err`` is a load-pushback error the gate should react to.

    Returns ``(status, retry_after)`` for a 429/503-class error, else ``(None, None)``.
    Prefers the agent's central ``error_classifier`` taxonomy — so message-only rate
    limits, transient 402s, 529s, and OpenRouter-wrapped overloads are all caught the
    same way the main retry loop sees them — and falls back to a self-contained
    status-code check if that module can't be imported (gate stays standalone).
    """
    retry_after = _extract_retry_after(err)
    try:
        from agent.error_classifier import classify_api_error, FailoverReason

        c = classify_api_error(err)
        if c.reason in (FailoverReason.rate_limit, FailoverReason.overloaded):
            # Normalize to a representative status so note_rate_limited can pick the
            # right cooldown floor (429 burst vs gentler 503 overload).
            status = c.status_code
            if status not in _BACKOFF_STATUSES:
                status = 503 if c.reason is FailoverReason.overloaded else 429
            return status, retry_after
        return None, None
    except Exception:
        pass
    # Fallback: classifier unavailable — use the raw status code / SDK type name.
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(err, "status", None)
    if status in _BACKOFF_STATUSES:
        return status, retry_after
    if type(err).__name__ == "RateLimitError":
        return 429, retry_after
    return None, None


def note_rate_limited_from_error(err) -> bool:
    """If ``err`` is a Codex rate-limit (429) or overload (503/529) error, arm the
    coordinated cooldown and shrink concurrency.  Returns True if it was such an error.
    """
    status, retry_after = _classify_backoff(err)
    if status is None:
        return False
    note_rate_limited(retry_after, status=status)
    return True
