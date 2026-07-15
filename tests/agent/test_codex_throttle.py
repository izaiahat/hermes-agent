from __future__ import annotations

import importlib
import multiprocessing
import queue
import threading
import time
from types import SimpleNamespace

import pytest


def _gate_process_worker(index, release_event, result_queue):
    from agent.codex_throttle import CodexGateAdmissionError, codex_request_gate

    try:
        with codex_request_gate():
            result_queue.put(("acquired", index, None))
            release_event.wait(timeout=5)
    except CodexGateAdmissionError as exc:
        result_queue.put(("error", index, exc.reason))


def _reload_throttle(monkeypatch, tmp_path, *, base="60", cap="300"):
    # Keep unit tests isolated from the operator's live root throttle policy.
    monkeypatch.setenv(
        "HERMES_CODEX_THROTTLE_ENV_FILE", str(tmp_path / "missing-throttle.env")
    )
    monkeypatch.setenv("HERMES_CODEX_RATE_LIMIT_COOLDOWN_SECONDS", base)
    monkeypatch.setenv("HERMES_CODEX_RATE_LIMIT_COOLDOWN_MAX_SECONDS", cap)
    from agent import codex_throttle

    mod = importlib.reload(codex_throttle)
    gate_dir = tmp_path / "codex_gate"
    gate_dir.mkdir()
    monkeypatch.setattr(mod, "_gate_dir", lambda: gate_dir)
    return mod


class _Response:
    headers = {"retry-after": "1"}


class _RateLimitError(Exception):
    status_code = 429
    response = _Response()


def test_codex_retry_delay_uses_shared_cooldown_floor_not_retry_after_one(
    tmp_path, monkeypatch
):
    throttle = _reload_throttle(monkeypatch, tmp_path, base="60", cap="300")

    assert throttle.note_rate_limited_from_error(_RateLimitError()) is True

    delay = throttle.recommended_retry_delay()
    assert 55 <= delay <= 60


def test_codex_cooldown_escalates_and_is_capped(tmp_path, monkeypatch):
    throttle = _reload_throttle(monkeypatch, tmp_path, base="60", cap="300")

    for _ in range(4):
        throttle.note_rate_limited()

    delay = throttle.recommended_retry_delay()
    assert 250 <= delay <= 300


def test_conversation_loop_detects_chatgpt_codex_backend():
    from agent.conversation_loop import _is_openai_codex_backend_agent

    assert _is_openai_codex_backend_agent(
        SimpleNamespace(provider="openai-codex", base_url="https://example.com")
    )
    assert _is_openai_codex_backend_agent(
        SimpleNamespace(
            provider="custom", base_url="https://chatgpt.com/backend-api/codex/"
        )
    )
    assert not _is_openai_codex_backend_agent(
        SimpleNamespace(provider="openai", base_url="https://api.openai.com/v1")
    )


def _reload_with_policy(monkeypatch, tmp_path, policy: str, **overrides):
    keys = [
        "HERMES_CODEX_MAX_CONCURRENCY",
        "HERMES_CODEX_MIN_CONCURRENCY",
        "HERMES_CODEX_CONCURRENCY_START",
        "HERMES_CODEX_ADAPTIVE_CONCURRENCY",
        "HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    policy_path = tmp_path / "codex-throttle.env"
    policy_path.write_text(policy, encoding="utf-8")
    monkeypatch.setenv("HERMES_CODEX_THROTTLE_ENV_FILE", str(policy_path))
    for key, value in overrides.items():
        monkeypatch.setenv(key, str(value))
    from agent import codex_throttle

    return importlib.reload(codex_throttle), policy_path


def test_root_throttle_policy_loads_for_unwrapped_process(monkeypatch, tmp_path):
    throttle, policy_path = _reload_with_policy(
        monkeypatch,
        tmp_path,
        "\n".join([
            "HERMES_CODEX_MAX_CONCURRENCY=5",
            "HERMES_CODEX_MIN_CONCURRENCY=1",
            "HERMES_CODEX_CONCURRENCY_START=5",
            "HERMES_CODEX_ADAPTIVE_CONCURRENCY=1",
            "HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS=300",
        ]),
    )
    snapshot = throttle.runtime_config()
    assert snapshot["loaded_env_file"] == str(policy_path)
    assert snapshot["max_concurrency"] == 5
    assert snapshot["concurrency_start"] == 5
    assert snapshot["acquire_timeout_seconds"] == 300


def test_root_policy_is_authoritative_over_inherited_environment(monkeypatch, tmp_path):
    throttle, _ = _reload_with_policy(
        monkeypatch,
        tmp_path,
        "HERMES_CODEX_MAX_CONCURRENCY=5\n",
        HERMES_CODEX_MAX_CONCURRENCY="2",
    )
    assert throttle.runtime_config()["max_concurrency"] == 5
    assert throttle.runtime_config()["policy_file_authoritative"] is True


def test_enabled_gate_fails_closed_when_shared_directory_is_unavailable(
    monkeypatch, tmp_path
):
    throttle = _reload_throttle(monkeypatch, tmp_path)
    monkeypatch.setattr(throttle, "_DISABLED", False)
    monkeypatch.setattr(throttle, "_gate_dir", lambda: None)
    with pytest.raises(
        throttle.CodexGateAdmissionError, match="request blocked"
    ) as exc:
        throttle.codex_request_gate().__enter__()
    assert exc.value.reason == "unavailable"


def test_gate_acquire_timeout_fails_closed_instead_of_proceeding_ungated(
    monkeypatch, tmp_path
):
    throttle = _reload_throttle(monkeypatch, tmp_path)
    gate_dir = tmp_path / "timeout-gate"
    gate_dir.mkdir()
    monkeypatch.setattr(throttle, "_DISABLED", False)
    monkeypatch.setattr(throttle, "_ACQUIRE_TIMEOUT", 0.02)
    monkeypatch.setattr(throttle, "_gate_dir", lambda: gate_dir)
    monkeypatch.setattr(throttle, "_read_cooldown_until", lambda _: 0.0)
    monkeypatch.setattr(throttle, "_read_permit", lambda _: 1)
    monkeypatch.setattr(
        throttle._CodexGate, "_try_acquire_slot", lambda self, gate_dir, permit: False
    )
    monkeypatch.setattr(
        throttle._CodexGate,
        "_sleep_interruptible",
        lambda self, seconds, deadline: time.sleep(0.001),
    )
    with pytest.raises(
        throttle.CodexGateAdmissionError, match="request blocked"
    ) as exc:
        throttle.codex_request_gate().__enter__()
    assert exc.value.reason == "timeout"


def test_box_wide_gate_admits_five_and_blocks_the_sixth(monkeypatch, tmp_path):
    throttle, _ = _reload_with_policy(
        monkeypatch,
        tmp_path,
        "\n".join([
            "HERMES_CODEX_MAX_CONCURRENCY=5",
            "HERMES_CODEX_MIN_CONCURRENCY=5",
            "HERMES_CODEX_CONCURRENCY_START=5",
            "HERMES_CODEX_ADAPTIVE_CONCURRENCY=0",
            "HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS=1",
        ]),
    )
    gate_dir = tmp_path / "five-slot-gate"
    gate_dir.mkdir()
    monkeypatch.setattr(throttle, "_gate_dir", lambda: gate_dir)

    release = threading.Event()
    all_acquired = threading.Event()
    acquired = []
    lock = threading.Lock()

    def holder():
        with throttle.codex_request_gate():
            with lock:
                acquired.append(1)
                if len(acquired) == 5:
                    all_acquired.set()
            release.wait(timeout=5)

    threads = [threading.Thread(target=holder) for _ in range(5)]
    for thread in threads:
        thread.start()
    assert all_acquired.wait(timeout=3)

    monkeypatch.setattr(throttle, "_ACQUIRE_TIMEOUT", 0.05)
    with pytest.raises(TimeoutError, match="request blocked"):
        throttle.codex_request_gate().__enter__()

    release.set()
    for thread in threads:
        thread.join(timeout=3)
    assert all(not thread.is_alive() for thread in threads)


def test_missing_fcntl_fails_closed(monkeypatch, tmp_path):
    throttle = _reload_throttle(monkeypatch, tmp_path)
    monkeypatch.setattr(throttle, "_DISABLED", False)
    monkeypatch.setattr(throttle, "_HAVE_FCNTL", False)
    with pytest.raises(throttle.CodexGateAdmissionError) as exc:
        throttle.codex_request_gate().__enter__()
    assert exc.value.reason == "unavailable"


def test_acquire_timeout_includes_global_cooldown(monkeypatch, tmp_path):
    throttle = _reload_throttle(monkeypatch, tmp_path)
    gate_dir = tmp_path / "cooldown-timeout"
    gate_dir.mkdir()
    monkeypatch.setattr(throttle, "_DISABLED", False)
    monkeypatch.setattr(throttle, "_ACQUIRE_TIMEOUT", 0.03)
    monkeypatch.setattr(throttle, "_gate_dir", lambda: gate_dir)
    monkeypatch.setattr(throttle, "_read_cooldown_until", lambda _: time.time() + 30)
    started = time.monotonic()
    with pytest.raises(throttle.CodexGateAdmissionError) as exc:
        throttle.codex_request_gate().__enter__()
    assert exc.value.reason == "timeout"
    assert time.monotonic() - started < 0.5


def test_zero_timeout_rejects_immediately(monkeypatch, tmp_path):
    throttle = _reload_throttle(monkeypatch, tmp_path)
    gate_dir = tmp_path / "zero-timeout"
    gate_dir.mkdir()
    monkeypatch.setattr(throttle, "_DISABLED", False)
    monkeypatch.setattr(throttle, "_ACQUIRE_TIMEOUT", 0.0)
    monkeypatch.setattr(throttle, "_gate_dir", lambda: gate_dir)
    started = time.monotonic()
    with pytest.raises(throttle.CodexGateAdmissionError) as exc:
        throttle.codex_request_gate().__enter__()
    assert exc.value.reason == "timeout"
    assert time.monotonic() - started < 0.1


def test_policy_cannot_raise_hard_ceiling_above_five(monkeypatch, tmp_path):
    throttle, _ = _reload_with_policy(
        monkeypatch,
        tmp_path,
        "HERMES_CODEX_MAX_CONCURRENCY=99\n",
    )
    assert throttle.runtime_config()["max_concurrency"] == 5


def test_box_wide_gate_is_shared_across_processes(monkeypatch, tmp_path):
    policy = tmp_path / "multiprocess-policy.env"
    policy.write_text(
        "\n".join([
            "HERMES_CODEX_MAX_CONCURRENCY=5",
            "HERMES_CODEX_MIN_CONCURRENCY=5",
            "HERMES_CODEX_CONCURRENCY_START=5",
            "HERMES_CODEX_ADAPTIVE_CONCURRENCY=0",
            "HERMES_CODEX_GATE_ACQUIRE_TIMEOUT_SECONDS=0.5",
        ]),
        encoding="utf-8",
    )
    root = tmp_path / "isolated-hermes-root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_CODEX_THROTTLE_ENV_FILE", str(policy))

    ctx = multiprocessing.get_context("spawn")
    release = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(target=_gate_process_worker, args=(i, release, result_queue))
        for i in range(6)
    ]
    for process in processes:
        process.start()

    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=8))
        except queue.Empty:
            break
    release.set()
    for process in processes:
        process.join(timeout=8)

    assert len(results) == 6
    assert sum(item[0] == "acquired" for item in results) == 5
    errors = [item for item in results if item[0] == "error"]
    assert errors == [("error", errors[0][1], "timeout")]
    assert all(process.exitcode == 0 for process in processes)
