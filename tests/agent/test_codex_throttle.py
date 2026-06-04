from __future__ import annotations

import importlib
from types import SimpleNamespace


def _reload_throttle(monkeypatch, tmp_path, *, base="60", cap="300"):
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


def test_codex_retry_delay_uses_shared_cooldown_floor_not_retry_after_one(tmp_path, monkeypatch):
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
        SimpleNamespace(provider="custom", base_url="https://chatgpt.com/backend-api/codex/")
    )
    assert not _is_openai_codex_backend_agent(
        SimpleNamespace(provider="openai", base_url="https://api.openai.com/v1")
    )
