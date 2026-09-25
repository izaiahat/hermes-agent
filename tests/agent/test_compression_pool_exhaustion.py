"""Included-quota compression: model-scoped pool exhaustion preserves every turn."""
import json
from types import SimpleNamespace

import pytest

from agent.credential_pool import CredentialPool, PooledCredential


def test_exhausted_pool_never_reaches_paid_or_claude_and_checkpoints_once(tmp_path, monkeypatch):
    import agent.auxiliary_client as aux
    from agent.context_compressor import CompressionPoolExhausted, ContextCompressor
    from agent.conversation_compression import _preserve_exhausted_compression

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(CredentialPool, "_codex_quota_restored_upstream", lambda self, entry: False)
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {
        "provider": "openai-codex", "model": "gpt-6-sol", "fallback_chain": []
    })
    rows = [PooledCredential(
        provider="openai-codex", id=f"fixture-{i}", label=f"fixture-{i}",
        priority=i, source="manual:device_code", auth_type="oauth",
        access_token=f"fixture-access-{i}", refresh_token=f"fixture-refresh-{i}",
        last_status="exhausted" if i != 1 else "ok",
        last_error_code=429 if i != 1 else None,
        last_error_reset_at=4102444800 if i != 1 else None,
    ).to_dict() for i in range(3)]
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {"openai-codex": rows}
    }))
    calls = []
    monkeypatch.setattr(aux, "call_llm", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr("agent.context_compressor.call_llm", lambda **kwargs: calls.append(kwargs))
    eligible = aux._select_pool_entry("openai-codex", model="gpt-6-sol")[1]
    assert eligible is not None and eligible.id == "fixture-1"
    compressor = ContextCompressor(model="gpt-6-astra-900k", provider="openai-codex")
    # Simulate exhaustion via the registered pool cooldown API in the isolated
    # auth store, rather than sending requests or editing production credentials.
    pool = aux.load_pool("openai-codex")
    pool.mark_exhausted_and_rotate(status_code=400, credential_id="fixture-1",
                                   model="gpt-6-sol", failure_reason="model_entitlement")
    assert aux._select_pool_entry("openai-codex", model="gpt-6-sol") == (True, None)
    with pytest.raises(CompressionPoolExhausted):
        compressor._call_summary_llm("summary prompt", 0)
    assert compressor._on_summary_failure(CompressionPoolExhausted("pool exhausted"), [], None, "") is None
    assert compressor._last_summary_pool_exhausted is True
    assert compressor._abort_on_summary_failure({}, 2, None) is True
    assert compressor._last_compress_aborted is True
    assert aux._read_codex_access_token(model="gpt-6-sol") is None
    assert calls == []
    # A stale paid or Claude entry cannot be reached through the generic chain.
    assert aux._try_configured_fallback_chain("compression", "openai-codex") == (None, None, "")

    alerts = []
    monkeypatch.setattr("tools.discord_tool._get_bot_token", lambda: "fixture-bot-token")
    monkeypatch.setattr("tools.discord_tool._discord_request", lambda *args, **kwargs: alerts.append((args, kwargs)))

    # get_hermes_home is imported inside the function from the configuration module.
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: tmp_path)
    transcript = [{"role": "user", "content": "Continue the preserved work"},
                  {"role": "assistant", "content": "Acknowledged"}]
    agent = SimpleNamespace(session_id="fixture-session", context_compressor=compressor)
    checkpoint = _preserve_exhausted_compression(agent, transcript)
    assert checkpoint is not None
    assert checkpoint.stat().st_mode & 0o777 == 0o600
    data = json.loads(checkpoint.read_text())
    assert data["messages"] == transcript
    assert data["deterministic_summary"]
    assert data["reason"] == "codex_compression_pool_exhausted"
    assert _preserve_exhausted_compression(agent, transcript) == checkpoint
    assert len(alerts) == 1
    assert alerts[0][0][1] == "/channels/1487300132054765629/messages"
    assert "fixture-access" not in str(alerts)
