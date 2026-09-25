"""Compression must select Codex credentials for its model, not an unscoped pool."""

import json
import time
from types import SimpleNamespace

from agent.credential_pool import PooledCredential


def test_compression_uses_live_codex_pool_entry_despite_other_model_cooldowns(tmp_path, monkeypatch):
    import agent.auxiliary_client as aux
    from agent.credential_pool import CredentialPool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))  # Never adopt the host's Codex CLI login.
    monkeypatch.setattr(CredentialPool, "_codex_quota_restored_upstream", lambda self, entry: False)
    until = time.time() + 86400
    rows = [
        PooledCredential(
            provider="openai-codex", id=f"account-{i}", label=f"account-{i}",
            priority=i, source="manual:device_code", auth_type="oauth",
            access_token=f"test-access-{i}", refresh_token=f"test-refresh-{i}",
            last_status="exhausted" if i == 0 else "ok",
            last_error_code=429 if i == 0 else None,
            last_error_reason="usage_limit_reached" if i == 0 else None,
            last_error_reset_at=until if i == 0 else None,
            model_cooldowns={"unrelated-model": until} if i else {},
        ).to_dict()
        for i in range(3)
    ]
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {"openai-codex": rows},
    }), encoding="utf-8")
    aux.shutdown_cached_clients()
    try:
        # Use the same cached-client route as call_llm(task="compression"), with real
        # auth-store loading, pool selection, and Codex transport construction.
        client, model = aux._get_cached_client("openai-codex", "gpt-6-sol", task="compression")
        assert model == "gpt-6-sol"
        assert client is not None
        assert client._real_client.api_key == "test-access-1"
    finally:
        aux.shutdown_cached_clients()


def test_raw_codex_resolves_requested_model_despite_other_model_cooldowns(tmp_path, monkeypatch):
    """The raw Responses client must use the same eligible pool entry as compression."""
    import agent.auxiliary_client as aux
    from agent.credential_pool import CredentialPool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(CredentialPool, "_codex_quota_restored_upstream", lambda self, entry: False)
    until = time.time() + 86400
    rows = [
        PooledCredential(
            provider="openai-codex", id=f"account-{i}", label=f"account-{i}",
            priority=i, source="manual:device_code", auth_type="oauth",
            access_token=f"fixture-access-{i}", refresh_token=f"fixture-refresh-{i}",
            last_status="exhausted" if i == 0 else "ok",
            last_error_code=429 if i == 0 else None,
            last_error_reason="usage_limit_reached" if i == 0 else None,
            last_error_reset_at=until if i == 0 else None,
            model_cooldowns={"unrelated-model": until} if i else {},
        ).to_dict()
        for i in range(2)
    ]
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {"openai-codex": rows},
    }), encoding="utf-8")
    raw, raw_model = aux.resolve_provider_client("openai-codex", model="gpt-6-sol", raw_codex=True)
    wrapped, wrapped_model = aux.resolve_provider_client("openai-codex", model="gpt-6-sol", raw_codex=False)
    try:
        assert raw_model == wrapped_model == "gpt-6-sol"
        assert raw is not None
        assert wrapped is not None
        assert raw.api_key == wrapped._real_client.api_key == "fixture-access-1"
        assert raw.responses is not None
    finally:
        if raw is not None:
            raw.close()
        if wrapped is not None:
            wrapped._real_client.close()

    # The same entry must become unavailable when the requested model itself cools down;
    # the globally exhausted entry must not be resurrected as an alternative.
    rows[1]["model_cooldowns"]["gpt-6-sol"] = until
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {"openai-codex": rows},
    }), encoding="utf-8")
    raw, raw_model = aux.resolve_provider_client("openai-codex", model="gpt-6-sol", raw_codex=True)
    assert raw is None and raw_model is None
    wrapped, wrapped_model = aux.resolve_provider_client("openai-codex", model="gpt-6-sol", raw_codex=False)
    assert wrapped is None and wrapped_model is None


def test_model_switch_feasibility_does_not_warn_with_model_eligible_codex_pool(tmp_path, monkeypatch):
    """The switch's eager probe must not mistake another model's 429 for missing auth."""
    import agent.auxiliary_client as aux
    from agent.conversation_compression import check_compression_model_feasibility
    from agent.credential_pool import CredentialPool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(CredentialPool, "_codex_quota_restored_upstream", lambda self, entry: False)
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {
        "provider": "openai-codex", "model": "gpt-6-sol",
    })
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *args, **kwargs: 900_000)
    until = time.time() + 86400
    rows = [
        PooledCredential(
            provider="openai-codex", id=f"account-{i}", label=f"account-{i}",
            priority=i, source="manual:device_code", auth_type="oauth",
            access_token=f"fixture-access-{i}", refresh_token=f"fixture-refresh-{i}",
            last_status="exhausted" if i == 0 else "ok",
            last_error_code=429 if i == 0 else None,
            last_error_reason="usage_limit_reached" if i == 0 else None,
            last_error_reset_at=until if i == 0 else None,
            model_cooldowns={"unrelated-model": until} if i else {},
        ).to_dict()
        for i in range(2)
    ]
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1, "credential_pool": {"openai-codex": rows},
    }), encoding="utf-8")
    notices = []
    compressor = SimpleNamespace(threshold_tokens=200_000, context_length=900_000)
    agent = SimpleNamespace(
        model="gpt-6-astra-900k", provider="openai-codex", base_url="",
        compression_enabled=True, context_compressor=compressor,
        _custom_providers=None, _aux_compression_context_length_config=None,
        _last_feasibility_notice=None, _compression_warning=None,
        _current_main_runtime=lambda: {"provider": "openai-codex", "model": "gpt-6-astra-900k"},
        _emit_diagnostic_status=notices.append,
    )
    aux.shutdown_cached_clients()
    try:
        check_compression_model_feasibility(agent)
        assert agent._compression_warning is None
        assert not notices
    finally:
        aux.shutdown_cached_clients()
