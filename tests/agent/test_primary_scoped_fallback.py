"""A configured Claude-only backup must not replace a pinned Codex seat."""
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from hermes_cli.auth import AuthError
from hermes_cli.runtime_provider import resolve_runtime_with_fallback


CLAUDE = "claude-subscription-directsdk-experimental"
BACKUP = {"provider": "openai-codex", "model": "gpt-6-astra-900k", "from_provider": CLAUDE}


@pytest.mark.parametrize("primary,model,expected", [
    (CLAUDE, "claude-opus-5-5[1m]", True),
    ("openai-codex", "gpt-6-sol", False),
])
def test_primary_scoped_fallback_activation(primary, model, expected, monkeypatch):
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", lambda *a, **kw: MagicMock())
    base_url = "https://api.anthropic.com" if primary == CLAUDE else "https://chatgpt.com/backend-api/codex"
    with patch("model_tools.get_tool_definitions", return_value=[]):
        agent = AIAgent(
            model=model, provider=primary, api_key="test-only", base_url=base_url,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model=[BACKUP], enabled_toolsets=[],
        )
    fallback_client = MagicMock(base_url="https://chatgpt.com/backend-api/codex", api_key="test-only")
    with patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client, BACKUP["model"])) as resolver:
        switched = agent._try_activate_fallback()
    assert switched is expected
    assert (agent.provider, agent.model) == (
        (BACKUP["provider"], BACKUP["model"]) if expected else (primary, model)
    )
    if not expected:
        resolver.assert_not_called()


@pytest.mark.parametrize("primary,should_switch", [(CLAUDE, True), ("openai-codex", False)])
def test_pre_agent_fallback_is_primary_scoped(primary, should_switch, monkeypatch):
    def resolve(**kwargs):
        if kwargs["requested"] == primary:
            raise AuthError("primary unavailable")
        return {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    config = {"model": {"provider": primary}, "fallback_providers": [BACKUP]}
    if should_switch:
        runtime, entry = resolve_runtime_with_fallback(config, requested=primary)
        assert runtime["provider"] == "openai-codex"
        assert entry == BACKUP
    else:
        with pytest.raises(AuthError, match="primary unavailable"):
            resolve_runtime_with_fallback(config, requested=primary)


@pytest.mark.parametrize("primary,should_switch", [(CLAUDE, True), ("openai-codex", False)])
def test_tui_pre_agent_fallback_is_primary_scoped(primary, should_switch, monkeypatch):
    from tui_gateway import server

    def resolve(**kwargs):
        if kwargs["requested"] == primary:
            raise AuthError("primary unavailable")
        return {"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    monkeypatch.setattr(server, "_load_fallback_model", lambda: [BACKUP])
    if should_switch:
        result = server._resolve_runtime_with_fallback({"requested": primary})
        assert result.used_fallback and result.selected_model == BACKUP["model"]
    else:
        with pytest.raises(AuthError, match="primary unavailable"):
            server._resolve_runtime_with_fallback({"requested": primary})
