"""Regression test for #11884: _make_agent must resolve runtime provider.

Without resolve_runtime_provider(), bare-slug models in config
(e.g. ``claude-opus-4-6`` with ``model.provider: anthropic``) leave
provider/base_url/api_key empty in AIAgent, causing HTTP 404.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_make_agent_passes_resolved_provider():
    """_make_agent forwards provider/base_url/api_key/api_mode from
    resolve_runtime_provider to AIAgent."""

    fake_runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test-key",
        "api_mode": "anthropic_messages",
        "command": None,
        "args": None,
        "credential_pool": None,
    }

    fake_cfg = {
        "model": {"default": "claude-opus-4-6", "provider": "anthropic"},
        "agent": {"system_prompt": "test", "max_turns": 2200},
    }

    with patch("tui_gateway.server._load_cfg", return_value=fake_cfg), \
         patch("tui_gateway.server._get_db", return_value=MagicMock()), \
         patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"), \
         patch("tui_gateway.server._load_reasoning_config", return_value=None), \
         patch("tui_gateway.server._load_service_tier", return_value=None), \
         patch("tui_gateway.server._load_enabled_toolsets", return_value=None), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=fake_runtime) as mock_resolve, \
         patch("run_agent.AIAgent") as mock_agent:

        from tui_gateway.server import _make_agent
        _make_agent("sid-1", "key-1")

        mock_resolve.assert_called_once_with(requested=None)

        call_kwargs = mock_agent.call_args
        assert call_kwargs.kwargs["provider"] == "anthropic"
        assert call_kwargs.kwargs["base_url"] == "https://api.anthropic.com"
        assert call_kwargs.kwargs["api_key"] == "sk-test-key"
        assert call_kwargs.kwargs["api_mode"] == "anthropic_messages"
        assert call_kwargs.kwargs["max_iterations"] == 2200


def test_background_agent_kwargs_uses_nested_agent_max_turns():
    """Background TUI runs must inherit agent.max_turns, not a root-level fallback."""

    fake_cfg = {"agent": {"max_turns": 2200}}
    fake_agent = SimpleNamespace(
        base_url="https://api.example.test/v1",
        api_key="sk-test-key",
        provider="custom",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=None,
        model="model-a",
        enabled_toolsets=["terminal"],
        ephemeral_system_prompt=None,
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        provider_require_parameters=False,
        provider_data_collection=None,
        reasoning_config=None,
        service_tier=None,
        request_overrides={},
        _fallback_model=None,
    )

    with patch("tui_gateway.server._load_cfg", return_value=fake_cfg), \
         patch("tui_gateway.server._load_reasoning_config", return_value=None), \
         patch("tui_gateway.server._load_service_tier", return_value=None), \
         patch("tui_gateway.server._get_db", return_value=MagicMock()):
        from tui_gateway.server import _background_agent_kwargs

        kwargs = _background_agent_kwargs(fake_agent, "task-1")

    assert kwargs["max_iterations"] == 2200
