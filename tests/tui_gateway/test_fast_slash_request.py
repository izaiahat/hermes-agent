"""Slash fast mirrors must change the next request, not just the status badge."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.chat_completion_helpers import build_api_kwargs
from agent.transports.codex import ResponsesApiTransport
import tui_gateway.server as server


@pytest.mark.parametrize("mode", ["normal", "off", "fast", "on"])
def test_fast_mirror_updates_next_request_without_changing_context(mode, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("request construction must not use the network")

    monkeypatch.setattr("socket.socket.connect", no_network)
    transport = ResponsesApiTransport()
    agent = SimpleNamespace(
        model="gpt-6-astra-900k", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses",
        reasoning_config={"enabled": True, "effort": "high"},
        service_tier="priority", request_overrides={
            "service_tier": "priority", "speed": "fast", "extra_headers": {"X-Test": "keep"}},
        session_id="offline-fast-mirror", tools=[], max_tokens=None,
        _get_transport=lambda: transport,
        _prepare_messages_for_non_vision_model=lambda messages: messages,
        _resolved_api_call_timeout=lambda: 120,
    )
    messages = [{"role": "system", "content": "Keep the prompt stable."},
                {"role": "user", "content": "Offline request construction only."}]
    original_messages = deepcopy(messages)
    original_reasoning = deepcopy(agent.reasoning_config)
    session = {"session_key": "offline-fast-mirror", "agent": agent, "history": messages}
    before = build_api_kwargs(agent, messages)
    assert before["service_tier"] == "priority"

    with patch.object(server, "_emit"), patch.object(server, "_session_info", return_value={}), \
            patch.object(server, "_emit_session_info"), \
            patch.object(server, "_persist_live_session_runtime"), \
            patch.object(server, "_write_config_key") as write_config:
        server._mirror_fast("offline-fast-mirror", session, agent, mode)

    after = build_api_kwargs(agent, messages)
    if mode in {"normal", "off"}:
        assert "service_tier" not in after
        assert agent.service_tier is None
        assert "service_tier" not in agent.request_overrides
        assert session["create_service_tier_override"] == ""
    else:
        assert after["service_tier"] == "priority"
        assert agent.service_tier == "priority"
    assert "speed" not in agent.request_overrides
    assert "speed" not in after
    assert after["extra_headers"]["X-Test"] == "keep"
    assert after["model"] == before["model"]
    assert after["reasoning"] == before["reasoning"]
    assert after["input"] == before["input"]
    assert agent.reasoning_config == original_reasoning
    assert messages == original_messages
    write_config.assert_not_called()
