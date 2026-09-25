"""Active-model context windows, independent of a smaller summary model."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from hermes_cli.config import load_config
from agent.context_compressor import ContextCompressor
from agent.conversation_compression import revalidate_compression_feasibility
from agent.model_metadata import estimate_tokens_rough, get_model_context_length
from agent.native_compaction import resolve_compact_threshold
from run_agent import AIAgent


@pytest.fixture(autouse=True)
def _stub_anthropic_transport_for_fresh_agent(monkeypatch):
    # Exercise the real AIAgent init/config pipeline, without requiring the optional SDK or login.
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", lambda *args, **kwargs: MagicMock())
    # pytest's isolated HERMES_HOME has no operator config; supply the same threshold as that TUI.
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["compression"]["threshold"] = 0.85
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)


@pytest.mark.parametrize("model,provider,window", [
    ("claude-opus-5-5[1m]", "claude-subscription-directsdk-experimental", 1_000_000),
    ("gpt-6-sol", "openai-codex", 1_050_000),
    ("gpt-6-astra-900k", "openai-codex", 900_000),
])
def test_fresh_tui_style_agent_uses_active_window_without_implicit_cap(model, provider, window):
    # The TUI's _make_agent constructs AIAgent with these route and quiet-mode args.
    agent = AIAgent(model=model, provider=provider, quiet_mode=True,
                    api_key="local-test-placeholder", base_url="https://chatgpt.com/backend-api/codex" if provider == "openai-codex" else "https://api.anthropic.com",
                    enabled_toolsets=[], session_db=None, skip_memory=True, skip_context_files=True)
    cc = agent.context_compressor
    assert cc.context_length == window
    assert cc.threshold_tokens_cap is None
    assert cc.threshold_tokens == int(window * 0.85)


def test_fresh_agent_model_switch_opus_sol_opus_tracks_each_window():
    agent = AIAgent(model="claude-opus-5-5[1m]",
                    provider="claude-subscription-directsdk-experimental", quiet_mode=True,
                    api_key="local-test-placeholder", base_url="https://api.anthropic.com",
                    enabled_toolsets=[], session_db=None, skip_memory=True, skip_context_files=True)
    cc = agent.context_compressor
    assert (cc.context_length, cc.threshold_tokens) == (1_000_000, 850_000)
    # Use the same compressor update routine as /model, including re-resolution and feasibility.
    from agent.agent_runtime_helpers import _update_switch_compressor
    for model, provider, context, expected in (
        ("gpt-6-sol", "openai-codex", 1_050_000, 892_500),
        ("claude-opus-5-5[1m]", "claude-subscription-directsdk-experimental", 1_000_000, 850_000),
    ):
        agent.model, agent.provider = model, provider
        agent.base_url = "https://chatgpt.com/backend-api/codex" if provider == "openai-codex" else "https://api.anthropic.com"
        _update_switch_compressor(agent, [], None, {})
        assert (cc.context_length, cc.threshold_tokens) == (context, expected)


def test_codex_sol_window_survives_catalog_miss():
    with patch("agent.models_dev.lookup_models_dev_context", return_value=None):
        assert get_model_context_length(
            "gpt-6-sol", provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
        ) == 1_050_000


def test_aux_window_does_not_lower_active_model_trigger_on_switch():
    compressor = ContextCompressor(
        "gpt-6-sol", config_context_length=1_050_000,
        threshold_percent=0.85, quiet_mode=True,
    )
    agent = SimpleNamespace(
        model="gpt-6-sol", provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex", api_key="",
        api_mode="codex_responses", compression_enabled=True,
        context_compressor=compressor, _custom_providers=[],
        _aux_compression_context_length_config=None,
        _compression_warning=None, _last_feasibility_notice=None,
        _compression_feasibility_checked=False,
        _current_main_runtime=lambda: {},
        _emit_status=lambda _: None, status_callback=None,
    )
    client = MagicMock(base_url="https://chatgpt.com/backend-api/codex", api_key="")
    with patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", None, None, None, None)), \
         patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=256_000):
        revalidate_compression_feasibility(agent)
    assert agent._compression_feasibility_checked is True
    assert compressor.threshold_tokens == 892_500
    agent.model = "claude-opus-5-5[1m]"
    compressor.update_model(agent.model, context_length=1_000_000)
    agent._compression_feasibility_checked = False
    with patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", None, None, None, None)), \
         patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=256_000):
        revalidate_compression_feasibility(agent)
    assert compressor.threshold_tokens == 850_000
    # A leftover auxiliary ceiling from an older process must not silently cap recomputations.
    setattr(compressor, "_aux_context_ceiling", 256_000)
    compressor.update_model(agent.model, context_length=1_000_000)
    assert compressor.threshold_tokens == 850_000
    compressor._previous_summary = "語" * 160_000
    prompt = compressor._build_summary_prompt("文" * 160_000, 512, None, "", False)
    assert estimate_tokens_rough(prompt) <= int(256_000 * 0.80)
    assert "summary input truncated" in prompt


def test_default_profile_native_trigger_tracks_active_window():
    if not (Path.home() / ".hermes/config.yaml").exists():
        pytest.skip("No installed default-profile config")
    compression = load_config()["compression"]
    assert compression["codex_responses_compact_threshold"] is None
    assert resolve_compact_threshold(compression["codex_responses_compact_threshold"], 765_000) == 756_808
