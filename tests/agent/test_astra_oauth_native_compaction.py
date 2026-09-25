"""Exact gpt-6-astra is native-compaction eligible only on official Codex OAuth (#103720).

Both the destination capability (``resolve_native_compaction_capabilities``) and the
per-request gate (``native_compaction_context_management``) must agree, and the request
gate must exclude Astra relays even when a trusted proxy advertises native compaction.
"""

from types import SimpleNamespace

import pytest

from agent.native_compaction import (
    native_compaction_context_management,
    resolve_native_compaction_capabilities,
)

_CODEX = "https://chatgpt.com/backend-api/codex"


@pytest.mark.parametrize("model,provider,base_url,eligible", [
    ("gpt-6-astra", "openai-codex", _CODEX, True),
    ("GPT-6-ASTRA", "openai-codex", "https://chatgpt.com:443/backend-api/codex/", True),
    ("gpt-6-astra", "openai", "https://api.openai.com/v1", False),
    ("gpt-6-astra", "openai", _CODEX, False),
    ("gpt-6-astra", "openai-codex", "https://relay.example/v1", False),
    ("gpt-6-astra", "openai-codex", "https://chatgpt.com.example/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", "http://chatgpt.com/backend-api/codex", False),
    ("gpt-6-astra", "openai-codex", None, False),
    ("gpt-6-astra-mini", "openai-codex", _CODEX, False),
    ("gpt-6-other", "openai-codex", _CODEX, False),
    ("gpt-5.6", "openai", "https://api.openai.com/v1", True),
    ("gpt-5.6", "openai-codex", _CODEX, True),
])
def test_astra_capability_and_request_gate_agree(model, provider, base_url, eligible):
    is_codex = provider == "openai-codex"
    resolved = resolve_native_compaction_capabilities(
        model=model, provider=provider, base_url=base_url, is_codex_backend=is_codex,
    )
    assert resolved["native_compaction"] is eligible
    agent = SimpleNamespace(
        model=model, provider=provider, base_url=base_url,
        codex_responses_native_compaction=True, compression_enabled=True,
        capabilities={"openai_native_compaction": True},
    )
    for runtime in (None, resolved):
        agent.runtime_capabilities = runtime
        payload = native_compaction_context_management(agent, is_codex_backend=is_codex)
        assert (payload is not None) is eligible


@pytest.mark.parametrize("provider,base_url,eligible", [
    ("openai-codex", _CODEX, True),
    ("openai", "https://api.openai.com/v1", False),
    ("openai", _CODEX, False),
    ("openai-codex", "https://relay.example/v1", False),
    ("openai-codex", "https://chatgpt.com.example/backend-api/codex", False),
    ("openai-codex", "http://chatgpt.com/backend-api/codex", False),
    ("openai-codex", None, False),
])
def test_astra_context_alias_compaction_before_wire_normalization(provider, base_url, eligible):
    from hermes_cli.model_normalize import normalize_model_for_provider
    from agent.transports.codex import ResponsesApiTransport

    # Runtime normalization preserves the opt-in; only the transport strips it.
    model = normalize_model_for_provider("gpt-6-astra-900k", provider)
    assert model == "gpt-6-astra-900k"
    is_codex = provider == "openai-codex"
    resolved = resolve_native_compaction_capabilities(
        model=model, provider=provider, base_url=base_url, is_codex_backend=is_codex,
    )
    assert resolved == {"native_compaction": eligible}
    agent = SimpleNamespace(
        model=model, provider=provider, base_url=base_url,
        codex_responses_native_compaction=True, compression_enabled=True,
        capabilities={"openai_native_compaction": True},
        context_compressor=SimpleNamespace(threshold_tokens=436_000),
    )
    for runtime in (None, resolved):
        agent.runtime_capabilities = runtime
        payload = native_compaction_context_management(agent, is_codex_backend=is_codex)
        if not eligible:
            assert payload is None
            continue
        assert payload == [{"type": "compaction", "compact_threshold": 427_808}]
        kwargs = ResponsesApiTransport().build_kwargs(
            model=model, messages=[{"role": "user", "content": "Hi"}], tools=[],
            provider=provider, base_url=base_url, is_codex_backend=is_codex,
            context_management=payload,
        )
        assert kwargs["model"] == "gpt-6-astra"
        assert kwargs["context_management"] == payload


@pytest.mark.parametrize("model,context,baseline,scaled", [
    ("gpt-6-astra-900k", 872_000, 272_000, True),
    ("gpt-6-astra-900k", 600_000, 272_000, True),
    ("gpt-6-astra-900k", 872_000, None, False),
    ("gpt-6-astra-900k", 272_000, 272_000, False),
    ("gpt-6-astra", 872_000, 272_000, False),
    ("gpt-5.6-sol-900k", 872_000, 272_000, False),
    ("gpt-5.6-luna-900k", 872_000, 272_000, False),
    ("claude-opus-4-6", 872_000, 272_000, False),
])
def test_astra_proportional_compression_is_model_isolated(monkeypatch, model, context, baseline, scaled):
    from copy import deepcopy
    from agent.agent_init import _build_context_engine, _parse_compression_config
    from agent.native_compaction import resolve_compact_threshold

    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **kw: context)
    cfg = {"compression": {
        "astra_extended_context_reference_tokens": baseline,
        "threshold": .85, "target_ratio": .2, "protect_last_n": 16, "protect_first_n": 3,
        "threshold_tokens": 256_000, "proactive_prune_tokens": 48_000,
        "proactive_prune_min_result_chars": 8_000, "proactive_prune_min_reclaim_tokens": 4_096,
        "tool_output_retention_turns": 5, "tool_output_retention_min_chars": 200,
        "tool_output_retention_max_inline_chars": 25_000, "tool_output_retention_min_inline_results": 3,
        "hygiene_hard_message_limit": 300, "codex_responses_native": True,
        "codex_responses_compact_threshold": 231_000,
    }}
    original = deepcopy(cfg)
    agent = SimpleNamespace(model=model, provider="openai-codex", base_url=_CODEX,
                            api_mode="codex_responses", max_tokens=None, quiet_mode=True, session_id="test")
    cs = _parse_compression_config(agent, cfg)
    _build_context_engine(agent, cfg, cs, [], None, None)
    cc = agent.context_compressor
    scale = lambda value: value * context // 272_000 if scaled else value
    assert cc.threshold_tokens_cap == scale(256_000)
    assert cc.threshold_tokens == min(int(context * .85), scale(256_000))
    assert cc.proactive_prune_tokens == scale(48_000)
    assert cc.proactive_prune_min_result_chars == scale(8_000)
    assert cc.proactive_prune_min_reclaim_tokens == scale(4_096)
    assert agent._tool_output_retention_min_chars == scale(200)
    assert agent._tool_output_retention_max_inline_chars == scale(25_000)
    expected_native = scale(223_008) if scaled else resolve_compact_threshold(231_000, cc.threshold_tokens)
    assert resolve_compact_threshold(agent.codex_responses_compact_threshold, cc.threshold_tokens) == expected_native
    assert (cc.threshold_percent, cc.summary_target_ratio, cc.protect_first_n, cc.protect_last_n) == (.85, .2, 3, 16)
    assert (agent._tool_output_retention_turns, agent._tool_output_retention_min_inline_results) == (5, 3)
    assert cfg == original


@pytest.mark.parametrize("transition", ["switch", "fallback"])
@pytest.mark.parametrize("start_extended", [True, False])
@pytest.mark.parametrize("other,provider,url,mode", [
    ("gpt-5.6-sol-900k", "openai-codex", _CODEX, "codex_responses"),
    ("gpt-5.6-luna-900k", "openai-codex", _CODEX, "codex_responses"),
    ("claude-opus-4-6", "anthropic", "https://api.anthropic.com", "anthropic_messages"),
])
def test_astra_budgets_follow_runtime_transitions(monkeypatch, transition, start_extended, other, provider, url, mode):
    """Exercise real switch/fallback/restore entry points; only clients/probes are stubbed."""
    from unittest.mock import MagicMock
    from run_agent import AIAgent
    from agent.agent_init import _build_context_engine, _parse_compression_config
    from agent.agent_runtime_helpers import _build_primary_runtime_snapshot

    monkeypatch.setattr("agent.context_compressor.get_model_context_length", lambda *a, **kw: 872_000)
    monkeypatch.setattr("agent.model_metadata.get_model_context_length", lambda *a, **kw: 872_000)
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda *a, **kw: None)
    monkeypatch.setattr("agent.conversation_compression.revalidate_compression_feasibility", lambda *a: None)
    monkeypatch.setattr("agent.agent_runtime_helpers._build_switched_client", lambda *a: None)
    monkeypatch.setattr("agent.agent_runtime_helpers._rebuild_primary_client", lambda *a, **kw: None)
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", lambda *a, **kw: MagicMock())
    cfg = {"compression": {
        "astra_extended_context_reference_tokens": 272_000,
        "threshold": .85, "threshold_tokens": 256_000,
        "proactive_prune_tokens": 48_000, "proactive_prune_min_result_chars": 8_000,
        "proactive_prune_min_reclaim_tokens": 4_096,
        "tool_output_retention_min_chars": 200, "tool_output_retention_max_inline_chars": 25_000,
        "codex_responses_native": True, "codex_responses_compact_threshold": 231_000,
    }}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **kw: cfg)
    astra_route = ("gpt-6-astra-900k", "openai-codex", _CODEX, "codex_responses")
    other_route = (other, provider, url, mode)
    initial = astra_route if start_extended else other_route
    other, provider, url, mode = other_route if start_extended else astra_route
    agent = AIAgent.__new__(AIAgent)
    for name, value in dict(
        model=initial[0], provider=initial[1], requested_provider=initial[1],
        base_url=initial[2], api_mode=initial[3], api_key="test-only", max_tokens=None,
        quiet_mode=True, session_id="test", _client_kwargs={}, client=MagicMock(),
        _anthropic_api_key="test-only", _anthropic_base_url=None, _is_anthropic_oauth=False,
        _use_prompt_caching=False, _use_native_cache_layout=False, _fallback_activated=False,
        _fallback_index=0, _fallback_chain=[], _config_context_length=None,
    ).items():
        setattr(agent, name, value)
    _build_context_engine(agent, cfg, _parse_compression_config(agent, cfg), [], None, None)
    agent._primary_runtime = _build_primary_runtime_snapshot(agent, agent.api_mode)
    cc = agent.context_compressor

    def budgets():
        return (cc.threshold_tokens_cap, cc.threshold_tokens, cc.proactive_prune_tokens,
                cc.proactive_prune_min_result_chars, cc.proactive_prune_min_reclaim_tokens,
                agent._tool_output_retention_min_chars, agent._tool_output_retention_max_inline_chars,
                agent.codex_responses_compact_threshold, cc._micro_compact_defrag_threshold_tokens)

    original = budgets()
    unscaled = (256_000, 256_000, 48_000, 8_000, 4_096, 200, 25_000, 231_000, 2_000)
    extended = (820_705, 741_200, 153_882, 25_647, 13_131, 641, 80_147, 714_937, 6_411)
    assert original == (extended if start_extended else unscaled)
    for _ in range(2):
        if transition == "switch":
            agent.switch_model(other, provider, api_key="test-only", base_url=url, api_mode=mode)
        else:
            client = MagicMock(base_url=url, api_key="test-only")
            monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", lambda *a, **kw: (client, None))
            agent._fallback_chain = [{"model": other, "provider": provider, "base_url": url, "api_mode": mode}]
            assert agent._try_activate_fallback()
        assert agent.model == other
        assert budgets() == (unscaled if start_extended else extended)
        if transition == "switch":
            agent.switch_model(initial[0], initial[1], api_key="test-only", base_url=initial[2], api_mode=initial[3])
        else:
            assert agent._restore_primary_runtime()
        assert agent.model == initial[0]
        assert budgets() == original  # no cumulative multiplication on repeated cycles
