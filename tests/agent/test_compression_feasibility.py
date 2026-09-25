"""Tests for auxiliary prompt feasibility without capping the active model's trigger.

The deferred check preserves the 64K minimum and bounds summarizer input.
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


@pytest.fixture(autouse=True)
def _stable_aux_provider_config():
    """Keep feasibility tests independent from the developer's config.yaml."""
    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("auto", None, None, None, None),
    ):
        yield


def _make_agent(
    *,
    compression_enabled: bool = True,
    threshold_percent: float = 0.50,
    main_context: int = 200_000,
) -> AIAgent:
    """Build a minimal AIAgent with a compressor, skipping __init__."""
    agent = AIAgent.__new__(AIAgent)
    agent.model = "test-main-model"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "sk-test"
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent.compression_enabled = compression_enabled
    agent._print_fn = None
    agent.suppress_status_output = False
    agent._stream_consumers = []
    agent._executing_tools = False
    agent._mute_post_response = False
    agent.status_callback = None
    agent.tool_progress_callback = None
    agent._compression_warning = None
    agent._aux_compression_context_length_config = None
    agent._custom_providers = []
    agent.tools = []

    compressor = MagicMock(spec=ContextCompressor)
    compressor.context_length = main_context
    compressor.threshold_tokens = int(main_context * threshold_percent)
    compressor.summary_target_ratio = 0.20
    compressor.tail_token_budget = int(
        compressor.threshold_tokens * compressor.summary_target_ratio
    )
    agent.context_compressor = compressor

    return agent


@pytest.mark.parametrize("main_context,aux_context", [(1_000_000, 512_000), (400_000, 80_000)])
def test_aux_sync_keeps_lean_tail_policy(main_context, aux_context):
    """Auxiliary prompt sizing does not change the main trigger or lean retention."""
    agent = _make_agent(main_context=main_context)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=main_context,
        threshold_percent=0.85, quiet_mode=True,
    )
    before = compressor.tail_token_budget
    agent._emit_status = lambda message: None
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=aux_context):
        agent._check_compression_model_feasibility()
        assert compressor.threshold_tokens == int(main_context * 0.85)
        assert compressor._summary_input_aux_context == aux_context
        assert compressor.tail_token_budget == before
        # Repeated feasibility and subsequent model recalibration retain policy.
        agent._check_compression_model_feasibility()
        assert compressor.tail_token_budget == before
        compressor.update_model("test-main-model", context_length=main_context)
        assert compressor.tail_token_budget == before


def test_aux_sync_legacy_tail_keeps_main_trigger():
    """Explicit legacy retention follows the main trigger, not the auxiliary window."""
    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000,
        threshold_percent=0.85, tail_mode="legacy", quiet_mode=True,
    )
    before = compressor.tail_token_budget
    agent._emit_status = lambda message: None
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", return_value=512_000):
        agent._check_compression_model_feasibility()
    assert compressor.threshold_tokens == 850_000
    assert compressor.tail_token_budget == before
    assert compressor.tail_token_budget == int(compressor.threshold_tokens * compressor.summary_target_ratio)


def test_fallback_activation_on_never_probed_session_stays_lazy():
    """A session that never ran the feasibility probe does not resolve an auxiliary client while a
    fallback is being activated; the compaction-time probe still owns the first verdict (#114707)."""
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    agent = _make_agent(main_context=200_000)
    agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=200_000, threshold_percent=0.50, quiet_mode=True,
    )
    agent._config_context_length = None
    agent.model = "fallback-model"
    with patch("agent.auxiliary_client.get_text_auxiliary_client") as aux_client, \
         patch("agent.model_metadata.get_model_context_length", return_value=1_000_000):
        _update_fallback_context_compressor(agent)
    aux_client.assert_not_called()
    assert getattr(agent, "_compression_feasibility_checked", False) is False


def test_fallback_activation_reprobes_aux_window_without_clamping_main():
    """Runtime changes refresh auxiliary prompt sizing without clamping the main trigger."""
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    agent = _make_agent(main_context=200_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=200_000, threshold_percent=0.50, quiet_mode=True,
    )
    notices = 0

    def _count(_message):
        nonlocal notices
        notices += 1

    agent._emit_status = _count
    agent._config_context_length = None
    agent._compression_feasibility_checked = True  # probed on the primary; aux fit there
    agent.model = "fallback-model"
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", side_effect=[1_000_000, 80_000]):
        _update_fallback_context_compressor(agent)
    assert compressor.context_length == 1_000_000
    assert compressor.threshold_tokens == 500_000
    assert compressor._summary_input_aux_context == 80_000
    assert agent._compression_feasibility_checked is True
    # Same-runtime window correction follows the main model's new window.
    compressor.update_model(
        "fallback-model", context_length=800_000, base_url=agent.base_url, api_key=agent.api_key,
        provider=agent.provider, api_mode=agent.api_mode,
    )
    assert compressor.threshold_tokens == 400_000
    # No warning on the next runtime change (fallback/restore cycles).
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
         patch("agent.model_metadata.get_model_context_length", side_effect=[1_000_000, 80_000]):
        agent.base_url = "https://other-route.example/v1"  # runtime change, identical verdict text
        _update_fallback_context_compressor(agent)
    assert compressor.threshold_tokens == 500_000
    assert notices == 0


def test_unclamp_clears_stale_clamp_warning():
    """When a re-probe finds the summariser fits again, the stale 'auto-lowered' text must not survive
    for ``replay_compression_warning`` to resend on a session that is no longer clamped (#114707)."""
    from agent.chat_completion_helpers import _update_fallback_context_compressor

    agent = _make_agent(main_context=200_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=200_000, threshold_percent=0.50, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._config_context_length = None
    agent._compression_feasibility_checked = True
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    for main_ctx, aux_ctx, label in ((1_000_000, 80_000, "big"), (100_000, 80_000, "small")):
        agent.model = label
        with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")), \
             patch("agent.model_metadata.get_model_context_length", side_effect=[main_ctx, aux_ctx]):
            _update_fallback_context_compressor(agent)
    assert compressor._summary_input_aux_context == 80_000
    assert compressor.threshold_tokens == 75_000
    assert agent._compression_warning is None
    assert getattr(agent, "_last_feasibility_notice", None) is None


def test_near_threshold_probe_sizes_aux_prompt_before_first_compaction():
    """A fresh instance probes once its request first reaches the smallest window any summariser may
    have, sizing only the auxiliary prompt before the main-window trigger fires;
    requests below that stay probe-free (#28957)."""
    from agent.conversation_compression import ensure_compression_feasibility_checked
    from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

    agent = _make_agent(main_context=1_000_000)
    compressor = agent.context_compressor = ContextCompressor(
        "test-main-model", config_context_length=1_000_000, threshold_percent=0.75, quiet_mode=True,
    )
    agent._emit_status = lambda message: None
    agent._compression_feasibility_checked = False
    client = MagicMock(base_url="http://localhost/v1", api_key="test-key")
    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(client, "aux")) as aux_client, \
         patch("agent.model_metadata.get_model_context_length", return_value=80_000):
        ensure_compression_feasibility_checked(agent, MINIMUM_CONTEXT_LENGTH - 1)
        aux_client.assert_not_called()
        assert agent._compression_feasibility_checked is False
        ensure_compression_feasibility_checked(agent, MINIMUM_CONTEXT_LENGTH)
        ensure_compression_feasibility_checked(agent, 200_000)
    assert aux_client.call_count == 1
    assert agent._compression_feasibility_checked is True
    assert compressor.threshold_tokens == 750_000
    assert compressor._summary_input_aux_context == 80_000
    assert compressor.should_compress(200_000) is False


# ── Core warning logic ──────────────────────────────────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_small_aux_sizes_prompt_without_lowering_threshold(mock_get_client, mock_ctx_len):
    """An auxiliary window below the trigger cannot lower the main model's trigger."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    # threshold = 100,000 — aux has 80,000 (above 64K floor, below threshold)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert messages == []
    assert agent._compression_warning is None
    assert agent.context_compressor.threshold_tokens == 100_000
    assert agent.context_compressor._summary_input_aux_context == 80_000
    assert agent.context_compressor.tail_token_budget == 20_000


@patch("agent.model_metadata.get_model_context_length", return_value=32_768)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_rejects_aux_below_minimum_context(mock_get_client, mock_ctx_len):
    """Hard floor: aux context < MINIMUM_CONTEXT_LENGTH (64K) → session
    refuses to start (ValueError), mirroring the main-model rejection."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "tiny-aux-model")

    agent._emit_status = lambda msg: None

    with pytest.raises(ValueError) as exc_info:
        agent._check_compression_model_feasibility()

    err = str(exc_info.value)
    assert "tiny-aux-model" in err
    assert "32,768" in err
    assert "64,000" in err
    assert "below the minimum" in err




def test_feasibility_check_passes_live_main_runtime():
    """Compression feasibility should probe using the live session runtime."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    agent.model = "gpt-5.4"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_key = "codex-token"
    agent.api_mode = "codex_responses"

    mock_client = MagicMock()
    mock_client.base_url = "https://chatgpt.com/backend-api/codex"
    mock_client.api_key = "codex-token"

    with patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(mock_client, "gpt-5.4")) as mock_get_client, \
         patch("agent.model_metadata.get_model_context_length", return_value=200_000):
        agent._emit_status = lambda msg: None
        agent._check_compression_model_feasibility()

    mock_get_client.assert_called_once_with(
        "compression",
        main_runtime={
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
            "api_mode": "codex_responses",
            "auth_mode": "",
            "session_id": "",
        },
    )


@patch("agent.model_metadata.get_model_context_length", return_value=1_000_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_feasibility_check_passes_config_context_length(mock_get_client, mock_ctx_len):
    """auxiliary.compression.context_length from config is forwarded to
    get_model_context_length so custom endpoints that lack /models still
    report the correct context window (fixes #8499)."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.85)
    agent._aux_compression_context_length_config = 1_000_000
    mock_client = MagicMock()
    mock_client.base_url = "http://custom-endpoint:8080/v1"
    mock_client.api_key = "sk-custom"
    mock_get_client.return_value = (mock_client, "custom/big-model")

    agent._emit_status = lambda msg: None
    agent._check_compression_model_feasibility()

    mock_ctx_len.assert_called_once_with(
        "custom/big-model",
        base_url="http://custom-endpoint:8080/v1",
        api_key="sk-custom",
        config_context_length=1_000_000,
        provider="openrouter",
        custom_providers=[],
    )


@patch("agent.model_metadata.get_model_context_length", return_value=1_000_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_feasibility_inherits_matching_main_context_override(
    mock_get_client, mock_ctx_len
):
    agent = cast(Any, _make_agent(main_context=1_000_000, threshold_percent=0.85))
    agent.model = "gpt-5.6-sol"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_key = "codex-token"
    agent.api_mode = "codex_responses"
    agent._config_context_length = 1_000_000

    mock_client = MagicMock()
    mock_client.base_url = "https://chatgpt.com/backend-api/codex/"
    mock_client.api_key = "codex-token"
    mock_get_client.return_value = (mock_client, "gpt-5.6-sol")

    messages = []
    agent._emit_status = lambda message: messages.append(message)
    agent._check_compression_model_feasibility()

    # Matching runtime: use the already-resolved main context, not another provider probe.
    mock_ctx_len.assert_not_called()
    assert messages == []
    assert agent._compression_warning is None
    assert agent.context_compressor.threshold_tokens == 850_000




def test_init_feasibility_check_uses_aux_context_override_from_config():
    """Lazy feasibility check should cache and forward auxiliary.compression.context_length.

    NB: feasibility check is deferred from AIAgent.__init__ to the first
    actual compression attempt (saves ~400ms cold startup on short sessions
    that never trigger compression). The test drives the check explicitly
    via ``agent._check_compression_model_feasibility()`` to assert the
    config-override threading.
    """

    class _StubCompressor:
        def __init__(self, *args, **kwargs):
            self.context_length = 200_000
            self.threshold_tokens = 100_000
            self.threshold_percent = 0.50

        def get_tool_schemas(self):
            return []

        def on_session_start(self, *args, **kwargs):
            return None

    cfg = {
        "auxiliary": {
            "compression": {
                "context_length": 1_000_000,
            },
        },
    }
    mock_client = MagicMock()
    mock_client.base_url = "http://custom-endpoint:8080/v1"
    mock_client.api_key = "sk-custom"

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.agent_init.ContextCompressor", new=_StubCompressor),
        patch("agent.auxiliary_client.get_text_auxiliary_client", return_value=(mock_client, "custom/big-model")),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000) as mock_ctx_len,
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

        # Config override is captured eagerly in __init__ (still needed
        # because the threshold-derivation logic at construction time
        # consults it).
        assert agent._aux_compression_context_length_config == 1_000_000

        # The expensive feasibility probe is deferred. Drive it manually
        # to validate the call shape still forwards the override correctly.
        agent._check_compression_model_feasibility()

    mock_ctx_len.assert_called_once_with(
        "custom/big-model",
        base_url="http://custom-endpoint:8080/v1",
        api_key="sk-custom",
        config_context_length=1_000_000,
        provider="",
        custom_providers=[],
    )


@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_warns_when_no_auxiliary_provider(mock_get_client):
    """Warning emitted when no auxiliary provider is configured."""
    agent = _make_agent()
    mock_get_client.return_value = (None, None)

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert len(messages) == 1
    assert "No auxiliary LLM provider" in messages[0]
    assert agent._compression_warning is not None


def test_no_unavailable_warning_when_configured_fallback_chain_resolves():
    """Primary compression provider can be down if configured fallback works."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    fallback_client = MagicMock()
    fallback_client.base_url = "https://chatgpt.com/backend-api/codex"
    fallback_client.api_key = "codex-oauth-token"

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    with patch(
        "agent.auxiliary_client._resolve_task_provider_model",
        return_value=("ollama-cloud", "deepseek-v4-flash:cloud", None, None, None),
    ), patch(
        "agent.auxiliary_client.get_text_auxiliary_client",
        return_value=(None, None),
    ), patch(
        "agent.auxiliary_client._try_configured_fallback_for_unavailable_client",
        return_value=(fallback_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)"),
    ) as mock_fallback, patch(
        "agent.model_metadata.get_model_context_length",
        return_value=200_000,
    ) as mock_ctx_len:
        agent._check_compression_model_feasibility()

    assert messages == []
    assert agent._compression_warning is None
    mock_fallback.assert_called_once_with("compression", "ollama-cloud")
    mock_ctx_len.assert_called_once()
    assert mock_ctx_len.call_args.args == ("gpt-5.4-mini",)
    assert mock_ctx_len.call_args.kwargs["provider"] == "openai-codex"










# ── Two-phase: __init__ + run_conversation replay ───────────────────


@patch("agent.model_metadata.get_model_context_length", return_value=80_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_smaller_aux_has_no_spurious_gateway_warning(mock_get_client, mock_ctx_len):
    """An auxiliary below the main trigger is not a session warning."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    vprint_messages = []
    agent._emit_status = lambda msg: vprint_messages.append(msg)
    agent._check_compression_model_feasibility()

    assert vprint_messages == []
    assert agent._compression_warning is None

    callback_events = []
    agent.status_callback = lambda ev, msg: callback_events.append((ev, msg))
    agent._replay_compression_warning()
    assert callback_events == []


@patch("agent.model_metadata.get_model_context_length", return_value=200_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_no_replay_when_no_warning(mock_get_client, mock_ctx_len):
    """_replay_compression_warning is a no-op when there's no stored warning."""
    agent = _make_agent(main_context=200_000, threshold_percent=0.50)
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "big-model")

    agent._emit_status = lambda msg: None
    agent._check_compression_model_feasibility()

    assert agent._compression_warning is None

    callback_events = []
    agent.status_callback = lambda ev, msg: callback_events.append((ev, msg))
    agent._replay_compression_warning()

    assert len(callback_events) == 0






# ── #67422: threshold suggestion must survive the small-context floor ────────




@patch("agent.model_metadata.get_model_context_length", return_value=300_000)
@patch("agent.auxiliary_client.get_text_auxiliary_client")
def test_large_context_keeps_main_threshold_with_small_aux(mock_get_client, mock_ctx_len):
    """No lower-threshold suggestion for a usable, smaller auxiliary model."""
    agent = _make_agent(main_context=1_000_000, threshold_percent=0.50)
    # threshold = 500,000 — aux has 300,000
    mock_client = MagicMock()
    mock_client.base_url = "https://openrouter.ai/api/v1"
    mock_client.api_key = "sk-aux"
    mock_get_client.return_value = (mock_client, "google/gemini-3-flash-preview")

    messages = []
    agent._emit_status = lambda msg: messages.append(msg)

    agent._check_compression_model_feasibility()

    assert messages == []
    assert agent.context_compressor.threshold_tokens == 500_000




