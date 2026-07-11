from agent.conversation_loop import _is_reasoning_effort_invalid_request
from agent.transports.codex import ResponsesApiTransport
from gateway.run import _gateway_provider_error_reply
from hermes_constants import clamp_reasoning_effort_for_provider


class FakeProviderError(Exception):
    def __init__(self, *, status_code, body):
        super().__init__(body)
        self.status_code = status_code
        self.body = body


def test_openai_codex_legacy_model_clamps_max_to_xhigh():
    effort, was_clamped, supported = clamp_reasoning_effort_for_provider(
        "max", "openai-codex", "gpt-5.5"
    )

    assert effort == "xhigh"
    assert was_clamped is True
    assert "max" not in supported


def test_openai_codex_gpt56_sol_clamps_ultra_to_xhigh():
    effort, was_clamped, supported = clamp_reasoning_effort_for_provider(
        "ultra", "openai-codex", "gpt-5.6-sol"
    )

    assert effort == "xhigh"
    assert was_clamped is True
    assert "ultra" not in supported


def test_openai_codex_gpt56_luna_clamps_ultra_to_xhigh():
    effort, was_clamped, supported = clamp_reasoning_effort_for_provider(
        "ultra", "openai-codex", "gpt-5.6-luna"
    )

    assert effort == "xhigh"
    assert was_clamped is True
    assert "ultra" not in supported


def test_codex_responses_transport_clamps_max_for_legacy_codex_model():
    kwargs = ResponsesApiTransport().build_kwargs(
        "gpt-5.5",
        [{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "max"},
        provider="openai-codex",
        is_codex_backend=True,
    )

    assert kwargs["reasoning"]["effort"] == "xhigh"


def test_codex_responses_transport_clamps_ultra_for_gpt56_sol():
    kwargs = ResponsesApiTransport().build_kwargs(
        "gpt-5.6-sol",
        [{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "ultra"},
        provider="openai-codex",
        is_codex_backend=True,
    )

    assert kwargs["reasoning"]["effort"] == "xhigh"


def test_unknown_provider_retains_global_ultra_support():
    effort, was_clamped, supported = clamp_reasoning_effort_for_provider(
        "ultra", "custom-provider", "custom-model"
    )

    assert effort == "ultra"
    assert was_clamped is False
    assert "ultra" in supported


def test_reasoning_effort_bad_request_detector_reads_structured_param():
    err = FakeProviderError(
        status_code=400,
        body={
            "error": {
                "message": "Invalid value: 'max'",
                "type": "invalid_request_error",
                "param": "reasoning.effort",
                "code": "invalid_value",
            }
        },
    )

    assert _is_reasoning_effort_invalid_request(err)


def test_gateway_reports_reasoning_effort_config_error():
    reply = _gateway_provider_error_reply(
        "HTTP 400 {'error': {'type': 'invalid_request_error', "
        "'param': 'reasoning.effort', 'message': \"Invalid value: 'max'\"}}"
    )

    assert "Provider rejected a configuration value" in reply
    assert "reasoning.effort" in reply
    assert "failed after retries" not in reply
