from agent.conversation_loop import _is_reasoning_effort_invalid_request
from agent.transports.codex import ResponsesApiTransport
from gateway.run import _gateway_provider_error_reply
from hermes_constants import clamp_reasoning_effort_for_provider


class FakeProviderError(Exception):
    def __init__(self, *, status_code, body):
        super().__init__(body)
        self.status_code = status_code
        self.body = body


def test_openai_codex_clamps_max_to_xhigh():
    effort, was_clamped, supported = clamp_reasoning_effort_for_provider(
        "max", "openai-codex"
    )

    assert effort == "xhigh"
    assert was_clamped is True
    assert "max" not in supported


def test_codex_responses_transport_never_emits_max_for_codex_backend():
    kwargs = ResponsesApiTransport().build_kwargs(
        "gpt-5.5",
        [{"role": "user", "content": "hi"}],
        tools=None,
        reasoning_config={"enabled": True, "effort": "max"},
        provider="openai-codex",
        is_codex_backend=True,
    )

    assert kwargs["reasoning"]["effort"] == "xhigh"


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
