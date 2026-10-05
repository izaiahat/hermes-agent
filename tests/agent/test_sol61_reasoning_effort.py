"""GPT-6.1 Sol's documented reasoning ladder must survive Codex wire clamping."""

from agent.reasoning_effort import clamp_reasoning_config, route_supported_efforts
from agent.transports.codex import ResponsesApiTransport


def test_sol61_preserves_max_on_codex_wire():
    """Regression: the legacy vocabulary silently downgraded Sol 6.1 max to xhigh."""
    model = "gpt-6.1-sol"
    transport = ResponsesApiTransport()
    for requested, expected in (
        ("max", "max"),
        ("medium", "medium"),
        ("ultra", "max"),
        ("none", "low"),
        ("minimal", "low"),
        ("low", "low"),
        ("high", "high"),
        ("xhigh", "xhigh"),
    ):
        config = {"enabled": True, "effort": requested}
        kwargs = transport.build_kwargs(
            model,
            [{"role": "user", "content": "Test reasoning wire metadata."}],
            provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            is_codex_backend=True,
            reasoning_config=config,
        )
        assert kwargs["model"] == model
        assert kwargs["reasoning"]["effort"] == expected, requested
        # The agent's entry clamp must not degrade the effort before transport construction.
        clamped = clamp_reasoning_config(
            config, route_supported_efforts("openai-codex", model)
        )
        assert clamped is not None
        assert clamped["effort"] == expected, requested
