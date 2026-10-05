"""Regression tests for OpenAI Codex model validation when the listing lags behind
actually usable backend model IDs.

The bug originally reported in #16172: `/model` and `switch_model()` rejected
`gpt-5.3-codex-spark` because the curated listing omitted it, even though direct
runtime calls succeeded. PR #19729 fixed this by soft-accepting unknown-but-
plausible Codex slugs with a warning, and this test pins the soft-accept
behavior so it doesn't regress.

Note: gpt-5.3-codex-spark itself is now in the curated catalog (PR #22991),
so the real-world Spark request takes the `recognized=True` fast path. This
test still uses Spark as the example slug but explicitly mocks
``provider_model_ids`` to omit it, exercising the soft-accept path generically
for any future entitlement-gated Codex slug that ships before Hermes catalogs
it.
"""

from unittest.mock import patch

import pytest

from hermes_cli.model_switch import switch_model
from hermes_cli.models_validate import validate_requested_model


def test_openai_codex_unknown_but_plausible_model_is_accepted_with_warning():
    """If the Codex listing is incomplete, `/model` should soft-accept the model
    with a warning instead of hard-rejecting it.
    """
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"],
    ):
        result = validate_requested_model("gpt-5.3-codex-spark", "openai-codex")

    assert result["accepted"] is True
    assert result["persist"] is True
    assert result["recognized"] is False
    assert "gpt-5.3-codex-spark" in result["message"]
    assert "gpt-5.3-codex" in result["message"]


def test_switch_model_allows_openai_codex_model_missing_from_listing():
    """switch_model() should succeed for Codex models that the runtime accepts
    even when the listing has not caught up yet.
    """
    with patch(
        "hermes_cli.models.provider_model_ids",
        return_value=["gpt-5.5", "gpt-5.4", "gpt-5.3-codex"],
    ):
        result = switch_model(
            "gpt-5.3-codex-spark",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="",
            current_api_key="",
            user_providers=None,
        )

    assert result.success is True
    assert result.new_model == "gpt-5.3-codex-spark"
    assert result.target_provider == "openai-codex"
    assert result.warning_message
    assert "OpenAI Codex model listing" in result.warning_message


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6-astra-900k", "gpt-6-luna"])
def test_config_metadata_does_not_make_builtin_codex_a_custom_endpoint(model):
    """A providers.openai-codex metadata row must keep the Codex catalog validator."""
    from hermes_cli.auth import DEFAULT_CODEX_BASE_URL

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
        "api_key": "test-token", "base_url": DEFAULT_CODEX_BASE_URL,
        "api_mode": "codex_responses",
    }), patch("hermes_cli.models.provider_model_ids", return_value=[
        "gpt-6-sol", "gpt-6-astra-900k", "gpt-6-luna",
    ]), patch("hermes_cli.models.probe_api_models") as probe:
        result = switch_model(
            model, current_provider="openai-codex", current_model="gpt-6-astra-900k",
            current_base_url=DEFAULT_CODEX_BASE_URL, current_api_key="test-token",
            explicit_provider="openai-codex",
            user_providers={"openai-codex": {"models": {"gpt-6-sol": {"context_length": 1050000}},
                                               "request_timeout_seconds": 120}},
        )

    assert result.success is True, result.error_message
    assert result.target_provider == "openai-codex"
    assert result.new_model == model
    probe.assert_not_called()


def test_codex_metadata_row_keeps_unknown_foreign_model_rejected():
    from hermes_cli.auth import DEFAULT_CODEX_BASE_URL

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
        "api_key": "test-token", "base_url": DEFAULT_CODEX_BASE_URL,
        "api_mode": "codex_responses",
    }), patch("hermes_cli.models.provider_model_ids", return_value=["gpt-6-sol"]), \
            patch("hermes_cli.models.probe_api_models") as probe:
        result = switch_model(
            "qwen-unknown", current_provider="openai-codex", current_model="gpt-6-sol",
            current_base_url=DEFAULT_CODEX_BASE_URL, current_api_key="test-token",
            explicit_provider="openai-codex",
            user_providers={"openai-codex": {"request_timeout_seconds": 120}},
        )

    assert result.success is False
    probe.assert_not_called()


def test_codex_named_custom_endpoint_still_probes_its_own_listing():
    custom_url = "https://codex-proxy.example.invalid/v1"
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
        "api_key": "test-token", "base_url": custom_url, "api_mode": "codex_responses",
    }), patch("hermes_cli.models.probe_api_models", return_value={
        "models": None, "probed_url": custom_url + "/models",
        "resolved_base_url": custom_url, "suggested_base_url": None, "used_fallback": False,
    }) as probe:
        result = switch_model(
            "gpt-6-sol", current_provider="openai-codex", current_model="gpt-6-astra-900k",
            current_base_url=custom_url, current_api_key="test-token",
            explicit_provider="openai-codex",
            user_providers={"openai-codex": {"base_url": custom_url}},
        )

    assert result.success is False
    assert "custom endpoint's model listing" in result.error_message
    probe.assert_called_once()
