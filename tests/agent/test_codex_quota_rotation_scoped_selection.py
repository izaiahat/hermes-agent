"""Codex quota rotation must not inherit unrelated model cooldowns."""

import time
from types import SimpleNamespace

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import CredentialPool, PooledCredential
from agent.error_classifier import FailoverReason


def test_exhausted_codex_entry_rotates_to_healthy_entry_with_unrelated_model_cooldown():
    depleted = PooledCredential(
        provider="openai-codex", id="depleted", label="depleted", auth_type="oauth",
        priority=0, source="manual:device_code", access_token="old-token",
        last_status="exhausted", last_status_at=time.time(), last_error_code=429,
        last_error_reset_at=time.time() + 3600,
    )
    healthy = PooledCredential(
        provider="openai-codex", id="healthy", label="healthy", auth_type="oauth",
        priority=1, source="manual:device_code", access_token="usable-token",
        model_cooldowns={"another-model": time.time() + 3600},
    )
    pool = CredentialPool("openai-codex", [depleted, healthy])

    def swap(entry):
        agent.api_key = entry.runtime_api_key
        agent._credential_pool_entry_id = entry.id
        return True

    agent = SimpleNamespace(
        _credential_pool=pool, provider="openai-codex", base_url="", model="gpt-6-sol",
        api_key="old-token", _credential_pool_entry_id="depleted", _swap_credential=swap,
        _is_entitlement_failure=lambda *_: False,
    )
    # Mirror a gateway/TUI turn: a 429 must move the active request key
    # to the healthy credential even when that account has another model benched.
    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=False,
        classified_reason=FailoverReason.rate_limit,
        error_context={"reason": "usage_limit_reached", "message": "The usage limit has been reached"},
    )
    assert recovered is True
    assert agent.api_key == "usable-token"
    assert agent._credential_pool_entry_id == "healthy"
