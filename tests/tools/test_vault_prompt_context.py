"""Masked vault prompt routing stays interactive-only; no prompt is displayed here."""
from agent.vault_backends import unlock
from tools import approval_context


def test_prompt_gate_requires_surface_callback_and_attended_turn(monkeypatch):
    monkeypatch.setattr(approval_context, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_is_single_query_approval_context", lambda: False)
    unlock.set_unlock_prompt_callback(lambda *a: "")  # sentinel only; never called
    try:
        assert unlock.can_prompt_here()
        monkeypatch.setattr(approval_context, "_is_single_query_approval_context", lambda: True)
        assert not unlock.can_prompt_here()
        monkeypatch.setattr(approval_context, "_is_single_query_approval_context", lambda: False)
        unlock.set_unlock_prompt_callback(None)
        assert not unlock.can_prompt_here()
    finally:
        unlock.set_unlock_prompt_callback(None)
