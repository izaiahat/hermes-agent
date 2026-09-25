"""Vault origin reads must not borrow an unrelated or stale browser tab."""
import json
from types import SimpleNamespace

from tools import browser_vault_tool as vault


def test_bound_target_lost_refuses_stale_origin_and_never_prompts(monkeypatch):
    from tools import browser_supervisor
    from agent.vault_backends import unlock

    calls = []
    supervisor = SimpleNamespace(browser_exec_target_id="closed-login-tab",
                                 focus_page=lambda *a, **kw: {"ok": False, "error": "target closed"},
                                 evaluate_runtime=lambda expr: {"ok": True, "result": "https://unrelated.test/login"})
    monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda task: supervisor)
    monkeypatch.setattr(vault, "_ensure_supervisor", lambda task: supervisor)
    unlock.set_unlock_prompt_callback(lambda *a: "")
    unlock.set_save_login_prompt_callback(lambda *a: calls.append("prompt") or None)
    try:
        assert vault._current_page_origin("task") is None
        outcome = json.loads(vault.browser_vault_save_login(task_id="task"))
        assert not outcome["success"]
        assert not calls
    finally:
        unlock.set_unlock_prompt_callback(None)
        unlock.set_save_login_prompt_callback(None)


def test_unbound_save_login_requires_a_login_form_not_the_first_other_tab(monkeypatch):
    from tools import browser_supervisor
    from agent.vault_backends import unlock

    calls = []
    supervisor = SimpleNamespace(browser_exec_target_id=None,
                                 focus_page=lambda origin, **kw: {"ok": False, "error": "no login form"},
                                 evaluate_runtime=lambda expr: {"ok": True, "result": "https://unrelated.test/"})
    monkeypatch.setattr(browser_supervisor.SUPERVISOR_REGISTRY, "get", lambda task: supervisor)
    monkeypatch.setattr(vault, "_ensure_supervisor", lambda task: supervisor)
    unlock.set_unlock_prompt_callback(lambda *a: "")
    unlock.set_save_login_prompt_callback(lambda *a: calls.append("prompt") or None)
    try:
        outcome = json.loads(vault.browser_vault_save_login(task_id="task"))
        assert not outcome["success"]
        assert not calls
    finally:
        unlock.set_unlock_prompt_callback(None)
        unlock.set_save_login_prompt_callback(None)
