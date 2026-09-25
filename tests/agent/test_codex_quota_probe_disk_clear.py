"""A positive live quota probe must survive the cross-process disk merge."""
import time
from typing import Any

from agent.credential_pool import load_pool
from hermes_cli import auth as auth_mod


def test_verified_early_recovery_persists_only_the_reset_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    rows: list[dict[str, Any]] = [
        dict(id=ident, label=ident, auth_type="oauth", source="manual:device_code",
             priority=i, access_token=f"token-{i}", last_status="exhausted",
             last_status_at=now - 60, last_error_code=429,
             last_error_reason="usage_limit_reached", last_error_reset_at=now + 86400,
             model_cooldowns={"unrelated-model": now + 86400} if i == 1 else None)
        for i, ident in enumerate(("still-exhausted", "reset-account"))
    ]
    auth_mod.write_credential_pool("openai-codex", rows)
    pool = load_pool("openai-codex")
    monkeypatch.setattr(pool, "_codex_quota_restored_upstream", lambda entry: entry.id == "reset-account")

    chosen = pool.select(model="gpt-6-astra-900k")
    assert chosen is not None and chosen.id == "reset-account"
    persisted = {row["id"]: row for row in auth_mod.read_credential_pool("openai-codex")}
    assert persisted["reset-account"]["last_status"] == "ok"
    assert persisted["reset-account"]["status_cleared_at"] > now - 60
    assert persisted["reset-account"]["model_cooldowns"]["unrelated-model"] == rows[1]["model_cooldowns"]["unrelated-model"]
    assert persisted["still-exhausted"]["last_status"] == "exhausted"
    assert persisted["still-exhausted"]["last_error_reset_at"] == rows[0]["last_error_reset_at"]


def test_newer_concurrent_429_wins_over_verified_early_clear():
    now = time.time()
    recovered = dict(id="reset-account", last_status="ok", status_cleared_at=now)
    newer_429 = dict(id="reset-account", last_status="exhausted", last_status_at=now + 1,
                     last_error_code=429, last_error_reset_at=now + 3600)
    assert auth_mod._merge_disk_cooldown_state(recovered, newer_429, "openai-codex")["last_status"] == "exhausted"
    older_429 = {**newer_429, "last_status_at": now - 1}
    assert auth_mod._merge_disk_cooldown_state(recovered, older_429, "openai-codex")["last_status"] == "ok"
