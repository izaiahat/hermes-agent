"""Codex pool entries and the auth.json singleton: who may adopt whose tokens.

``manual:device_code`` is ambiguous — a legacy alias of the singleton or an independent account
added with ``hermes auth add openai-codex``. Adopting the singleton into an independent account
silently turned two logins into one (both then hit the same usage limit; salvaged from #100423 and
#106788, cluster #92198 / #95297, issue #106705).
"""
from __future__ import annotations

import base64
import json
import time

import pytest

import hermes_cli.auth as auth_mod
from agent.credential_pool import load_pool


def _jwt(account: str, sub: str, exp: float) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(exp), "sub": sub, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _write_store(home, singleton_tokens: dict, singleton_last_refresh: str, manual: dict) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {"openai-codex": {
            "tokens": singleton_tokens, "last_refresh": singleton_last_refresh, "auth_mode": "chatgpt"}},
        "credential_pool": {"openai-codex": [
            {"id": "seeded", "label": "device_code", "auth_type": "oauth", "priority": 0,
             "source": "device_code", **singleton_tokens, "last_refresh": singleton_last_refresh},
            {"id": "manual", "label": "second", "auth_type": "oauth", "priority": 1,
             "source": "manual:device_code", **manual},
        ]},
    }), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _stub_refresh(monkeypatch, minted: str, minted_rt: str, posted: list) -> None:
    def fake(access_token, refresh_token):
        posted.append(refresh_token)
        return {"access_token": minted, "refresh_token": minted_rt, "last_refresh": _iso(time.time())}

    monkeypatch.setattr(auth_mod, "refresh_codex_oauth_pure", fake)


def test_independent_manual_account_refreshes_with_its_own_pair(home, monkeypatch):
    """An independent second account never adopts the singleton: its refresh POSTs its OWN refresh
    token, and the persisted row still identifies the second principal afterwards."""
    now = time.time()
    a_at = _jwt("acct-A", "user-A", now + 8 * 3600)
    b_at = _jwt("acct-B", "user-B", now + 60)  # expiring → the pool defers it to _refresh_entry()
    _write_store(home, {"access_token": a_at, "refresh_token": "rt-A"}, _iso(now - 3600),
                 {"access_token": b_at, "refresh_token": "rt-B", "last_refresh": _iso(now - 7200)})
    posted: list = []
    b_new = _jwt("acct-B", "user-B", now + 8 * 3600)
    _stub_refresh(monkeypatch, b_new, "rt-B2", posted)

    pool = load_pool("openai-codex")
    refreshed = pool._refresh_entry(next(e for e in pool.entries() if e.id == "manual"), force=False)

    assert posted == ["rt-B"]
    assert refreshed is not None and refreshed.refresh_token == "rt-B2"
    on_disk = json.loads((home / "auth.json").read_text(encoding="utf-8"))
    manual = next(e for e in on_disk["credential_pool"]["openai-codex"] if e["id"] == "manual")
    assert (manual["access_token"], manual["refresh_token"]) == (b_new, "rt-B2")
    # The singleton (account A) is untouched by account B's rotation.
    assert on_disk["providers"]["openai-codex"]["tokens"] == {"access_token": a_at, "refresh_token": "rt-A"}


def test_same_account_alias_adopts_only_a_newer_singleton(home, monkeypatch):
    """A legacy alias (same principal) must follow a singleton that was re-authed AFTER it, but must
    not fall back onto a singleton older than its own rotation — that replays a consumed token."""
    now = time.time()
    alias_at = _jwt("acct-A", "user-A", now + 8 * 3600)
    stale_singleton_at = _jwt("acct-A", "user-A", now + 3600)
    _write_store(home, {"access_token": stale_singleton_at, "refresh_token": "rt-consumed"}, _iso(now - 7200),
                 {"access_token": alias_at, "refresh_token": "rt-alias", "last_refresh": _iso(now - 60)})
    pool = load_pool("openai-codex")
    alias = next(e for e in pool.entries() if e.id == "manual")

    synced = pool._sync_entry_from_auth_store(alias)
    assert (synced.access_token, synced.refresh_token) == (alias_at, "rt-alias")

    # The user re-authenticates the singleton (newer stamp, same principal): the alias follows.
    fresh_at = _jwt("acct-A", "user-A", now + 9 * 3600)
    _write_store(home, {"access_token": fresh_at, "refresh_token": "rt-fresh"}, _iso(now + 5),
                 {"access_token": alias_at, "refresh_token": "rt-alias", "last_refresh": _iso(now - 60)})
    synced = load_pool("openai-codex")._sync_entry_from_auth_store(alias)
    assert (synced.access_token, synced.refresh_token) == (fresh_at, "rt-fresh")


def test_long_lived_codex_pool_adopts_peer_rotation_and_removal(home, monkeypatch):
    now = time.time()
    old = _jwt("acct-B", "user-B", now + 60)
    fresh = _jwt("acct-B", "user-B", now + 8 * 3600)
    _write_store(home, {"access_token": _jwt("acct-A", "user-A", now + 8 * 3600),
                        "refresh_token": "rt-A"}, _iso(now - 3600),
                 {"access_token": old, "refresh_token": "rt-consumed"})
    pool = load_pool("openai-codex")
    stale = next(e for e in pool.entries() if e.id == "manual")
    store = json.loads((home / "auth.json").read_text())
    rows = store["credential_pool"]["openai-codex"]
    rows[1].update(access_token=fresh, refresh_token="rt-rotated", last_status="ok")
    (home / "auth.json").write_text(json.dumps(store))
    posted = []
    _stub_refresh(monkeypatch, fresh, "rt-unwanted", posted)

    adopted = pool._refresh_entry(stale, force=True)
    assert adopted is not None and adopted.access_token == fresh
    assert posted == []
    selected = pool.select()
    assert selected is not None and selected.id == "seeded"
    rows.pop(1)
    (home / "auth.json").write_text(json.dumps(store))
    assert pool._refresh_entry(stale, force=True) is None
    assert [e.id for e in pool.entries()] == ["seeded"]


def test_codex_sync_hydrates_metadata_only_singleton_reference(home):
    now = time.time()
    token = _jwt("acct-A", "user-A", now + 8 * 3600)
    (home / "auth.json").parent.mkdir(parents=True, exist_ok=True)
    store = {"version": 1, "providers": {"openai-codex": {
        "tokens": {"access_token": token, "refresh_token": "fixture-root-refresh"},
    }}, "credential_pool": {"openai-codex": [{
        "id": "reference", "source": "device_code", "auth_type": "oauth",
    }]}}
    (home / "auth.json").write_text(json.dumps(store))
    pool = load_pool("openai-codex")
    for _ in range(3):
        selected = pool.select()
        assert selected is not None and selected.access_token == token
    # Removing the provider AND its persisted row must not resurrect memory.
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": {"openai-codex": []}}))
    assert pool.select() is None
