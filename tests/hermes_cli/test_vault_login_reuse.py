"""Reuse an authorized saved login without password ingress or weakened fill binding."""
import argparse
import json


import pytest

from agent.vault_store import VaultError, VaultStore
from hermes_cli import vault


def _source(store, *, identifier="ops@example.test"):
    return store.add_item("login", "Shared affiliate login", {
        "identifier_type": "email", "identifier": identifier,
        "password": "synthetic-shared-login-canary", "otp_secret": "JBSWY3DPEHPK3PXP",
    }, origin="https://source.example")


def test_native_cli_reuses_encrypted_password_without_prompts(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(vault.sys.stdin, "isatty", lambda: False)
    def no_prompt(*args, **kwargs):
        raise AssertionError("saved login must not request credential input")
    monkeypatch.setattr(vault.getpass, "getpass", no_prompt)
    monkeypatch.setattr("builtins.input", no_prompt)
    store = VaultStore(tmp_path / "vault")
    source = _source(store)
    assert source.identifier is not None
    manifest = tmp_path / "origins.json"
    manifest.write_text(json.dumps({"logins": [
        {"origin": "https://portal.example", "label": "New portal"},
        {"origin": "https://portal.example", "label": "Duplicate"},
    ]}))
    parser = argparse.ArgumentParser()
    vault.register_cli(parser)
    args = parser.parse_args(["add-logins", "--manifest", str(manifest),
                             "--identifier", source.identifier, "--reuse-from", source.id])
    vault.vault_command(args)
    target = next(i for i in store.list_items() if i.origin == "https://portal.example")
    assert target.identifier == source.identifier
    assert target.origin == "https://portal.example"
    assert not target.allowed_origins  # normal exact-origin items, no wildcard aliases
    assert not target.has_otp  # a different site's MFA seed must never be cloned
    assert store.resolve_secret(target.id) == {"password": "synthetic-shared-login-canary"}
    source_after = store.get_meta(source.id)
    assert source_after is not None and source_after.has_otp
    vault.vault_command(args)
    assert len(store.list_items()) == 2  # replay is a no-op
    assert "synthetic-shared-login-canary" not in capsys.readouterr().out
    assert b"synthetic-shared-login-canary" not in store._vault_path.read_bytes()


def test_reuse_preserves_existing_login_and_rejects_identity_mismatch(tmp_path):
    store = VaultStore(tmp_path / "vault")
    source = _source(store)
    assert source.identifier is not None
    same = store.add_item("login", "Existing", {
        "identifier_type": "email", "identifier": source.identifier, "password": "existing-canary",
    }, origin="https://portal.example")
    result = store.bind_login(source.id, "https://portal.example", "Portal", identifier=source.identifier)
    assert result.id == same.id
    assert store.resolve_secret(same.id) == {"password": "existing-canary"}
    with pytest.raises(VaultError, match="identifier"):
        store.bind_login(source.id, "https://other.example", "Other", identifier="other@example.test")
    conflict = store.add_item("login", "Other identity", {
        "identifier_type": "email", "identifier": "other@example.test", "password": "other-canary",
    }, origin="https://conflict.example")
    assert conflict.origin is not None
    with pytest.raises(VaultError, match="existing"):
        store.bind_login(source.id, conflict.origin, "Conflict", identifier=source.identifier)
    assert len(store.list_items()) == 3
    assert store.resolve_secret(source.id)["password"] == "synthetic-shared-login-canary"


def test_reuse_requires_local_login_and_exact_https_origin(tmp_path):
    store = VaultStore(tmp_path / "vault")
    source = _source(store)
    assert source.identifier is not None
    for origin in ["http://portal.example", "https://portal.example/login", "https://*.example",
                   "https://user:pass@portal.example", "https://portal.example?return=1"]:
        with pytest.raises(VaultError):
            store.bind_login(source.id, origin, "Portal", identifier=source.identifier)
    card = store.add_item("payment", "Test card", {
        "card_number": "4242424242424242", "exp_month": "01", "exp_year": "2030", "cvc": "123",
    }, origin="https://shop.example")
    for handle in [card.id, "op:external-item", "vault_missing"]:
        with pytest.raises(VaultError):
            store.bind_login(handle, "https://portal.example", "Portal", identifier=source.identifier)
    assert len(store.list_items()) == 2
