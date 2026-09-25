"""Origin-bound bulk login enrollment through the native vault CLI."""
import json
from types import SimpleNamespace

from agent.vault_store import VaultStore
from hermes_cli import vault


def _run(tmp_path, monkeypatch, entries, *, dry_run=False):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    manifest = tmp_path / "logins.json"
    manifest.write_text(json.dumps({"logins": entries}))
    vault.vault_command(SimpleNamespace(_vault_handler=vault._cmd_add_logins,
                                       manifest=str(manifest), identifier="ops@example.test",
                                       dry_run=dry_run))
    return VaultStore(home / "vault")


def test_deduped_multi_origin_entries_are_encrypted_and_bound(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vault.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    reads = []
    monkeypatch.setattr(vault.getpass, "getpass", lambda _: reads.append(1) or "synthetic-test-password")
    store = _run(tmp_path, monkeypatch, [
        {"origin": "https://one.example", "label": "One"},
        {"origin": "https://one.example", "label": "One again"},
        {"origin": "https://two.example", "label": "Two"},
    ])
    items = store.list_items()
    assert len(reads) == 1
    assert {(i.origin, i.label, i.kind, i.identifier_type, i.identifier) for i in items} == {
        ("https://one.example", "One", "login", "email", "ops@example.test"),
        ("https://two.example", "Two", "login", "email", "ops@example.test"),
    }
    assert all(store.resolve_secret(i.id) == {"password": "synthetic-test-password"} for i in items)
    assert b"synthetic-test-password" not in (tmp_path / "home/vault/vault.json.enc").read_bytes()
    assert "synthetic-test-password" not in capsys.readouterr().out


def test_existing_login_never_overwritten_and_rerun_is_noop(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    store = VaultStore(home / "vault")
    other = store.add_item("login", "Other", {"identifier_type": "email", "identifier": "other@example.test",
                                               "password": "other-test-password"}, origin="https://one.example")
    monkeypatch.setattr(vault.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    reads = []
    monkeypatch.setattr(vault.getpass, "getpass", lambda _: reads.append(1) or "synthetic-test-password")
    entries = [{"origin": "https://one.example", "label": "One"},
               {"origin": "https://two.example", "label": "Two"}]
    _run(tmp_path, monkeypatch, entries)
    first = [(i.id, i.origin) for i in store.list_items()]
    _run(tmp_path, monkeypatch, entries)
    assert [(i.id, i.origin) for i in store.list_items()] == first
    assert store.resolve_secret(other.id) == {"password": "other-test-password"}
    assert len(reads) == 1


def test_noninteractive_refuses_before_password_or_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vault.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(vault.getpass, "getpass", lambda _: (_ for _ in ()).throw(AssertionError("prompted")))
    store = _run(tmp_path, monkeypatch, [{"origin": "https://one.example", "label": "One"}])
    assert "interactive terminal with hidden input required" in capsys.readouterr().out
    assert store.list_items() == []
