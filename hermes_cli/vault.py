"""``hermes vault`` — manage the local encrypted autofill vault.

Subcommands:
- ``hermes vault add``   interactive wizard; the password is read via
  getpass (never echoed, never accepted as argv). The login identifier is
  visible metadata and prompted normally.
- ``hermes vault list``  metadata — labels, kinds, identifiers, origins,
  handles. Passwords are never shown.
- ``hermes vault rm``    remove an item by handle/id.

The vault backs the password-blind browser autofill tools
(``browser_vault_list`` / ``browser_vault_fill``): the agent sees handles
and login identifiers, types the identifier itself, and fills the password
server-side without ever seeing it.
"""

from __future__ import annotations

import getpass
import json
import sys
import warnings
from pathlib import Path
from urllib.parse import urlsplit


def _console():
    from rich.console import Console

    return Console()


def _cmd_add(args) -> None:
    from agent.vault_store import (
        LOGIN_IDENTIFIER_TYPES,
        VAULT_KINDS,
        VaultError,
        get_vault_store,
    )

    c = _console()
    c.print(
        "[bold]Add a vault item[/] (the password is encrypted at rest and the "
        "agent never sees it; the identifier is visible metadata the agent "
        "can type itself)"
    )

    kind = (args.kind or "").strip().lower()
    while kind not in VAULT_KINDS:
        kind = input(f"Kind ({'/'.join(VAULT_KINDS)}) [login]: ").strip().lower() or "login"
        if kind not in VAULT_KINDS:
            c.print(f"[red]Unknown kind {kind!r}[/]")
            kind = ""

    label = ""
    while not label:
        label = input("Label (e.g. 'GitHub work account'): ").strip()

    try:
        if kind == "login":
            origin = ""
            while not origin:
                origin = input("Site origin (e.g. https://github.com): ").strip()
            id_type = ""
            while id_type not in LOGIN_IDENTIFIER_TYPES:
                id_type = (
                    input(f"Identifier type ({'/'.join(LOGIN_IDENTIFIER_TYPES)}) [email]: ")
                    .strip()
                    .lower()
                    or "email"
                )
            identifier = ""
            while not identifier:
                identifier = input(f"{id_type.capitalize()}: ").strip()
            password = ""
            while not password:
                password = getpass.getpass("Password (hidden): ")
            otp_secret = getpass.getpass(
                "Authenticator key (optional, hidden; the 2FA \"setup key\" or otpauth:// link — Enter to skip): ")
            # identifier_type/identifier are stored as metadata (not secret);
            # add_item moves them out of the encrypted payload.
            secret = {
                "identifier_type": id_type,
                "identifier": identifier,
                "password": password,
                **({"otp_secret": otp_secret} if otp_secret.strip() else {}),
            }
            meta = get_vault_store().add_item(
                kind="login", label=label, secret=secret, origin=origin
            )
        else:
            from agent.vault_store import ADDRESS_FIELDS, PAYMENT_FIELDS, REQUIRED_FIELDS

            fields = PAYMENT_FIELDS if kind == "payment" else ADDRESS_FIELDS
            origin = ""
            while not origin:
                origin = input("Site origin the item may be filled on (e.g. https://shop.example.com): ").strip()
            c.print(f"[dim]{kind} fields are filled only on that origin; card values are read hidden.[/]")
            secret = {}
            for field in fields:
                required = field in REQUIRED_FIELDS[kind]
                prompt = f"{field.replace('_', ' ')}{'' if required else ' (optional)'}: "
                read = getpass.getpass if kind == "payment" else input
                value = read(prompt).strip()
                while required and not value:
                    value = read(prompt).strip()
                if value:
                    secret[field] = value
            meta = get_vault_store().add_item(kind=kind, label=label, secret=secret, origin=origin)
    except VaultError as exc:
        c.print(f"[red]Error:[/] {exc}")
        return

    c.print(f"[green]Stored.[/] handle=[bold]{meta.id}[/] kind={meta.kind} origin={meta.origin or '-'}")


def _cmd_add_logins(args) -> None:
    """Preview exact origin bindings, then reuse a saved login or collect one hidden password."""
    from agent.vault_store import VaultError, get_vault_store, normalize_origin

    c = _console()
    identifier = args.identifier.strip()
    if not identifier or "@" not in identifier:
        raise VaultError("--identifier must be an email address")
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VaultError("cannot read a JSON login manifest") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"logins"} or not isinstance(manifest["logins"], list):
        raise VaultError("manifest must contain only a logins array")
    logins = {}
    for entry in manifest["logins"]:
        if not isinstance(entry, dict) or set(entry) != {"origin", "label"}:
            raise VaultError("each login must contain only origin and label")
        origin, label = entry["origin"], entry["label"]
        if not isinstance(origin, str) or not isinstance(label, str) or not label.strip():
            raise VaultError("each login needs a string origin and nonempty label")
        parts = urlsplit(origin)
        if (parts.scheme != "https" or parts.username or parts.password or parts.path or parts.query
                or parts.fragment or normalize_origin(origin) != origin):
            raise VaultError("login origins must be exact canonical HTTPS origins without paths or credentials")
        logins.setdefault(origin, label.strip())
    if not logins:
        raise VaultError("manifest has no logins")

    store = get_vault_store()
    reuse_from = getattr(args, "reuse_from", None)
    if reuse_from:
        source = store.get_meta(reuse_from)
        if source is None or source.kind != "login":
            raise VaultError("--reuse-from requires an existing local login handle")
        if source.identifier_type != "email" or source.identifier != identifier:
            raise VaultError("--reuse-from login identifier must match --identifier")
    existing = {}
    for item in store.list_items():
        if item.kind == "login":
            existing.setdefault(item.origin, set()).add(item.identifier)
    pending = []
    for origin, label in logins.items():
        identities = existing.get(origin, set())
        status = "skip (existing identity)" if identifier in identities else (
            "skip (different identity; preserve existing)" if identities else "add")
        c.print(f"{origin}  {label}  {status}")
        if not identities:
            pending.append((origin, label))
    c.print(f"{len(logins)} distinct origins; {len(pending)} to add; {len(logins) - len(pending)} skipped.")
    if args.dry_run or not pending:
        return
    if reuse_from:
        for origin, label in pending:
            meta = store.bind_login(reuse_from, origin, label, identifier=identifier)
            c.print(f"Bound {meta.origin}  handle={meta.id}")
        return
    if not sys.stdin.isatty():
        raise VaultError("interactive terminal with hidden input required; nothing saved")
    try:
        answer = input(f"Save {len(pending)} origin-bound logins for {identifier}? Type yes to confirm: ")
    except (EOFError, KeyboardInterrupt) as exc:
        raise VaultError("confirmation cancelled; nothing saved") from exc
    if answer.strip().lower() != "yes":
        c.print("Cancelled; nothing saved.")
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            password = getpass.getpass("Password (hidden, once): ")
    except (getpass.GetPassWarning, EOFError, KeyboardInterrupt, OSError) as exc:
        raise VaultError("hidden password input unavailable; nothing saved") from exc
    if not password:
        raise VaultError("empty password; nothing saved")
    for origin, label in pending:
        # Recheck before each write; never replace an identity already present on the origin.
        if any(item.kind == "login" and item.origin == origin for item in store.list_items()):
            c.print(f"Skipped {origin}: existing login")
            continue
        store.add_item("login", label, {"identifier_type": "email", "identifier": identifier,
                                        "password": password}, origin=origin)
        c.print(f"Stored {origin}")


def _cmd_list(args) -> None:
    """Local items always; external managers only for the lifetime of this CLI process (a
    `hermes vault list` unlock does not carry into a chat session — unlock there when asked)."""
    from agent.vault_backends import enabled_backends

    c = _console()
    rows, locked = [], []
    for backend in enabled_backends():
        if backend.needs_unlock and not backend.is_unlocked():
            locked.append(backend.display_name)
            continue
        rows.extend((backend.display_name, meta) for meta in backend.list_items())
    if not rows and not locked:
        c.print("[dim]Vault is empty. Add an item with `hermes vault add`.[/]")
        return
    if rows:
        from rich.table import Table

        table = Table(title=f"Vault items ({len(rows)})")
        for col in ("Handle", "Source", "Kind", "Label", "Identifier", "Origin"):
            table.add_column(col, style="bold" if col == "Handle" else None)
        for source, meta in rows:
            table.add_row(meta.id, source, meta.kind, meta.label, meta.identifier or "-", meta.origin or "-")
        c.print(table)
        c.print("[dim]Passwords are never shown; the agent fills them server-side from the handle.[/]")
    for name in locked:
        c.print(f"[yellow]{name}[/] is enabled but locked — the agent will ask you to unlock it when it needs a login.")


def _cmd_sources(args) -> None:
    """Show the detected password managers; `--disable`/`--enable` flip the opt-out (`vault.<name>.enabled`)."""
    from agent.vault_backends import enabled_backends
    from agent.vault_backends.base import external_backend_classes, is_installed
    from hermes_cli.config import _ensure_dict, load_config, save_config

    c = _console()
    classes = {cls.name: cls for cls in external_backend_classes()}
    if args.enable or args.disable:
        name = args.enable or args.disable
        if name not in classes:
            c.print(f"[red]Unknown password manager {name!r}[/] (expected one of {', '.join(classes)})")
            return
        cfg = load_config()
        section = _ensure_dict(_ensure_dict(cfg, "vault"), name)
        if args.enable:
            section.pop("enabled", None)  # detected managers are on by default; drop the opt-out
        else:
            section["enabled"] = False
        save_config(cfg)
        c.print(f"[green]{classes[name].display_name} {'on' if args.enable else 'off'}[/] for browser logins.")
        return
    enabled = {b.name for b in enabled_backends()}
    for name, cls in classes.items():
        if name in enabled:
            status = "[green]detected[/] · the agent asks you to unlock it when it needs a login"
        elif is_installed(name):
            status = "[dim]turned off[/] (`hermes vault sources --enable {name}` to use it)".format(name=name)
        else:
            status = "[dim]not installed[/]"
        c.print(f"  {cls.display_name:<10} {status}")
    c.print("[dim]Managers are picked up automatically when their CLI is installed and signed in.[/]")


def _cmd_rm(args) -> None:
    from agent.vault_store import get_vault_store

    c = _console()
    if get_vault_store().remove_item(args.handle):
        c.print(f"[green]Removed[/] {args.handle}")
    else:
        c.print(f"[red]No vault item with handle {args.handle!r}[/]")


def register_cli(subparser) -> None:
    """Build the ``hermes vault`` argparse tree (called from main.py)."""
    subs = subparser.add_subparsers(dest="vault_action")

    p_add = subs.add_parser(
        "add",
        help="Save a login, card or address ahead of time (optional: the agent asks you on the page when it needs one)",
    )
    p_add.add_argument(
        "--kind", choices=["login", "payment", "address"], default=None,
        help="Item kind (interactive prompt when omitted)",
    )
    p_add.set_defaults(_vault_handler=_cmd_add)

    p_bulk = subs.add_parser("add-logins", help="Preview and add origin-bound logins with one hidden password")
    p_bulk.add_argument("--manifest", required=True, help="JSON file with non-secret logins: origin and label only")
    p_bulk.add_argument("--identifier", required=True, help="Email identifier stored as visible metadata")
    p_bulk.add_argument("--reuse-from", metavar="HANDLE",
                        help="Reuse a saved local login password for explicitly authorized origins, without prompting; never copies MFA")
    p_bulk.add_argument("--dry-run", action="store_true", help="Preview exact deduped origins without a password or writes")
    p_bulk.set_defaults(_vault_handler=_cmd_add_logins)

    p_list = subs.add_parser("list", help="List vault items (metadata only, never values)")
    p_list.set_defaults(_vault_handler=_cmd_list)

    p_rm = subs.add_parser("rm", help="Remove a vault item by handle")
    p_rm.add_argument("handle", help="Item handle (see `hermes vault list`)")
    p_rm.set_defaults(_vault_handler=_cmd_rm)

    p_src = subs.add_parser("sources", help="Show detected password managers (1Password, Bitwarden); they are on automatically")
    group = p_src.add_mutually_exclusive_group()
    group.add_argument("--disable", metavar="NAME", help="Stop using a detected manager: onepassword | bitwarden")
    group.add_argument("--enable", metavar="NAME", help="Undo --disable")
    p_src.set_defaults(_vault_handler=_cmd_sources)


def vault_command(args) -> None:
    from agent.vault_store import VaultError

    handler = getattr(args, "_vault_handler", None)
    try:
        if handler is None:
            _cmd_list(args)
            return
        handler(args)
    except VaultError as exc:
        _console().print(f"[red]Error:[/] {exc}")
