from __future__ import annotations

from argparse import Namespace

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.main import _load_chat_query_file


def test_chat_parser_accepts_query_file_and_keeps_query_off_argv(tmp_path):
    prompt = "α" * 70_000
    path = tmp_path / "prompt.txt"
    path.write_text(prompt, encoding="utf-8")
    parser, _subparsers, _chat = build_top_level_parser()

    args = parser.parse_args(["chat", "--cli", "--query-file", str(path)])

    assert args.query is None
    assert args.query_file == str(path)
    _load_chat_query_file(args)
    assert args.query == prompt


def test_chat_parser_rejects_query_and_query_file_together(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("prompt", encoding="utf-8")
    parser, _subparsers, _chat = build_top_level_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["chat", "-q", "inline", "--query-file", str(path)])

    assert exc.value.code == 2


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"", "file is empty"),
        (b"has\x00nul", "NUL byte"),
        (b"\xff", "utf-8"),
    ],
)
def test_query_file_rejects_invalid_payloads(tmp_path, capsys, payload, expected):
    path = tmp_path / "prompt.bin"
    path.write_bytes(payload)
    args = Namespace(query_file=str(path), query=None)

    with pytest.raises(SystemExit) as exc:
        _load_chat_query_file(args)

    assert exc.value.code == 2
    assert expected.lower() in capsys.readouterr().err.lower()


def test_query_file_rejects_symlink(tmp_path, capsys):
    target = tmp_path / "target.txt"
    target.write_text("prompt", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    with pytest.raises(SystemExit) as exc:
        _load_chat_query_file(Namespace(query_file=str(link), query=None))

    assert exc.value.code == 2
    assert "non-symlink" in capsys.readouterr().err


def test_query_file_rejects_more_than_eight_mib(tmp_path, capsys):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (8 * 1024 * 1024 + 1))

    with pytest.raises(SystemExit) as exc:
        _load_chat_query_file(Namespace(query_file=str(path), query=None))

    assert exc.value.code == 2
    assert "8 MiB" in capsys.readouterr().err
