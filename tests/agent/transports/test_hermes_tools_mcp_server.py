"""Tests for the hermes-tools-as-MCP server module surface.

We don't run a live MCP session in unit tests — that requires the codex
subprocess + client + an event loop. These tests pin the static
contract: the module imports, the EXPOSED_TOOLS list is sane, and the
build helper assembles a server when the SDK is present.
"""

from __future__ import annotations

import json
from unittest.mock import patch



class TestModuleSurface:
    def test_module_imports_clean(self):
        from agent.transports import hermes_tools_mcp_server as m
        assert callable(m.main)
        assert callable(m._build_server)
        assert isinstance(m.EXPOSED_TOOLS, tuple)
        assert len(m.EXPOSED_TOOLS) > 0

    def test_exposed_tools_are_safe_subset(self):
        """We MUST NOT expose tools codex already has, because codex'
        own builtins are better-integrated with its sandbox + approvals.
        Specifically: no terminal/shell, no read_file/write_file, no
        patch — those are codex's built-in tools."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        forbidden = {
            "terminal", "shell", "read_file", "write_file", "patch",
            "search_files", "process",
        }
        leaked = forbidden & set(EXPOSED_TOOLS)
        assert not leaked, (
            f"these tools must NOT be exposed via the codex callback "
            f"because codex has built-in equivalents: {leaked}"
        )

    def test_expected_hermes_specific_tools_listed(self):
        """The Hermes-specific tools should be present so users on the
        codex runtime keep access to them."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for required in (
            "web_search",
            "web_extract",
            "browser_navigate",
            "vision_analyze",
            "image_generate",
            "skill_view",
        ):
            assert required in EXPOSED_TOOLS, f"missing {required!r}"

    def test_agent_loop_tools_not_exposed(self):
        """delegate_task / memory / session_search / todo require the
        running AIAgent context to dispatch, so a stateless MCP callback
        can't drive them. They must NOT be in EXPOSED_TOOLS."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for agent_loop_tool in ("delegate_task", "memory", "session_search", "todo"):
            assert agent_loop_tool not in EXPOSED_TOOLS, (
                f"{agent_loop_tool!r} requires the agent loop context "
                "and can't be reached through a stateless MCP callback"
            )

    def test_kanban_worker_tools_exposed(self):
        """Kanban workers run as `hermes chat -q` subprocesses; if they
        come up on the codex_app_server runtime, the worker can do the
        actual work via codex's shell but needs the kanban tools through
        the MCP callback to report back to the kernel. Without these
        tools available, the worker would hang at completion time."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        # Worker handoff tools — every dispatched worker uses at least
        # one of {complete, block, comment} to close out its task.
        for worker_tool in (
            "kanban_complete",
            "kanban_block",
            "kanban_comment",
            "kanban_heartbeat",
        ):
            assert worker_tool in EXPOSED_TOOLS, (
                f"{worker_tool!r} missing from codex callback — kanban "
                "workers on codex_app_server runtime would hang"
            )

    def test_kanban_orchestrator_tools_exposed(self):
        """Orchestrator agents need to dispatch new tasks, query the
        board, and unblock/link tasks. Exposed so an orchestrator on
        codex_app_server can do its job."""
        from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS
        for orch_tool in (
            "kanban_create",
            "kanban_show",
            "kanban_list",
            "kanban_unblock",
            "kanban_link",
        ):
            assert orch_tool in EXPOSED_TOOLS, (
                f"{orch_tool!r} missing from codex callback"
            )


class TestMain:
    def test_main_returns_2_when_mcp_unavailable(self, monkeypatch):
        """When the mcp package isn't installed, main() should exit
        cleanly with code 2 and an install hint, not crash."""
        import agent.transports.hermes_tools_mcp_server as m

        def boom_build(*a, **kw):
            raise ImportError("mcp not installed")

        monkeypatch.setattr(m, "_build_server", boom_build)
        rc = m.main(["--verbose"])
        assert rc == 2

    def test_main_handles_keyboard_interrupt(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class FakeServer:
            def run(self):
                raise KeyboardInterrupt()

        monkeypatch.setattr(m, "_build_server", lambda: FakeServer())
        rc = m.main([])
        assert rc == 0

    def test_main_returns_1_on_runtime_error(self, monkeypatch):
        import agent.transports.hermes_tools_mcp_server as m

        class CrashingServer:
            def run(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(m, "_build_server", lambda: CrashingServer())
        rc = m.main([])
        assert rc == 1


class TestCodexDelegateTaskShim:
    def _setup_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
        monkeypatch.setenv("HERMES_CODEX_DELEGATE_BOARD", "test-codex-delegate")
        monkeypatch.setenv("HERMES_CODEX_DELEGATE_ASSIGNEE", "default")
        monkeypatch.setenv("HERMES_CODEX_DELEGATE_WORKDIR", str(tmp_path / "workspace"))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    def _complete_spawn(self, *, summaries):
        def _spawn(task, workspace, board=None):
            from hermes_cli import kanban_db as kb

            conn = kb.connect(board=board)
            try:
                idx = len(summaries) + 1
                summary = f"child {idx} complete"
                summaries.append((task.id, workspace, board, summary))
                kb.complete_task(
                    conn,
                    task.id,
                    result=f"result for {task.id}",
                    summary=summary,
                    metadata={"total_tokens": idx * 10, "secret": "sk-test_not_returned"},
                )
            finally:
                conn.close()
            return None
        return _spawn

    def test_callback_tool_is_registered_separately_from_registry_tools(self):
        import agent.transports.hermes_tools_mcp_server as m

        assert "codex_delegate_task" in m.CODEX_CALLBACK_TOOLS
        assert "codex_delegate_task" not in m.EXPOSED_TOOLS

        class FakeMCP:
            def __init__(self):
                self.tools = {}

            def add_tool(self, fn, *, name, description):
                self.tools[name] = {"fn": fn, "description": description}

        fake = FakeMCP()
        assert m._register_codex_callback_tools(fake) == 1
        assert "codex_delegate_task" in fake.tools
        assert callable(fake.tools["codex_delegate_task"]["fn"])

    def test_synchronous_three_child_fanout_synthesizes_results(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m

        self._setup_env(monkeypatch, tmp_path)
        summaries = []
        payload = m._codex_delegate_task_impl(
            "Return independent test summaries.",
            max_children=3,
            synchronous=True,
            timeout_seconds=5,
            spawn_fn=self._complete_spawn(summaries=summaries),
            sleep_fn=lambda _seconds: None,
        )

        assert payload["status"] == "success"
        assert payload["max_children"] == 3
        assert len(payload["task_ids"]) == 3
        assert len(payload["children"]) == 3
        assert len(summaries) == 3
        assert all(child["status"] == "done" for child in payload["children"])
        assert "3/3 done" in payload["synthesis"]
        assert all(child.get("run", {}).get("usage", {}).get("total_tokens") for child in payload["children"])

    def test_synchronous_ten_child_fanout_under_load(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m

        self._setup_env(monkeypatch, tmp_path)
        summaries = []
        payload = m._codex_delegate_task_impl(
            "Return ten tiny independent results.",
            max_children=10,
            synchronous=True,
            timeout_seconds=10,
            spawn_fn=self._complete_spawn(summaries=summaries),
            sleep_fn=lambda _seconds: None,
        )

        assert payload["status"] == "success"
        assert len(payload["task_ids"]) == 10
        assert len(payload["children"]) == 10
        assert len(summaries) == 10
        assert "10/10 done" in payload["synthesis"]

    def test_async_mode_returns_durable_task_ids_for_polling(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m
        from hermes_cli import kanban_db as kb

        self._setup_env(monkeypatch, tmp_path)
        spawned = []

        def _spawn(task, workspace, board=None):
            spawned.append((task.id, workspace, board))
            return None

        payload = m._codex_delegate_task_impl(
            "Queue async durable children.",
            max_children=2,
            synchronous=False,
            timeout_seconds=5,
            spawn_fn=_spawn,
            sleep_fn=lambda _seconds: None,
        )

        assert payload["status"] == "queued"
        assert len(payload["task_ids"]) == 2
        assert len(spawned) == 2
        assert payload["board"].startswith("test-codex-delegate-cdt_")

        conn = kb.connect(board=payload["board"])
        try:
            rows = [kb.get_task(conn, task_id) for task_id in payload["task_ids"]]
        finally:
            conn.close()
        assert all(row is not None for row in rows)
        assert {row.status for row in rows if row is not None} == {"running"}

    def test_async_invocations_use_isolated_boards_so_running_children_do_not_starve_next_call(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m

        self._setup_env(monkeypatch, tmp_path)
        spawned = []

        def _spawn(task, workspace, board=None):
            spawned.append((task.id, board))
            return None

        first = m._codex_delegate_task_impl(
            "First async fanout leaves two children running.",
            max_children=2,
            synchronous=False,
            timeout_seconds=5,
            spawn_fn=_spawn,
            sleep_fn=lambda _seconds: None,
        )
        second = m._codex_delegate_task_impl(
            "Second async fanout must still spawn.",
            max_children=1,
            synchronous=False,
            timeout_seconds=5,
            spawn_fn=_spawn,
            sleep_fn=lambda _seconds: None,
        )

        assert first["status"] == "queued"
        assert second["status"] == "queued"
        assert first["board"] != second["board"]
        assert len(spawned) == 3
        assert spawned[-1][0] == second["task_ids"][0]

    def test_child_spawn_crash_is_recovered_as_error_payload(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m

        self._setup_env(monkeypatch, tmp_path)

        def _crash(_task, _workspace, board=None):
            raise RuntimeError("synthetic child crash")

        payload = m._codex_delegate_task_impl(
            "This child should crash.",
            max_children=1,
            synchronous=True,
            timeout_seconds=5,
            spawn_fn=_crash,
            sleep_fn=lambda _seconds: None,
        )

        assert payload["status"] == "error"
        assert payload["children"][0]["status"] == "blocked"
        assert "synthetic child crash" in json.dumps(payload)

    def test_timeout_blocks_unfinished_children(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m

        self._setup_env(monkeypatch, tmp_path)
        monotonic_values = iter([0.0, 0.0, 1.0])

        def _monotonic():
            return next(monotonic_values, 1.0)

        payload = m._codex_delegate_task_impl(
            "This child should time out.",
            max_children=1,
            synchronous=True,
            timeout_seconds=0.05,
            spawn_fn=lambda _task, _workspace, board=None: None,
            sleep_fn=lambda _seconds: None,
            monotonic_fn=_monotonic,
        )

        assert payload["status"] == "timeout"
        assert payload["children"][0]["status"] == "blocked"
        assert "timeout" in payload["error"]

    def test_nonspawnable_assignee_fails_fast(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m

        self._setup_env(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_CODEX_DELEGATE_ASSIGNEE", "missing-profile-for-test")

        payload = m._codex_delegate_task_impl(
            "This child cannot spawn because the assignee is not a profile.",
            max_children=1,
            synchronous=True,
            timeout_seconds=30,
            spawn_fn=lambda _task, _workspace, board=None: None,
            sleep_fn=lambda _seconds: None,
        )

        assert payload["status"] == "error"
        assert "non-spawnable profile" in payload["error"]
        assert payload["children"][0]["status"] == "blocked"

    def test_redacted_json_stays_parseable_when_values_contain_auth_header(self):
        import agent.transports.hermes_tools_mcp_server as m

        raw = m._redacted_json({"x": "Authorization: Bearer eyJabcdefghijklmnop", "ok": True})
        parsed = json.loads(raw)

        assert parsed["ok"] is True
        assert "eyJabcdefghijklmnop" not in raw

    def test_usage_metadata_is_bounded_and_redacted(self):
        import agent.transports.hermes_tools_mcp_server as m

        usage = m._extract_usage_metadata(
            {
                "total_tokens": 42,
                "usage_details": {
                    "api_key": "sk-test_abcdefghijklmnop1234567890",
                    "blob": "x" * 2000,
                },
                "secret": "sk-test_should_not_be_returned",
            }
        )

        assert usage["total_tokens"] == 42
        assert "secret" not in usage
        raw = json.dumps(usage)
        assert "sk-test_abcdefghijklmnop1234567890" not in raw
        assert len(usage["usage_details"]) <= 1050

    def test_sealed_boundary_redacts_secrets_from_public_tool_result(self, monkeypatch, tmp_path):
        import agent.transports.hermes_tools_mcp_server as m

        self._setup_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "hermes_cli.plugins.get_pre_tool_call_block_message",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            m,
            "_apply_codex_delegate_hooks",
            lambda args, result, duration_ms: result,
        )
        monkeypatch.setattr(
            m,
            "_codex_delegate_task_impl",
            lambda *args, **kwargs: {
                "tool": "codex_delegate_task",
                "status": "success",
                "children": [
                    {
                        "status": "done",
                        "result": "OPENAI_API_KEY=sk-test_abcdefghijklmnop1234567890 Authorization: Bearer eyJabcdefghijklmnop",
                    }
                ],
            },
        )

        raw = m.codex_delegate_task("redact", max_children=1, synchronous=True, timeout_seconds=1)
        parsed = json.loads(raw)

        assert parsed["status"] == "success"
        assert "sk-test_abcdefghijklmnop1234567890" not in raw
        assert "eyJabcdefghijklmnop" not in raw
        assert "OPENAI_API_KEY=" in raw
        assert "***" in raw
