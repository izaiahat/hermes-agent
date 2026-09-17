"""Gateway regression for selective async callback suppression."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
import tools.async_delegation as async_delegation


def test_gateway_group_drops_consumed_callback_and_delivers_material_sibling(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(async_delegation, "_db_path", lambda: tmp_path / "state.db")
    session_key = "gateway-exact-once-session"
    artifact = tmp_path / "HANDOFF.md"
    artifact.write_text("done", encoding="utf-8")
    consumed = {
        "type": "async_delegation",
        "delegation_id": "deleg_consumed_gateway",
        "session_key": session_key,
        "status": "completed",
        "dispatched_at": time.time() - 10,
        "results": [{
            "status": "completed",
            "summary": f"Wrote {artifact}",
            "tool_trace": [{
                "tool": "write_file",
                "status": "ok",
                "input_summary": {"targets": {"path": str(artifact)}},
            }],
        }],
    }
    unseen = {
        "type": "async_delegation",
        "delegation_id": "deleg_unseen_gateway",
        "session_key": session_key,
        "status": "completed",
        "dispatched_at": time.time() - 5,
        "results": [{"status": "completed", "summary": "new material result"}],
    }
    async_delegation.record_parent_artifact_access(
        session_keys=[session_key],
        tool_name="read_file",
        args={"path": str(artifact)},
        result='{"content": "1|done", "total_lines": 1, "truncated": false}',
    )

    class _Runner:
        def __init__(self):
            self.deliveries = []

        def _completion_delivery_identity(self, event):
            return None

        async def _deliver_completion_notification(self, text, event):
            self.deliveries.append((text, event))
            return True

    runner = _Runner()
    try:
        result = asyncio.run(
            GatewayRunner._deliver_async_delegation_group(runner, [consumed, unseen])
        )
        assert result is True
        assert len(runner.deliveries) == 1
        assert runner.deliveries[0][1]["delegation_id"] == "deleg_unseen_gateway"
        assert "deleg_consumed_gateway" not in runner.deliveries[0][0]
    finally:
        async_delegation._artifact_observations.pop(session_key, None)


def test_gateway_process_watcher_suppresses_terminal_poll_result(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    import tools.process_registry as process_registry_module

    class _Registry:
        def __init__(self):
            self.sessions = [SimpleNamespace(
                output_buffer="done\n",
                exited=True,
                exit_code=0,
                command="echo done",
            )]

        def get(self, _session_id):
            return self.sessions.pop(0) if self.sessions else None

        def is_completion_consumed(self, _session_id):
            return False

        def is_completion_consumed_or_observed(self, _session_id):
            return True

    async def _instant_sleep(*_args, **_kwargs):
        pass

    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(process_registry_module, "process_registry", _Registry())
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = GatewayRunner(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter
    watcher = {
        "session_id": "proc_polled",
        "check_interval": 0,
        "platform": "telegram",
        "chat_id": "123",
        "notify_on_complete": True,
    }

    asyncio.run(runner._run_process_watcher(watcher))

    adapter.send.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()
