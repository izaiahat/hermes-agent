"""Semantic merge regressions: plugin contracts AND upstream loop/session safety."""
import asyncio
import threading

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_context import get_session_env


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["legacy", "event-only", "source-only", "both-async", "kwargs-async", "positional-event"])
async def test_plugin_contract_runs_in_triggering_session_without_loop_blocking(monkeypatch, shape):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner._draining = False
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="isolated-chat", user_id="isolated-user", chat_type="dm")
    event = MessageEvent(text="/owned argument", source=source, message_id="isolated-id")
    loop_thread = threading.get_ident()
    seen = {}

    def capture(args, **context):
        seen.update(context)
        seen["args"] = args
        seen["thread"] = threading.get_ident()
        seen["key"] = get_session_env("HERMES_SESSION_KEY")
        seen["chat"] = get_session_env("HERMES_SESSION_CHAT_ID")
        return "owned-response"

    def legacy(args):
        return capture(args)

    def event_only(args, *, event):
        return capture(args, event=event)

    def source_only(args, *, source):
        return capture(args, source=source)

    async def both_async(args, *, event, source):
        await asyncio.sleep(0)
        return capture(args, event=event, source=source)

    async def kwargs_async(args, **context):
        return capture(args, **context)

    def positional_event(args, trigger):
        return capture(args, event=trigger)

    handlers = {"legacy": legacy, "event-only": event_only, "source-only": source_only,
                "both-async": both_async, "kwargs-async": kwargs_async, "positional-event": positional_event}
    from hermes_cli import plugins
    monkeypatch.setattr(plugins, "get_plugin_command_handler", lambda name: handlers[shape] if name == "owned" else None)
    before = get_session_env("HERMES_SESSION_KEY")
    handled, result, command = await runner._hm_dispatch_quick_and_plugin_commands(event, source, "owned")
    assert (handled, result, command) == (True, "owned-response", "owned")
    assert seen["args"] == "argument"
    assert seen["key"] == runner._session_key_for_source(source)
    assert seen["chat"] == source.chat_id
    assert get_session_env("HERMES_SESSION_KEY") == before
    if shape in ("event-only", "both-async", "kwargs-async", "positional-event"):
        assert seen["event"] is event
    if shape in ("source-only", "both-async", "kwargs-async"):
        assert seen["source"] is source
    if "async" in shape:
        assert seen["thread"] == loop_thread
    else:
        assert seen["thread"] != loop_thread
