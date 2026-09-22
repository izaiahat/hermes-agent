"""Tests for per-channel model and system prompt overrides (Fixes #1955)."""

from unittest.mock import patch

import pytest

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import _get_channel_override, GatewayRunner
from gateway.session import SessionSource


class TestGetChannelOverride:

    def test_parsed_discord_default_sets_medium_without_changing_other_routes(self):
        from types import SimpleNamespace
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={
            Platform.DISCORD: PlatformConfig.from_dict({
                "channel_overrides": {"*": {"model": "gpt-6-sol", "reasoning_effort": "medium"}},
            }),
        })
        discord = SessionSource(platform=Platform.DISCORD, chat_id="new-channel", user_id="u")
        telegram = SessionSource(platform=Platform.TELEGRAM, chat_id="new-channel", user_id="u")
        with patch.object(runner, "_resolve_session_key_or_none", return_value="session"), \
             patch.object(runner, "_peek_session_state", return_value=None) as state, \
             patch.object(runner, "_load_reasoning_config", return_value={"enabled": True, "effort": "high"}):
            assert runner._resolve_session_reasoning_config(source=discord) == {"enabled": True, "effort": "medium"}
            assert runner._resolve_session_reasoning_config(source=telegram)["effort"] == "high"
            assert runner._resolve_session_reasoning_config()["effort"] == "high"
            state.return_value = SimpleNamespace(conversation=SimpleNamespace(reasoning_override={"enabled": True, "effort": "low"}))
            assert runner._resolve_session_reasoning_config(source=discord)["effort"] == "low"

    def test_platform_wildcard_is_last_fallback_without_cross_platform_leakage(self):
        default = ChannelOverride(model="discord-default")
        exact = ChannelOverride(model="exact-model")
        thread = ChannelOverride(model="thread-model")
        parent = ChannelOverride(model="parent-model")
        config = GatewayConfig(platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, channel_overrides={
                "*": default, "chat": exact, "thread": thread, "parent": parent,
            }),
            Platform.TELEGRAM: PlatformConfig(enabled=True),
        })
        assert _get_channel_override(config, Platform.DISCORD, "future-channel") is default
        assert _get_channel_override(
            config, Platform.DISCORD, "chat", thread_id="thread", parent_id="parent"
        ) is exact
        assert _get_channel_override(
            config, Platform.DISCORD, "new-thread", thread_id="thread", parent_id="parent"
        ) is thread
        assert _get_channel_override(
            config, Platform.DISCORD, "new-thread", parent_id="parent"
        ) is parent
        assert _get_channel_override(config, Platform.TELEGRAM, "future-channel") is None
        assert _get_channel_override(config, Platform.TELEGRAM, "chat") is None
        assert _get_channel_override(config, Platform.SLACK, "future-channel") is None


    def test_no_override_when_channel_not_in_overrides(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "999": ChannelOverride(model="openrouter/healer-alpha"),
                    },
                ),
            },
        )
        assert _get_channel_override(config, Platform.DISCORD, "123") is None

    def test_returns_override_when_channel_matches(self):
        ov = ChannelOverride(
            model="openrouter/healer-alpha",
            provider="openrouter",
            system_prompt="You are a summarizer.",
        )
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={"1234567890": ov},
                ),
            },
        )
        result = _get_channel_override(config, Platform.DISCORD, "1234567890")
        assert result is not None
        assert result.model == "openrouter/healer-alpha"
        assert result.provider == "openrouter"
        assert result.system_prompt == "You are a summarizer."


    def test_thread_id_lookup_when_chat_id_misses(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "thread_99": ChannelOverride(model="topic-model"),
                    },
                ),
            },
        )
        result = _get_channel_override(
            config, Platform.DISCORD, "parent_chan", thread_id="thread_99"
        )
        assert result is not None
        assert result.model == "topic-model"


class TestResolveModelForChannel:
    def test_uses_channel_override_when_present(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(model="anthropic/claude-opus-4.6"),
                    },
                ),
            },
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        model = runner._resolve_model_for_channel(Platform.DISCORD, "chan_1")
        assert model == "anthropic/claude-opus-4.6"


class TestGetSystemPromptForChannel:
    def test_uses_channel_override_when_present(self):
        config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(system_prompt="You are a coding assistant."),
                    },
                ),
            },
        )
        runner = object.__new__(GatewayRunner)
        runner.config = config
        runner._ephemeral_system_prompt = "Global prompt"
        prompt = runner._get_system_prompt_for_channel(Platform.DISCORD, "chan_1")
        assert prompt == "You are a coding assistant."


class TestResolveSessionAgentRuntimePriority:
    """Model/runtime priority: session /model → channel_overrides → global."""

    def test_channel_override_beats_global(self):
        runner = object.__new__(GatewayRunner)
        runner._session_model_overrides = {}
        runner.config = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    channel_overrides={
                        "chan_1": ChannelOverride(
                            model="channel/model",
                            provider="openrouter",
                        ),
                    },
                ),
            },
        )
        source = SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan_1",
            user_id="u1",
        )
        with patch("gateway.run._resolve_gateway_model", return_value="global/model"), \
             patch("gateway.run._resolve_runtime_agent_kwargs", return_value={
                 "provider": "anthropic",
                 "api_key": "k",
                 "base_url": "https://api.anthropic.com",
                 "api_mode": "chat_completions",
             }), \
             patch(
                 "gateway.run._resolve_runtime_agent_kwargs_for_provider",
                 return_value={
                     "provider": "openrouter",
                     "api_key": "k2",
                     "base_url": "https://openrouter.ai/api/v1",
                     "api_mode": "chat_completions",
                 },
             ):
            model, runtime = runner._resolve_session_agent_runtime(
                source=source,
                user_config={"model": {"default": "global/model"}},
            )
        assert model == "channel/model"
        assert runtime["provider"] == "openrouter"


