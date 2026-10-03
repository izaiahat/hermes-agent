"""Cron routing changes the selected worker, without touching schedules or job stores."""
from pathlib import Path
from types import SimpleNamespace

import yaml

from cron import scheduler


def test_cron_policy_reaches_constructor_and_preserves_job_pins(tmp_path, monkeypatch):
    home = tmp_path / "root"
    home.mkdir()
    config = {"worker_routing": {"enabled": True}, "model": {"provider": "openai-codex", "default": "gpt-6-astra-900k"},
              "agent": {"reasoning_effort": "max"}, "cron": {"preflight": False}, "platform_toolsets": {"cron": []}}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(scheduler, "_init_cron_mcp_tools", lambda _: None)
    monkeypatch.setattr(scheduler, "_load_credential_pool", lambda *_: None)
    calls = []

    def runtime(**kwargs):
        calls.append(kwargs)
        return {"provider": kwargs["requested"], "model": kwargs["target_model"], "api_mode": "codex_responses",
                "api_key": "offline-only", "base_url": "https://chatgpt.com/backend-api/codex",
                "request_overrides": {"service_tier": "priority", "extra_body": {"service_tier": "priority", "reasoning": {"effort": "low"}}}}

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", runtime)
    for pin, effort, tier in (({}, "medium", "priority"),
                             ({"model": "gpt-6-astra-900k", "reasoning_effort": "high"}, "high", "default")):
        job = {"id": "offline", "name": "offline", "prompt": "check", **pin}
        before = dict(job)
        jc = scheduler._load_cron_job_config(job, job["id"], job["name"])
        setup = scheduler._resolve_cron_agent_setup(job, job["id"], job["name"], jc)
        agent = scheduler._construct_cron_agent(lambda **kwargs: SimpleNamespace(**kwargs), job, config, setup,
                                                workdir=None, session_id="offline", session_db=None)
        assert job == before
        assert calls[-1]["target_model"] == agent.model
        assert agent.reasoning_config["effort"] == effort
        assert agent.request_overrides["service_tier"] == tier
        assert "service_tier" not in (agent.request_overrides.get("extra_body") or {})
        assert "reasoning" not in (agent.request_overrides.get("extra_body") or {})
        assert agent._worker_route["model"] == agent.model
        assert agent.fallback_model is None
