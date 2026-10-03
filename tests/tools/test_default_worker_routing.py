"""Default worker policy: explicit overrides survive resolution and real transport assembly."""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from openai import OpenAI

from tools import delegate_tool as delegate
from tools.delegate_tool_child_run import _route_receipt


@pytest.mark.parametrize("kind,expected", [
    ("mechanical", "low"), ("simple_check", "low"),
    ("implementation", "medium"), ("research", "medium"), ("synthesis", "medium"), ("routine_review", "medium"),
    ("complex_integration", "high"), ("debugging", "high"), ("independent_review", "high"),
    ("difficult", "xhigh"), ("unresolved", "xhigh"), ("safety", "max"),
])
def test_effort_policy_overrides_and_refusals(tmp_path, monkeypatch, kind, expected):
    from tools.delegate_tool_config import resolve_worker_route
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from agent.secret_scope import set_multiplex_active, set_secret_scope, reset_secret_scope
    root = tmp_path / "root"
    secondary = root / "profiles" / "other"
    secondary.mkdir(parents=True)
    config = {"worker_routing": {"enabled": True}, "model": {"provider": "openai-codex"}}
    for home in (root, secondary):
        (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    set_multiplex_active(True)
    secret_token = set_secret_scope({})
    try:
        for home in (root, secondary, root):
            token = set_hermes_home_override(home)
            try:
                route = resolve_worker_route({"task_kind": kind}, inherited_provider="openai-codex")
                if home == secondary:
                    assert route is None
                    continue
                assert route["reasoning_effort"] == expected
                assert route["model"] == "gpt-6.1-sol" and route["speed"] == "fast"
                for explicit in ("low", "medium", "high", "xhigh", "max"):
                    pinned = resolve_worker_route({"task_kind": kind, "reasoning_effort": explicit}, inherited_provider="openai-codex")
                    assert pinned["reasoning_effort"] == explicit
                for rejected in ("none", "minimal", "ultra", "unknown"):
                    with pytest.raises(ValueError):
                        resolve_worker_route({"reasoning_effort": rejected}, inherited_provider="openai-codex")
                assert resolve_worker_route({"provider": "anthropic"}, config=config) is None
                assert resolve_worker_route({}, inherited_provider="openai-codex", config={}) is None
            finally:
                reset_hermes_home_override(token)
    finally:
        reset_secret_scope(secret_token)
        set_multiplex_active(False)


@pytest.mark.parametrize("speed_override", [None, "standard"])
@pytest.mark.parametrize("effort_override", [None, "max"])
def test_native_task_selection_matches_actual_sdk_wire(tmp_path, monkeypatch, speed_override, effort_override):
    from agent.codex_runtime import run_codex_stream
    from tools.delegate_tool_config import _resolve_delegation_credentials
    home = tmp_path / "root"
    home.mkdir()
    cfg = {"worker_routing": {"enabled": True}, "model": {"provider": "openai-codex"},
           "delegation": {"provider": "openai-codex", "max_spawn_depth": 1}}
    (home / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    resolutions = []

    def runtime(requested, target_model):
        resolutions.append((requested, target_model))
        return dict(provider=requested, model=target_model, api_key="offline-only", api_mode="codex_responses",
                    base_url="https://chatgpt.com/backend-api/codex", request_overrides={})

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", runtime)
    parent = SimpleNamespace(model="gpt-6-astra-900k", provider="openai-codex", requested_provider="openai-codex",
                             api_mode="codex_responses", api_key="offline-only", base_url="https://chatgpt.com/backend-api/codex",
                             reasoning_config={"enabled": True, "effort": "max"}, request_overrides={"service_tier": "priority"},
                             enabled_toolsets=[], disabled_toolsets=[], _active_children=[], session_id=None)
    routes = cfg["delegation"]
    creds = _resolve_delegation_credentials(routes, parent)
    tasks = [{"goal": "check", "task_kind": "mechanical"},
             {"goal": "review", "model": "gpt-6-astra-900k", "reasoning_effort": "high", "task_kind": "independent_review"}]
    if effort_override:
        tasks[0]["reasoning_effort"] = effort_override
        tasks[0]["model"] = "openai-codex/gpt-6.1-sol"
    if speed_override:
        tasks[0]["speed"] = speed_override
    # A generic default tier cannot force the explicit Astra exception onto priority.
    routes = {**routes, "request_overrides": {"service_tier": "priority", "extra_body": {"service_tier": "priority"}}}
    delegate._resolve_task_routes(tasks, routes, parent, creds)
    assert tasks[0]["_resolved_route"]["model"] != parent.model
    assert resolutions[-2:] == [("openai-codex", tasks[0].get("model") or "gpt-6.1-sol"), ("openai-codex", "gpt-6-astra-900k")]
    built, err = delegate._build_children(tasks, [None, None], creds, top_role="leaf", max_iterations=2,
                                         parent_agent=parent, routing_cfg=routes, live_deleg_id=None, live_writers=[])
    assert err is None
    requests = []

    def send(request):
        requests.append(json.loads(request.content))
        completed = {"type": "response.completed", "response": {"id": "offline", "status": "completed", "output": [], "usage": None}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="data: " + json.dumps(completed) + "\n\n")

    try:
        with OpenAI(api_key="offline-only", base_url=parent.base_url, http_client=httpx.Client(transport=httpx.MockTransport(send))) as client:
            for (_, task, child), expected_effort, expected_tier in zip(built, (effort_override or "low", "high"), ("default" if speed_override else "priority", "default")):
                assert child._delegate_role == "leaf"
                kwargs = child._build_api_kwargs([{"role": "user", "content": task["goal"]}])
                final = run_codex_stream(child, kwargs, client=client)
                assert final.status == "completed"
                actual = requests[-1]
                receipt = _route_receipt(child)
                assert actual["reasoning"]["effort"] == receipt["effective"]["reasoning_effort"] == expected_effort
                assert actual["service_tier"] == receipt["wire_requests"][-1]["service_tier"] == expected_tier
                assert actual["model"] == receipt["wire_requests"][-1]["model"]
                assert receipt["requested"]["model"] == task.get("model")
                assert "offline-only" not in json.dumps(receipt)
                before = len(requests)
                with pytest.raises(ValueError):
                    run_codex_stream(child, {**kwargs, "extra_body": {"service_tier": "default" if expected_tier == "priority" else "priority"}}, client=client)
                assert len(requests) == before
        for effort in ("minimal", "ultra"):
            with pytest.raises(ValueError):
                delegate._resolve_task_routes([{"goal": "reject", "reasoning_effort": effort}], routes, parent, creds)
    finally:
        delegate._release_partial_children(parent, built)
