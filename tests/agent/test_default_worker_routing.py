"""Selected text auxiliary routes use the native worker policy at the Responses boundary."""
import json
from pathlib import Path

import httpx
import pytest
import yaml
from openai import OpenAI

from agent import auxiliary_client as aux


@pytest.mark.parametrize("task,effort", [("compression", "medium"), ("title_generation", "low")])
def test_auxiliary_policy_matches_actual_sdk_request(tmp_path, monkeypatch, task, effort):
    home = tmp_path / "root"
    home.mkdir()
    config = {"worker_routing": {"enabled": True}, "model": {"provider": "openai-codex", "default": "gpt-6-astra-900k"},
              "auxiliary": {task: {"provider": "auto", "model": "", "timeout": 10}}}
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    requests = []

    def send(request):
        requests.append(json.loads(request.content))
        item = {"type": "message", "id": "m1", "role": "assistant", "content": [{"type": "output_text", "text": '{"title":"Worker routing check"}'}]}
        done = {"type": "response.output_item.done", "item": item}
        completed = {"type": "response.completed", "response": {"id": "offline", "status": "completed", "output": [item], "usage": None}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(done) + "\n\ndata: " + json.dumps(completed) + "\n\n")

    with OpenAI(api_key="offline-only", base_url="https://chatgpt.com/backend-api/codex",
                http_client=httpx.Client(transport=httpx.MockTransport(send))) as client:
        monkeypatch.setattr(aux, "_build_codex_client", lambda model: (aux.CodexAuxiliaryClient(client, model), model))
        runtime = {"provider": "openai-codex", "model": "gpt-6-astra-900k", "api_mode": "codex_responses"}
        for model, override, expected_effort, tier in ((None, None, effort, "priority"),
                                                      ("gpt-6-astra-900k", {"enabled": True, "effort": "high"}, "high", "default")):
            info = {}
            aux.call_llm(task, model=model, main_runtime=runtime, messages=[{"role": "user", "content": "check"}],
                         reasoning_config=override, route_info=info)
            request = requests[-1]
            assert request["model"] == ("gpt-6-astra" if model else "gpt-6.1-sol")
            assert request["reasoning"]["effort"] == info["wire_requests"][-1]["reasoning_effort"] == expected_effort
            assert request["service_tier"] == info["wire_requests"][-1]["service_tier"] == tier
        for rejected in ("none", "minimal", "ultra"):
            before = len(requests)
            with pytest.raises(ValueError):
                aux.call_llm(task, main_runtime=runtime, messages=[{"role": "user", "content": "reject"}],
                             reasoning_config={"enabled": rejected != "none", "effort": rejected})
            assert len(requests) == before
        if task == "title_generation":
            from agent.title_generator import generate_title
            assert generate_title("Check worker routing", main_runtime=runtime) == "Worker routing check"
            assert requests[-1]["reasoning"]["effort"] == "low"
        else:
            import time
            from agent.context_compressor import ContextCompressor
            monkeypatch.setattr(aux, "_select_pool_entry", lambda *_, **__: (True, object()))
            compressor = ContextCompressor(model=runtime["model"], provider=runtime["provider"],
                                           api_mode="codex_responses", base_url=str(client.base_url), quiet_mode=True)
            assert "Worker routing check" in compressor._call_summary_llm("Offline summary", time.monotonic())
            assert compressor._last_aux_resolved_model == "gpt-6.1-sol"
        # An explicit provider override must never inherit the policy's Sol model.
        resolved = aux._resolve_task_provider_model(task, provider="anthropic", main_runtime=runtime)
        assert resolved[:2] == ("anthropic", None)


def test_curator_and_independent_review_preserve_explicit_routes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent import curator, review_engine
    home = tmp_path / "root"
    home.mkdir()
    config = {"worker_routing": {"enabled": True}, "model": {"provider": "openai-codex", "default": "gpt-6-astra-900k"},
              "agent": {"reasoning_effort": "max"}, "auxiliary": {
                  "curator": {"provider": "auto", "model": ""},
                  "review": {"provider": "openai-codex", "model": "gpt-6-astra-900k", "reasoning_effort": "xhigh"}}}
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    seen = []

    def runtime(**kwargs):
        seen.append(kwargs)
        return dict(provider=kwargs["requested"], model=kwargs["target_model"], api_key="offline-only",
                    base_url="https://chatgpt.com/backend-api/codex", api_mode="codex_responses",
                    request_overrides={"service_tier": "priority"})

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", runtime)
    captured = []

    def construct(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(**kwargs, _session_messages=[], run_conversation=lambda **_: {"final_response": "offline"}, close=lambda: None)

    monkeypatch.setattr("run_agent.AIAgent", construct)
    result = curator._run_llm_review("offline")
    assert result["model"] == captured[-1]["model"] == seen[-1]["target_model"] == "gpt-6.1-sol"
    assert captured[-1]["reasoning_config"]["effort"] == "medium"
    assert captured[-1]["request_overrides"]["service_tier"] == "priority"
    reviewer = review_engine._load_review_credentials_cfg()
    assert reviewer["model"] == "gpt-6-astra-900k"
    assert reviewer["reasoning_effort"] == "xhigh" and reviewer["speed"] == "standard"
    assert reviewer["task_kind"] == "independent_review"
    config["auxiliary"]["curator"]["reasoning_effort"] = "ultra"
    config_path.write_text(yaml.safe_dump(config))
    before = len(captured)
    result = curator._run_llm_review("reject")
    assert "Ultra" in result["error"] and len(captured) == before
