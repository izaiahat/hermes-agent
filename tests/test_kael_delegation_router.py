from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "kael_delegation_router.py"
SPEC = importlib.util.spec_from_file_location("kael_delegation_router", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

TaskProfile = MODULE.TaskProfile
route_task = MODULE.route_task
run_validation = MODULE.run_validation
render_examples = MODULE.render_examples
integration_status = MODULE.integration_status


def test_shared_state_routes_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="Update STATE.md after reviewing child reports",
            shared_state_write=True,
            estimated_tokens=30_000,
        )
    )
    assert decision.lane == "parent"
    assert decision.model == "gpt-5.4"
    assert decision.max_concurrent_children == 1


def test_single_tool_call_routes_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="Check one file and answer yes/no",
            single_tool_call=True,
            estimated_tokens=10_000,
        )
    )
    assert decision.lane == "parent"
    assert decision.max_concurrent_children == 1


def test_pure_reasoning_under_50k_routes_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="Reason about a short policy question",
            pure_reasoning=True,
            estimated_tokens=50_000,
        )
    )
    assert decision.lane == "parent"
    assert decision.max_concurrent_children == 1


def test_bounded_code_review_routes_to_gpt55() -> None:
    decision = route_task(
        TaskProfile(
            description="Review one module diff and return JSON findings",
            task_type="code",
            estimated_tokens=120_000,
            structured_output=True,
            code_review=True,
        )
    )
    assert decision.lane == "gpt55_specialist"
    assert decision.model == "gpt-5.5"
    assert decision.toolsets == ["file", "terminal"]
    assert decision.max_concurrent_children == 10


def test_exactly_200k_structured_still_routes_to_gpt55() -> None:
    decision = route_task(
        TaskProfile(
            description="Return structured markdown from a bounded research task",
            task_type="research",
            estimated_tokens=200_000,
            structured_output=True,
        )
    )
    assert decision.lane == "gpt55_specialist"
    assert decision.max_concurrent_children == 10


def test_exactly_300k_without_other_triggers_falls_back_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="A medium-large but non-decomposed inspection task",
            task_type="inspection",
            estimated_tokens=300_000,
            file_count=4,
        )
    )
    assert decision.lane == "parent"


def test_exactly_five_files_routes_to_native_codex_subagent() -> None:
    decision = route_task(
        TaskProfile(
            description="Analyze five files as one corpus",
            task_type="research",
            estimated_tokens=220_000,
            file_count=5,
        )
    )
    assert decision.lane == "codex_native_subagent"
    assert decision.model == "gpt-5.5"
    assert decision.provider == "openai-codex"
    assert decision.toolsets == ["file"]
    assert decision.max_concurrent_children == 5


def test_multifile_over_live_codex_ceiling_routes_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="Analyze six files as one oversized corpus",
            task_type="research",
            estimated_tokens=350_000,
            file_count=6,
        )
    )
    assert decision.lane == "parent"


def test_parallel_small_subtasks_route_to_gpt55_children() -> None:
    decision = route_task(
        TaskProfile(
            description="Audit 6 small files independently for style issues",
            task_type="inspection",
            estimated_tokens=80_000,
            parallel_subtasks=6,
            subtask_estimated_tokens=12_000,
            subtask_file_count=1,
        )
    )
    assert decision.lane == "parallel_fanout"
    assert decision.child_lane == "gpt55_specialist"
    assert decision.child_toolsets == ["file"]
    assert decision.max_concurrent_children == 10


def test_parallel_mixed_subtask_sizes_routes_to_native_codex_children() -> None:
    decision = route_task(
        TaskProfile(
            description="Run mixed-size parallel audits",
            task_type="research",
            estimated_tokens=150_000,
            parallel_subtasks=3,
            subtask_token_samples=(12_000, 220_000, 9_000),
            subtask_file_samples=(1, 7, 1),
        )
    )
    assert decision.lane == "parallel_fanout"
    assert decision.child_lane == "codex_native_subagent"
    assert decision.child_model == "gpt-5.5"
    assert decision.child_toolsets == ["file"]
    assert decision.max_concurrent_children == 5


def test_parallel_subtask_over_live_codex_ceiling_routes_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="Run mixed-size parallel audits with one oversized subtask",
            task_type="research",
            estimated_tokens=150_000,
            parallel_subtasks=3,
            subtask_token_samples=(12_000, 350_000, 9_000),
            subtask_file_samples=(1, 7, 1),
        )
    )
    assert decision.lane == "parent"


def test_shared_state_plus_parallel_still_routes_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="Update shared truth after many audits",
            task_type="inspection",
            estimated_tokens=250_000,
            parallel_subtasks=8,
            shared_state_write=True,
        )
    )
    assert decision.lane == "parent"


def test_multi_domain_routes_to_gpt54_orchestrator() -> None:
    decision = route_task(
        TaskProfile(
            description="Compare config, git workflow, and doctrine docs, then decide final policy",
            task_type="research",
            estimated_tokens=180_000,
            broad_domains=3,
        )
    )
    assert decision.lane == "gpt54_orchestrator_cli"
    assert decision.model == "gpt-5.4"
    assert decision.role == "orchestrator"
    assert decision.max_concurrent_children == 2


def test_fallback_path_routes_to_parent() -> None:
    decision = route_task(
        TaskProfile(
            description="General ambiguous task with moderate scope",
            task_type="inspection",
            estimated_tokens=250_000,
            file_count=2,
            structured_output=False,
            pure_reasoning=False,
            parallel_subtasks=0,
            broad_domains=0,
        )
    )
    assert decision.lane == "parent"


def test_invalid_task_type_raises() -> None:
    try:
        TaskProfile(description="bad", task_type="finance")
    except ValueError as exc:
        assert "Invalid task_type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid task_type")


def test_negative_counts_raise() -> None:
    try:
        TaskProfile(description="bad", estimated_tokens=-1)
    except ValueError as exc:
        assert "must be non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative tokens")


def test_examples_render_clean_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--examples"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 5
    assert payload[0]["decision"]["lane"] == "gpt55_specialist"


def test_validate_flag_passes_and_returns_clean_json() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--validate"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["case_count"] == 5


def test_integration_status_documents_doctrine_artifact_state() -> None:
    status = integration_status()
    assert status["status"] == "doctrine_encoding_test_artifact"
    assert status["runtime_invoked_by_core"] is False
    assert "Further runtime integration" in status["implication"]


def test_render_examples_matches_validation_examples() -> None:
    rendered = render_examples()
    assert len(rendered) == 5
    assert rendered[1]["decision"]["lane"] == "codex_native_subagent"


def test_run_validation_function_passes() -> None:
    result = run_validation()
    assert result["ok"] is True
    assert len(result["results"]) == 5
