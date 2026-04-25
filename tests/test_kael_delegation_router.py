from __future__ import annotations

import importlib.util
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
    assert decision.max_concurrent_children == 5


def test_large_corpus_routes_to_native_codex_subagent() -> None:
    decision = route_task(
        TaskProfile(
            description="Analyze a 50-file codebase and summarize drift",
            task_type="research",
            estimated_tokens=450_000,
            file_count=50,
        )
    )
    assert decision.lane == "codex_native_subagent"
    assert decision.model == "gpt-5.5"
    assert decision.provider == "openai-codex"
    assert decision.toolsets == ["file"]
    assert decision.command is None
    assert "delegate_task" in (decision.spawn_pattern or "")
    assert decision.max_concurrent_children == 2


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
    assert decision.max_concurrent_children == 5


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
    assert decision.max_concurrent_children == 1


def test_parallel_large_subtasks_route_to_native_codex_children() -> None:
    decision = route_task(
        TaskProfile(
            description="Run three clean-room corpus syntheses in parallel",
            task_type="research",
            estimated_tokens=900_000,
            parallel_subtasks=3,
            subtask_estimated_tokens=320_000,
            subtask_file_count=7,
            clean_room=True,
        )
    )
    assert decision.lane == "parallel_fanout"
    assert decision.child_lane == "codex_native_subagent"
    assert decision.child_model == "gpt-5.5"
    assert decision.child_toolsets == ["file"]
    assert decision.child_command is None
    assert decision.max_concurrent_children == 2
