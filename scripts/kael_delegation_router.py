#!/usr/bin/env python3
"""Production-grade hardcoded Kael delegation router.

This script encodes Kael's routing doctrine in a deterministic form so it can be:
- unit tested
- inspected from the CLI
- used as a doctrine/regression artifact beside the live skill/config/docs

Important: this script is currently a doctrine-encoding artifact, not a runtime
hook automatically invoked by Kael inside Hermes core. See `integration_status()`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

CONCURRENCY = {
    "gpt55_max_concurrent_children": 10,
    "native_codex_max_concurrent_children": 5,
    "gpt54_orchestrator_max_concurrent_children": 2,
    "max_spawn_depth": 4,
}

FAILURE_RECOVERY = {
    "gpt55_timeout": "retry once with the same prompt, then mark partial and continue",
    "native_codex_failure": (
        "stop new Codex spawns, capture the Cloudflare / rate-limit error, and "
        "fall back to safe gpt-5.5 leaves unless the operator explicitly requests "
        "a Codex retry"
    ),
    "delegate_task_failure": "fall back to terminal+hermes chat shell-out",
    "bad_child_output": "do not trust silently; add a follow-up lane and annotate the gap",
}

PARENT = {
    "lane": "parent",
    "model": "gpt-5.4",
    "provider": "openai-codex",
    "role": "parent",
    "max_context_tokens": 1_000_000,
    "retains": [
        "STATE.md writes",
        "final synthesis",
        "ship/no-ship judgments",
        "irreversible actions",
    ],
    "spawn_pattern": "stay in the current Kael parent session",
}

GPT55_SPECIALIST = {
    "lane": "gpt55_specialist",
    "model": "gpt-5.5",
    "provider": "openai-codex",
    "role": "leaf",
    "max_context_tokens": 272_000,
    "timeout_seconds": 600,
    "default_toolsets": {
        "code": ["file", "terminal"],
        "research": ["file", "web"],
        "inspection": ["file"],
    },
    "spawn_pattern": "delegate_task(goal=..., context=..., toolsets=...)",
}

CODEX_NATIVE = {
    "lane": "codex_native_subagent",
    "model": "gpt-5.5",
    "provider": "openai-codex",
    "role": "leaf",
    "api_mode": "codex_responses",
    "max_context_tokens": 1_000_000,
    "timeout_seconds": 600,
    "default_toolsets": {
        "read": ["file"],
        "write": ["file", "terminal"],
    },
    "spawn_pattern": (
        "delegate_task(goal=..., context=..., toolsets=['file']) using Hermes native "
        "openai-codex/codex_responses path"
    ),
    "anti_pattern": "never use raw codex exec / raw Node subprocesses",
}

GPT54_ORCHESTRATOR = {
    "lane": "gpt54_orchestrator_cli",
    "model": "gpt-5.4",
    "provider": "openai-codex",
    "role": "orchestrator",
    "max_context_tokens": 1_000_000,
    "timeout_seconds": 600,
    "command": (
        "hermes chat --provider openai-codex --model gpt-5.4 "
        "-s delegation-routing-v2 -Q -q '<self-contained orchestrator prompt>'"
    ),
    "spawn_pattern": "terminal(command='hermes chat ...')",
}

_ALLOWED_TASK_TYPES = frozenset({"code", "research", "inspection"})


@dataclass(slots=True)
class TaskProfile:
    """Normalized task facts used by the hardcoded router.

    The profile is intentionally compact and declarative. It captures the
    features of a task that matter to routing, not every execution detail.
    """

    description: str
    task_type: str = "inspection"
    estimated_tokens: int = 0
    subtask_estimated_tokens: int = 0
    file_count: int = 0
    subtask_file_count: int = 0
    shared_state_write: bool = False
    irreversible_action: bool = False
    single_tool_call: bool = False
    pure_reasoning: bool = False
    structured_output: bool = False
    code_review: bool = False
    clean_room: bool = False
    parallel_subtasks: int = 0
    broad_domains: int = 0
    write_allowed: bool = False
    json_mode: bool = False
    subtask_token_samples: tuple[int, ...] = field(default_factory=tuple)
    subtask_file_samples: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.task_type not in _ALLOWED_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{self.task_type}'. Allowed: {sorted(_ALLOWED_TASK_TYPES)}"
            )
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")

        for name in (
            "estimated_tokens",
            "subtask_estimated_tokens",
            "file_count",
            "subtask_file_count",
            "parallel_subtasks",
            "broad_domains",
        ):
            _validate_non_negative_int(name, getattr(self, name))

        self.subtask_token_samples = _normalize_int_samples(
            "subtask_token_samples", self.subtask_token_samples
        )
        self.subtask_file_samples = _normalize_int_samples(
            "subtask_file_samples", self.subtask_file_samples
        )


@dataclass(slots=True)
class RoutingDecision:
    """Structured router output suitable for tests and CLI JSON rendering."""

    lane: str
    reasons: list[str]
    model: str | None = None
    provider: str | None = None
    role: str | None = None
    toolsets: list[str] | None = None
    command: str | None = None
    spawn_pattern: str | None = None
    timeout_seconds: int | None = None
    failure_recovery: dict[str, str] | None = None
    child_lane: str | None = None
    child_model: str | None = None
    child_toolsets: list[str] | None = None
    child_command: str | None = None
    parallel_subtasks: int | None = None
    max_concurrent_children: int | None = None
    max_spawn_depth: int | None = None


@dataclass(slots=True)
class ValidationCase:
    """Expected outcome for a representative routing regression case."""

    name: str
    profile: TaskProfile
    expected_lane: str
    expected_max_concurrent_children: int
    expected_child_lane: str | None = None
    expected_model: str | None = None


def _validate_non_negative_int(name: str, value: Any) -> None:
    """Reject non-integers and negative counts/tokens."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _normalize_int_samples(name: str, values: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a sequence of non-negative integer samples."""
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of ints")
    normalized: list[int] = []
    for idx, value in enumerate(values):
        _validate_non_negative_int(f"{name}[{idx}]", value)
        normalized.append(value)
    return tuple(normalized)


def specialist_toolsets(task_type: str) -> list[str]:
    """Return default leaf toolsets for the given bounded-task type."""
    return GPT55_SPECIALIST["default_toolsets"].get(task_type, ["file"])


def codex_toolsets(profile: TaskProfile) -> list[str]:
    """Return the native-Codex toolset mix for the task profile."""
    if profile.write_allowed:
        return CODEX_NATIVE["default_toolsets"]["write"]
    return CODEX_NATIVE["default_toolsets"]["read"]


def apply_global_limits(
    decision: RoutingDecision, *, max_concurrent_children: int
) -> RoutingDecision:
    """Attach the current hard caps to a routing decision."""
    decision.max_concurrent_children = max_concurrent_children
    decision.max_spawn_depth = CONCURRENCY["max_spawn_depth"]
    return decision


def make_parent(reasons: list[str]) -> RoutingDecision:
    """Build a decision that keeps work in the Kael parent."""
    return apply_global_limits(
        RoutingDecision(
            lane=PARENT["lane"],
            reasons=reasons,
            model=PARENT["model"],
            provider=PARENT["provider"],
            role=PARENT["role"],
            spawn_pattern=PARENT["spawn_pattern"],
        ),
        max_concurrent_children=1,
    )


def make_gpt55(task_type: str, reasons: list[str]) -> RoutingDecision:
    """Build a bounded specialist-leaf decision."""
    return apply_global_limits(
        RoutingDecision(
            lane=GPT55_SPECIALIST["lane"],
            reasons=reasons,
            model=GPT55_SPECIALIST["model"],
            provider=GPT55_SPECIALIST["provider"],
            role=GPT55_SPECIALIST["role"],
            toolsets=specialist_toolsets(task_type),
            spawn_pattern=GPT55_SPECIALIST["spawn_pattern"],
            timeout_seconds=GPT55_SPECIALIST["timeout_seconds"],
            failure_recovery={
                "timeout": FAILURE_RECOVERY["gpt55_timeout"],
                "bad_output": FAILURE_RECOVERY["bad_child_output"],
            },
        ),
        max_concurrent_children=CONCURRENCY["gpt55_max_concurrent_children"],
    )


def make_codex(profile: TaskProfile, reasons: list[str]) -> RoutingDecision:
    """Build a native-Codex child decision for large-context work."""
    return apply_global_limits(
        RoutingDecision(
            lane=CODEX_NATIVE["lane"],
            reasons=reasons,
            model=CODEX_NATIVE["model"],
            provider=CODEX_NATIVE["provider"],
            role=CODEX_NATIVE["role"],
            toolsets=codex_toolsets(profile),
            spawn_pattern=CODEX_NATIVE["spawn_pattern"],
            timeout_seconds=CODEX_NATIVE["timeout_seconds"],
            failure_recovery={
                "failure": FAILURE_RECOVERY["native_codex_failure"],
                "bad_output": FAILURE_RECOVERY["bad_child_output"],
            },
        ),
        max_concurrent_children=CONCURRENCY["native_codex_max_concurrent_children"],
    )


def make_gpt54_orchestrator(reasons: list[str]) -> RoutingDecision:
    """Build a native Hermes CLI orchestrator-child decision."""
    return apply_global_limits(
        RoutingDecision(
            lane=GPT54_ORCHESTRATOR["lane"],
            reasons=reasons,
            model=GPT54_ORCHESTRATOR["model"],
            provider=GPT54_ORCHESTRATOR["provider"],
            role=GPT54_ORCHESTRATOR["role"],
            command=GPT54_ORCHESTRATOR["command"],
            spawn_pattern=GPT54_ORCHESTRATOR["spawn_pattern"],
            timeout_seconds=GPT54_ORCHESTRATOR["timeout_seconds"],
            failure_recovery={
                "delegate_task_failure": FAILURE_RECOVERY["delegate_task_failure"],
                "bad_output": FAILURE_RECOVERY["bad_child_output"],
            },
        ),
        max_concurrent_children=CONCURRENCY["gpt54_orchestrator_max_concurrent_children"],
    )


def _largest_subtask_tokens(profile: TaskProfile) -> int:
    """Return the largest known subtask token estimate.

    If explicit sample data exists, it wins. Otherwise fall back to the legacy
    single subtask estimate or the overall estimate.
    """
    if profile.subtask_token_samples:
        return max(profile.subtask_token_samples)
    if profile.subtask_estimated_tokens:
        return profile.subtask_estimated_tokens
    return profile.estimated_tokens


def _largest_subtask_file_count(profile: TaskProfile) -> int:
    """Return the largest known subtask file-count estimate."""
    if profile.subtask_file_samples:
        return max(profile.subtask_file_samples)
    if profile.subtask_file_count:
        return profile.subtask_file_count
    return profile.file_count


def choose_parallel_child(
    profile: TaskProfile,
) -> tuple[str, str | None, list[str] | None, str | None]:
    """Choose the child lane for a parallel-fanout task.

    This uses the *largest* known subtask size, so mixed-size bursts route to a
    safe lane if any one subtask needs the heavier native-Codex treatment.
    """
    subtask_tokens = _largest_subtask_tokens(profile)
    subtask_files = _largest_subtask_file_count(profile)
    if subtask_files >= 5 or subtask_tokens > 300_000 or profile.clean_room:
        return (
            CODEX_NATIVE["lane"],
            CODEX_NATIVE["model"],
            codex_toolsets(profile),
            None,
        )
    return (
        GPT55_SPECIALIST["lane"],
        GPT55_SPECIALIST["model"],
        specialist_toolsets(profile.task_type),
        None,
    )


def route_task(profile: TaskProfile) -> RoutingDecision:
    """Route a normalized task profile to the hardcoded Kael lane.

    Precedence is intentional:
    1. Parent-only safety / irreversibility
    2. Tiny local work
    3. Broad-domain orchestrator topology
    4. Parallel topology
    5. Large-corpus native Codex lane
    6. Bounded gpt-5.5 specialist
    7. Parent fallback
    """
    if profile.shared_state_write or profile.irreversible_action:
        return make_parent([
            "A: shared-state or irreversible action stays in parent",
        ])

    if profile.single_tool_call:
        return make_parent([
            "B: single tool call stays in parent",
        ])

    if profile.pure_reasoning and profile.estimated_tokens <= 50_000:
        return make_parent([
            "B: pure reasoning under 50k stays in parent",
        ])

    if profile.broad_domains >= 2:
        return make_gpt54_orchestrator([
            "F: multiple broad domains need local synthesis",
        ])

    if profile.parallel_subtasks >= 2:
        child_lane, child_model, child_toolsets, child_command = choose_parallel_child(
            profile
        )
        max_children = (
            CONCURRENCY["native_codex_max_concurrent_children"]
            if child_lane == CODEX_NATIVE["lane"]
            else CONCURRENCY["gpt55_max_concurrent_children"]
        )
        return apply_global_limits(
            RoutingDecision(
                lane="parallel_fanout",
                reasons=[
                    "E: task splits into independent parallel subtasks",
                    f"child lane selected from largest subtask: {child_lane}",
                ],
                spawn_pattern="spawn N children immediately and supervise in parallel",
                timeout_seconds=(
                    GPT55_SPECIALIST["timeout_seconds"]
                    if child_lane == GPT55_SPECIALIST["lane"]
                    else CODEX_NATIVE["timeout_seconds"]
                ),
                failure_recovery={
                    "gpt55_timeout": FAILURE_RECOVERY["gpt55_timeout"],
                    "native_codex_failure": FAILURE_RECOVERY["native_codex_failure"],
                    "bad_output": FAILURE_RECOVERY["bad_child_output"],
                },
                child_lane=child_lane,
                child_model=child_model,
                child_toolsets=child_toolsets,
                child_command=child_command,
                parallel_subtasks=profile.parallel_subtasks,
            ),
            max_concurrent_children=max_children,
        )

    if (
        profile.file_count >= 5
        or profile.estimated_tokens > 300_000
        or profile.clean_room
    ):
        return make_codex(
            profile,
            [
                "D: 5+ files, >300k tokens, or clean-room perspective routes to Hermes native Codex subagent"
            ],
        )

    bounded_structured = profile.estimated_tokens <= 200_000 and (
        profile.structured_output
        or profile.code_review
        or profile.task_type in {"code", "research", "inspection"}
    )
    if bounded_structured:
        return make_gpt55(
            profile.task_type,
            [
                "C: bounded reasoning / structured output under 200k routes to gpt-5.5 specialist"
            ],
        )

    return make_parent([
        "Fallback: task did not justify delegation overhead",
    ])


def example_profiles() -> list[TaskProfile]:
    """Return the five canonical example profiles shown in docs/reports."""
    return [
        TaskProfile(
            description="Review one module diff and return JSON findings",
            task_type="code",
            estimated_tokens=120_000,
            structured_output=True,
            code_review=True,
            json_mode=True,
        ),
        TaskProfile(
            description="Read 6 docs and summarize system drift",
            task_type="research",
            estimated_tokens=350_000,
            file_count=6,
        ),
        TaskProfile(
            description="Audit 6 small files independently for style issues",
            task_type="inspection",
            estimated_tokens=80_000,
            parallel_subtasks=6,
            subtask_estimated_tokens=12_000,
            subtask_file_count=1,
        ),
        TaskProfile(
            description="Update STATE.md after reviewing child reports",
            task_type="inspection",
            estimated_tokens=30_000,
            shared_state_write=True,
        ),
        TaskProfile(
            description="Compare config, git workflow, and doctrine docs, then decide final policy",
            task_type="research",
            estimated_tokens=180_000,
            broad_domains=3,
        ),
    ]


def validation_cases() -> list[ValidationCase]:
    """Return regression cases used by `--validate` and unit tests."""
    return [
        ValidationCase(
            name="bounded_json_code_review",
            profile=example_profiles()[0],
            expected_lane="gpt55_specialist",
            expected_max_concurrent_children=10,
            expected_model="gpt-5.5",
        ),
        ValidationCase(
            name="large_corpus_native_codex",
            profile=example_profiles()[1],
            expected_lane="codex_native_subagent",
            expected_max_concurrent_children=5,
            expected_model="gpt-5.5",
        ),
        ValidationCase(
            name="parallel_small_file_audits",
            profile=example_profiles()[2],
            expected_lane="parallel_fanout",
            expected_child_lane="gpt55_specialist",
            expected_max_concurrent_children=10,
        ),
        ValidationCase(
            name="shared_truth_parent",
            profile=example_profiles()[3],
            expected_lane="parent",
            expected_max_concurrent_children=1,
            expected_model="gpt-5.4",
        ),
        ValidationCase(
            name="multi_domain_orchestrator",
            profile=example_profiles()[4],
            expected_lane="gpt54_orchestrator_cli",
            expected_max_concurrent_children=2,
            expected_model="gpt-5.4",
        ),
    ]


def run_validation() -> dict[str, Any]:
    """Run regression validation over the canonical example cases."""
    cases = validation_cases()
    results: list[dict[str, Any]] = []
    ok = True
    for case in cases:
        decision = route_task(case.profile)
        mismatches: list[str] = []
        if decision.lane != case.expected_lane:
            mismatches.append(f"lane={decision.lane} expected={case.expected_lane}")
        if decision.max_concurrent_children != case.expected_max_concurrent_children:
            mismatches.append(
                "max_concurrent_children="
                f"{decision.max_concurrent_children} expected={case.expected_max_concurrent_children}"
            )
        if case.expected_child_lane is not None and decision.child_lane != case.expected_child_lane:
            mismatches.append(
                f"child_lane={decision.child_lane} expected={case.expected_child_lane}"
            )
        if case.expected_model is not None and decision.model != case.expected_model:
            mismatches.append(f"model={decision.model} expected={case.expected_model}")
        passed = not mismatches
        ok = ok and passed
        results.append(
            {
                "name": case.name,
                "passed": passed,
                "mismatches": mismatches,
                "decision": asdict(decision),
            }
        )
    return {
        "ok": ok,
        "case_count": len(cases),
        "results": results,
    }


def integration_status() -> dict[str, Any]:
    """Describe how this router currently relates to live Kael execution.

    Current assessment: doctrine-encoding test artifact with config/doc coupling,
    but not auto-invoked by Hermes core at runtime.
    """
    return {
        "status": "doctrine_encoding_test_artifact",
        "runtime_invoked_by_core": False,
        "references": [
            "~/.hermes/skills/delegation-routing-v2/SKILL.md",
            "~/.hermes/config.yaml::kael_delegation",
            "/home/ubuntu/business/reports/delegation-system-hardcoded-2026-04-25.md",
            "tests/test_kael_delegation_router.py",
        ],
        "implication": (
            "The script is authoritative as doctrine/regression logic, but Hermes core does "
            "not currently import and execute it automatically inside the parent agent loop. "
            "Further runtime integration would require explicit product/code changes."
        ),
    }


def render_examples() -> list[dict[str, Any]]:
    """Return the example profiles annotated with routing decisions."""
    return [
        {"task": profile.description, "decision": asdict(route_task(profile))}
        for profile in example_profiles()
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for direct router inspection/validation."""
    parser = argparse.ArgumentParser(description="Hardcoded Kael delegation router")
    parser.add_argument("description", nargs="?", default="", help="Task description")
    parser.add_argument(
        "--task-type",
        choices=sorted(_ALLOWED_TASK_TYPES),
        default="inspection",
    )
    parser.add_argument("--estimated-tokens", type=int, default=0)
    parser.add_argument("--subtask-estimated-tokens", type=int, default=0)
    parser.add_argument("--file-count", type=int, default=0)
    parser.add_argument("--subtask-file-count", type=int, default=0)
    parser.add_argument("--shared-state-write", action="store_true")
    parser.add_argument("--irreversible-action", action="store_true")
    parser.add_argument("--single-tool-call", action="store_true")
    parser.add_argument("--pure-reasoning", action="store_true")
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--code-review", action="store_true")
    parser.add_argument("--clean-room", action="store_true")
    parser.add_argument("--parallel-subtasks", type=int, default=0)
    parser.add_argument("--broad-domains", type=int, default=0)
    parser.add_argument("--write-allowed", action="store_true")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--subtask-token-samples", nargs="*", type=int, default=())
    parser.add_argument("--subtask-file-samples", nargs="*", type=int, default=())
    parser.add_argument("--examples", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--integration-status", action="store_true")
    return parser.parse_args()


def build_profile_from_args(args: argparse.Namespace) -> TaskProfile:
    """Build a validated `TaskProfile` from CLI args."""
    return TaskProfile(
        description=args.description,
        task_type=args.task_type,
        estimated_tokens=args.estimated_tokens,
        subtask_estimated_tokens=args.subtask_estimated_tokens,
        file_count=args.file_count,
        subtask_file_count=args.subtask_file_count,
        shared_state_write=args.shared_state_write,
        irreversible_action=args.irreversible_action,
        single_tool_call=args.single_tool_call,
        pure_reasoning=args.pure_reasoning,
        structured_output=args.structured_output,
        code_review=args.code_review,
        clean_room=args.clean_room,
        parallel_subtasks=args.parallel_subtasks,
        broad_domains=args.broad_domains,
        write_allowed=args.write_allowed,
        json_mode=args.json_mode,
        subtask_token_samples=tuple(args.subtask_token_samples),
        subtask_file_samples=tuple(args.subtask_file_samples),
    )


def main() -> None:
    """CLI entrypoint.

    - `--examples` prints clean example JSON.
    - `--validate` prints regression JSON and exits non-zero on failure.
    - `--integration-status` prints current integration assessment.
    - default mode prints one routed task as JSON.
    """
    args = parse_args()

    if args.examples:
        print(json.dumps(render_examples(), indent=2))
        return

    if args.validate:
        result = run_validation()
        print(json.dumps(result, indent=2))
        if not result["ok"]:
            raise SystemExit(1)
        return

    if args.integration_status:
        print(json.dumps(integration_status(), indent=2))
        return

    profile = build_profile_from_args(args)
    print(
        json.dumps(
            {
                "profile": asdict(profile),
                "decision": asdict(route_task(profile)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
