"""Hermes-tools-as-MCP server for the codex_app_server runtime.

When the user runs `openai/*` turns through the codex app-server, codex
owns the loop and builds its own tool list. By default, that means
Hermes' richer tool surface — web search, browser automation,
delegate_task subagents, vision analysis, persistent memory, skills,
cross-session search, image generation, TTS — is unreachable.

This module exposes a curated subset of those Hermes tools to the
spawned codex subprocess via stdio MCP. Codex registers it as a normal
MCP server (per `~/.codex/config.toml [mcp_servers.hermes-tools]`) and
the user gets full Hermes capability inside a Codex turn.

Scope (what we expose):
  - web_search, web_extract              — Firecrawl, no codex equivalent
  - browser_navigate / _click / _type /  — Camofox/Browserbase automation
    _snapshot / _scroll / _back / _press /
    _get_images / _console / _vision
  - vision_analyze                       — image inspection by vision model
  - image_generate                       — image generation
  - skill_view, skills_list              — Hermes' skill library
  - text_to_speech                       — TTS
  - kanban_* (complete/block/comment/    — kanban worker + orchestrator
    heartbeat/show/list/create/            handoff (stateless: read env var,
    unblock/link)                          write ~/.hermes/kanban.db)

What we DO NOT expose:
  - terminal / shell                     — codex's own shell tool
  - read_file / write_file / patch       — codex's apply_patch + shell
  - search_files / process               — codex's shell
  - clarify                              — codex's own UX
  - native delegate_task / memory /      — `_AGENT_LOOP_TOOLS` in Hermes
    session_search / todo                  (model_tools.py). They require
                                           the running AIAgent context to
                                           dispatch (mid-loop state), so a
                                           stateless MCP callback can't
                                           drive them. See the inline
                                           comment on EXPOSED_TOOLS below.
  - codex_delegate_task is a bespoke      — Kanban-backed Codex shim, not
    callback tool                          the native in-process delegate_task.

Run with: python -m agent.transports.hermes_tools_mcp_server
Spawned by: CodexAppServerSession.ensure_started() when the runtime is
            active and config opts in.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Tools we expose. Each name MUST match a registered Hermes tool that
# `model_tools.handle_function_call()` can dispatch.
#
# What we deliberately DO NOT expose:
#   - terminal / shell / read_file / write_file / patch / search_files /
#     process — codex's built-ins cover these and approval routes through
#     codex's own UI.
#   - delegate_task / memory / session_search / todo — these are
#     `_AGENT_LOOP_TOOLS` in Hermes (model_tools.py:493). They require
#     the running AIAgent context to dispatch (mid-loop state), so a
#     stateless MCP callback can't drive them. Hermes' default runtime
#     keeps these working; the codex_app_server runtime cannot.
EXPOSED_TOOLS: tuple[str, ...] = (
    "web_search",
    "web_extract",
    "browser_navigate",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_snapshot",
    "browser_scroll",
    "browser_back",
    "browser_get_images",
    "browser_console",
    "browser_vision",
    "vision_analyze",
    "image_generate",
    "skill_view",
    "skills_list",
    "text_to_speech",
    # Kanban worker handoff tools — gated on HERMES_KANBAN_TASK env var
    # (set by the kanban dispatcher when spawning a worker). Without these
    # in the callback, a worker spawned with openai_runtime=codex_app_server
    # could do the work but couldn't report completion back to the kernel,
    # making it hang until timeout. Stateless dispatch — they just read
    # the env var and write to ~/.hermes/kanban.db.
    "kanban_complete",
    "kanban_block",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_show",
    "kanban_list",
    # NOTE: kanban_create / kanban_unblock / kanban_link are orchestrator-
    # only — the kanban tool gates them on HERMES_KANBAN_TASK being unset.
    # They're exposed here for orchestrator agents running on the codex
    # runtime that need to dispatch new tasks.
    "kanban_create",
    "kanban_unblock",
    "kanban_link",
)


# Bespoke callback tools that do NOT go through model_tools.handle_function_call().
# They are intentionally separate from EXPOSED_TOOLS because they do not exist in
# the global Hermes registry and cannot be dispatched as ordinary model tools.
CODEX_CALLBACK_TOOLS: tuple[str, ...] = ("codex_delegate_task",)


_CODEX_DELEGATE_DEFAULT_BOARD = "codex-delegate"
_CODEX_DELEGATE_DEFAULT_TIMEOUT_SECONDS = 600.0
_CODEX_DELEGATE_DEFAULT_MAX_CHILDREN = 1
_CODEX_DELEGATE_MAX_CHILDREN = 10
_CODEX_DELEGATE_TERMINAL_STATUSES = {"done", "blocked", "archived"}


def _redacted_json(payload: Any) -> str:
    """JSON-encode a callback response after redacting string leaves.

    Redact before ``json.dumps`` so regex redaction cannot accidentally consume
    JSON delimiters inside an already-encoded string value.
    """
    from agent.redact import redact_sensitive_text

    def _scrub(value: Any) -> Any:
        if isinstance(value, str):
            return redact_sensitive_text(value, force=True)
        if isinstance(value, dict):
            return {
                redact_sensitive_text(str(key), force=True): _scrub(inner)
                for key, inner in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [_scrub(inner) for inner in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return redact_sensitive_text(str(value), force=True)

    return json.dumps(_scrub(payload), ensure_ascii=False, default=str)


def _coerce_bool(value: Any, *, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "sync", "synchronous"}:
        return True
    if text in {"0", "false", "no", "off", "async", "asynchronous"}:
        return False
    return default


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_timeout_seconds(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = _CODEX_DELEGATE_DEFAULT_TIMEOUT_SECONDS
    return max(0.05, min(24 * 60 * 60.0, parsed))


def _truncate(value: Any, *, limit: int = 8000) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _extract_usage_metadata(metadata: Optional[dict]) -> dict[str, Any]:
    """Return bounded token/cost-ish metadata only; never arbitrary payloads."""
    if not isinstance(metadata, dict):
        return {}
    from agent.redact import redact_sensitive_text

    allowed: dict[str, Any] = {}
    for key, value in metadata.items():
        key_text = str(key)
        lowered = key_text.lower()
        if not any(marker in lowered for marker in ("token", "usage", "cost", "price", "model", "api_call")):
            continue
        safe_key = redact_sensitive_text(key_text, force=True)
        if isinstance(value, (int, float, bool)) or value is None:
            allowed[safe_key] = value
        elif isinstance(value, str):
            allowed[safe_key] = _truncate(redact_sensitive_text(value, force=True), limit=500)
        else:
            allowed[safe_key] = _truncate(
                redact_sensitive_text(json.dumps(value, ensure_ascii=False, default=str), force=True),
                limit=1000,
            )
    return allowed


def _resolve_codex_delegate_assignee() -> str:
    explicit = os.environ.get("HERMES_CODEX_DELEGATE_ASSIGNEE", "").strip()
    if explicit:
        return explicit
    env_profile = os.environ.get("HERMES_PROFILE", "").strip()
    if env_profile:
        return env_profile
    try:
        from hermes_cli.profiles import get_active_profile_name

        active = get_active_profile_name()
        if active and active != "custom":
            return active
    except Exception:
        pass
    return "default"


def _resolve_codex_delegate_board() -> str:
    explicit = os.environ.get("HERMES_CODEX_DELEGATE_BOARD", "").strip()
    if explicit:
        return explicit
    # Deliberately isolate delegate-shim fan-out from the operator's live board.
    # The shim calls dispatch_once() itself; using the active board could spawn
    # unrelated ready cards while a Codex caller only asked for subagents.
    return _CODEX_DELEGATE_DEFAULT_BOARD


def _build_child_body(task_description: str, *, index: int, total: int) -> str:
    return (
        "# codex_delegate_task child\n\n"
        f"Child: {index}/{total}\n\n"
        "## Parent task\n"
        f"{task_description.strip()}\n\n"
        "## Operating contract\n"
        "- Work independently and return a concise, evidence-backed result.\n"
        "- Use the full Hermes worker tool/skill surface available to this profile.\n"
        "- Do not ask the operator unless genuinely blocked.\n"
        "- Complete this Kanban card with `kanban_complete` when done.\n"
        "- If you cannot complete it, use `kanban_block` with the concrete blocker.\n"
    )


def _dispatch_result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "reclaimed": getattr(result, "reclaimed", 0),
        "promoted": getattr(result, "promoted", 0),
        "spawned": list(getattr(result, "spawned", []) or []),
        "skipped_unassigned": list(getattr(result, "skipped_unassigned", []) or []),
        "skipped_nonspawnable": list(getattr(result, "skipped_nonspawnable", []) or []),
        "crashed": list(getattr(result, "crashed", []) or []),
        "auto_blocked": list(getattr(result, "auto_blocked", []) or []),
        "timed_out": list(getattr(result, "timed_out", []) or []),
        "stale": list(getattr(result, "stale", []) or []),
        "respawn_guarded": list(getattr(result, "respawn_guarded", []) or []),
    }


def _collect_codex_delegate_children(kb: Any, conn: Any, task_ids: list[str]) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = kb.get_task(conn, task_id)
        run = kb.latest_run(conn, task_id)
        child: dict[str, Any] = {"task_id": task_id}
        if task is None:
            child.update({"status": "missing", "error": "task row missing"})
            children.append(child)
            continue
        child.update(
            {
                "status": task.status,
                "assignee": task.assignee,
                "result": _truncate(task.result, limit=8000),
                "last_failure_error": _truncate(task.last_failure_error, limit=2000),
                "worker_pid": task.worker_pid,
                "completed_at": task.completed_at,
            }
        )
        if run is not None:
            child["run"] = {
                "id": run.id,
                "status": run.status,
                "outcome": run.outcome,
                "summary": _truncate(run.summary, limit=8000),
                "error": _truncate(run.error, limit=2000),
                "usage": _extract_usage_metadata(run.metadata),
                "started_at": run.started_at,
                "ended_at": run.ended_at,
            }
        children.append(child)
    return children


def _codex_delegate_synthesis(children: list[dict[str, Any]]) -> str:
    done = [c for c in children if c.get("status") == "done"]
    blocked = [c for c in children if c.get("status") == "blocked"]
    archived = [c for c in children if c.get("status") == "archived"]
    lines = [
        f"codex_delegate_task children complete: {len(done)}/{len(children)} done"
    ]
    if blocked:
        lines.append(f"blocked: {len(blocked)}")
    if archived:
        lines.append(f"archived: {len(archived)}")
    for idx, child in enumerate(children, start=1):
        run = child.get("run") or {}
        summary = run.get("summary") or child.get("result") or run.get("error") or child.get("last_failure_error") or ""
        summary = str(summary).strip().splitlines()[0][:500] if summary else ""
        lines.append(
            f"- child {idx} {child.get('task_id')} [{child.get('status')}]"
            + (f": {summary}" if summary else "")
        )
    return "\n".join(lines)


def _codex_delegate_status(children: list[dict[str, Any]]) -> str:
    statuses = {str(c.get("status")) for c in children}
    if statuses <= {"done"}:
        return "success"
    if statuses <= _CODEX_DELEGATE_TERMINAL_STATUSES:
        return "partial" if "done" in statuses else "error"
    return "running"


def _block_unfinished_children(kb: Any, conn: Any, task_ids: list[str], *, reason: str) -> None:
    for task_id in task_ids:
        task = kb.get_task(conn, task_id)
        if task is None or task.status in _CODEX_DELEGATE_TERMINAL_STATUSES:
            continue
        try:
            kb.reclaim_task(conn, task_id, reason=reason)
        except Exception:
            logger.debug("failed to reclaim timed-out delegate child %s", task_id, exc_info=True)
        try:
            kb.block_task(conn, task_id, reason=reason)
        except Exception:
            logger.debug("failed to block timed-out delegate child %s", task_id, exc_info=True)


def _invocation_board(base_board: str, invocation_id: str) -> str:
    """Return a per-call board slug to avoid cross-call concurrency starvation."""
    prefix = str(base_board or _CODEX_DELEGATE_DEFAULT_BOARD).strip().lower()
    # kanban_db.create_board performs final validation. This only enforces the
    # 64-char max after appending the invocation suffix.
    max_prefix = max(1, 64 - len(invocation_id) - 1)
    return f"{prefix[:max_prefix]}-{invocation_id}"


def _spawned_task_ids(dispatch: dict[str, Any]) -> set[str]:
    spawned: set[str] = set()
    for item in dispatch.get("spawned") or []:
        if isinstance(item, (list, tuple)) and item:
            spawned.add(str(item[0]))
    return spawned


def _async_delegate_status(
    dispatch: dict[str, Any],
    children: list[dict[str, Any]],
    task_ids: list[str],
) -> tuple[str, Optional[str]]:
    child_status = _codex_delegate_status(children)
    if child_status != "running":
        return child_status, None
    created = set(task_ids)
    spawned = _spawned_task_ids(dispatch) & created
    if spawned == created:
        return "queued", None
    blockers = set(dispatch.get("skipped_nonspawnable") or []) & created
    auto_blocked = set(dispatch.get("auto_blocked") or []) & created
    if spawned:
        reason = None
        if blockers or auto_blocked:
            reason = "some codex_delegate_task children could not be spawned"
        return "partial_queued", reason
    if blockers:
        return "error", "codex_delegate_task children were assigned to a non-spawnable profile"
    if auto_blocked:
        return "error", "codex_delegate_task children were auto-blocked during dispatch"
    return "error", "codex_delegate_task dispatch did not spawn any children"


def _sync_dispatch_failed(dispatch: dict[str, Any], task_ids: list[str]) -> Optional[str]:
    created = set(task_ids)
    if _spawned_task_ids(dispatch) & created:
        return None
    if set(dispatch.get("skipped_nonspawnable") or []) & created:
        return "codex_delegate_task children were assigned to a non-spawnable profile"
    if set(dispatch.get("auto_blocked") or []) & created:
        return "codex_delegate_task children were auto-blocked during dispatch"
    return None


def _codex_delegate_task_impl(
    task_description: str,
    max_children: int = _CODEX_DELEGATE_DEFAULT_MAX_CHILDREN,
    synchronous: bool = True,
    timeout_seconds: float = _CODEX_DELEGATE_DEFAULT_TIMEOUT_SECONDS,
    *,
    spawn_fn: Optional[Callable[..., Any]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Kanban-backed implementation for the Codex-only delegate shim."""
    description = str(task_description or "").strip()
    if not description:
        raise ValueError("task_description is required")
    child_count = _coerce_int(
        max_children,
        default=_CODEX_DELEGATE_DEFAULT_MAX_CHILDREN,
        minimum=1,
        maximum=_CODEX_DELEGATE_MAX_CHILDREN,
    )
    sync = _coerce_bool(synchronous, default=True)
    timeout_s = _coerce_timeout_seconds(timeout_seconds)
    runtime_cap_s = max(1, int(timeout_s))
    base_board = _resolve_codex_delegate_board()
    assignee = _resolve_codex_delegate_assignee()
    session_id = os.environ.get("HERMES_SESSION_ID") or os.environ.get("HERMES_GATEWAY_SESSION_ID")
    invocation_id = f"cdt_{uuid.uuid4().hex[:12]}"
    board = _invocation_board(base_board, invocation_id)
    cwd = Path(os.environ.get("HERMES_CODEX_DELEGATE_WORKDIR") or os.getcwd()).expanduser().resolve()
    title_seed = " ".join(description.split())[:80]

    from hermes_cli import kanban_db as kb

    try:
        kb.create_board(
            board,
            name="Codex Delegate",
            description="Isolated board for codex_delegate_task Kanban-backed fan-out.",
            default_workdir=str(cwd),
        )
    except Exception:
        # Non-fatal: connect(board=...) can still create/open the DB, and tests
        # often pin HERMES_KANBAN_DB without creating board metadata.
        logger.debug("could not create/update codex delegate board %s", board, exc_info=True)

    conn = kb.connect(board=board)
    dispatch_history: list[dict[str, Any]] = []
    try:
        task_ids: list[str] = []
        for idx in range(1, child_count + 1):
            task_id = kb.create_task(
                conn,
                title=f"codex_delegate_task {idx}/{child_count}: {title_seed}",
                body=_build_child_body(description, index=idx, total=child_count),
                assignee=assignee,
                created_by="codex_delegate_task",
                workspace_kind="dir",
                workspace_path=str(cwd),
                tenant="codex_delegate_task",
                priority=1000,
                max_runtime_seconds=runtime_cap_s,
                max_retries=1,
                session_id=session_id,
                board=board,
            )
            task_ids.append(task_id)

        first_dispatch = kb.dispatch_once(
            conn,
            board=board,
            spawn_fn=spawn_fn,
            max_spawn=child_count,
            max_in_progress=child_count,
            failure_limit=1,
            stale_timeout_seconds=max(1, runtime_cap_s),
        )
        dispatch_history.append(_dispatch_result_to_dict(first_dispatch))

        base_payload: dict[str, Any] = {
            "tool": "codex_delegate_task",
            "invocation_id": invocation_id,
            "base_board": base_board,
            "board": board,
            "assignee": assignee,
            "task_ids": task_ids,
            "max_children": child_count,
            "synchronous": sync,
            "timeout_seconds": timeout_s,
            "dispatch": dispatch_history[-1],
            "poll_hint": "Use kanban_show / kanban_list with both the returned board and task_ids to inspect async progress.",
        }

        initial_failure = _sync_dispatch_failed(dispatch_history[-1], task_ids)
        if initial_failure and sync:
            _block_unfinished_children(kb, conn, task_ids, reason=initial_failure)
            children = _collect_codex_delegate_children(kb, conn, task_ids)
            return {
                **base_payload,
                "status": "error",
                "children": children,
                "dispatch_history": dispatch_history,
                "synthesis": _codex_delegate_synthesis(children),
                "error": initial_failure,
            }

        if not sync:
            children = _collect_codex_delegate_children(kb, conn, task_ids)
            async_status, async_error = _async_delegate_status(dispatch_history[-1], children, task_ids)
            payload = {
                **base_payload,
                "status": async_status,
                "children": children,
                "synthesis": _codex_delegate_synthesis(children),
            }
            if async_error:
                payload["error"] = async_error
            return payload

        deadline = monotonic_fn() + timeout_s
        poll_interval = min(0.5, max(0.05, timeout_s / 20.0))
        while True:
            children = _collect_codex_delegate_children(kb, conn, task_ids)
            status = _codex_delegate_status(children)
            if status != "running":
                return {
                    **base_payload,
                    "status": status,
                    "children": children,
                    "dispatch_history": dispatch_history,
                    "synthesis": _codex_delegate_synthesis(children),
                }
            remaining = deadline - monotonic_fn()
            if remaining <= 0:
                reason = f"codex_delegate_task timeout after {timeout_s:g}s"
                _block_unfinished_children(kb, conn, task_ids, reason=reason)
                children = _collect_codex_delegate_children(kb, conn, task_ids)
                return {
                    **base_payload,
                    "status": "timeout",
                    "children": children,
                    "dispatch_history": dispatch_history,
                    "synthesis": _codex_delegate_synthesis(children),
                    "error": reason,
                }
            dispatch_result = kb.dispatch_once(
                conn,
                board=board,
                spawn_fn=spawn_fn,
                max_spawn=child_count,
                max_in_progress=child_count,
                failure_limit=1,
                stale_timeout_seconds=max(1, runtime_cap_s),
            )
            dispatch_dict = _dispatch_result_to_dict(dispatch_result)
            if any(dispatch_dict.get(k) for k in ("reclaimed", "spawned", "crashed", "auto_blocked", "timed_out", "stale")):
                dispatch_history.append(dispatch_dict)
            sleep_fn(min(poll_interval, max(0.01, remaining)))
    finally:
        conn.close()


def _plugin_context_ids() -> tuple[str, str, str]:
    return (
        os.environ.get("HERMES_KANBAN_TASK") or os.environ.get("HERMES_TASK_ID") or "",
        os.environ.get("HERMES_SESSION_ID") or os.environ.get("HERMES_GATEWAY_SESSION_ID") or "",
        os.environ.get("HERMES_TOOL_CALL_ID") or "",
    )


def _apply_codex_delegate_hooks(args: dict[str, Any], result: str, *, duration_ms: int) -> str:
    task_id, session_id, tool_call_id = _plugin_context_ids()
    try:
        from hermes_cli.plugins import invoke_hook

        invoke_hook(
            "post_tool_call",
            tool_name="codex_delegate_task",
            args=args,
            result=result,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            duration_ms=duration_ms,
        )
        hook_results = invoke_hook(
            "transform_tool_result",
            tool_name="codex_delegate_task",
            args=args,
            result=result,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
            duration_ms=duration_ms,
        )
        for hook_result in hook_results:
            if isinstance(hook_result, str):
                from agent.redact import redact_sensitive_text

                return redact_sensitive_text(hook_result, force=True)
    except Exception:
        logger.debug("codex_delegate_task plugin hook failed", exc_info=True)
    return result


def codex_delegate_task(
    task_description: str,
    max_children: int = _CODEX_DELEGATE_DEFAULT_MAX_CHILDREN,
    synchronous: bool = True,
    timeout_seconds: float = _CODEX_DELEGATE_DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Spawn durable Kanban-backed child workers from Codex app-server.

    This is the Codex-runtime compatibility shim for Hermes' native
    delegate_task UX. It returns JSON. In synchronous mode it waits for child
    completion and includes a deterministic synthesis; in async mode it returns
    task IDs immediately so the caller can poll with Kanban tools.
    """
    args = {
        "task_description": task_description,
        "max_children": max_children,
        "synchronous": synchronous,
        "timeout_seconds": timeout_seconds,
    }
    task_id, session_id, tool_call_id = _plugin_context_ids()
    start = time.monotonic()
    try:
        from hermes_cli.plugins import get_pre_tool_call_block_message

        block_message = get_pre_tool_call_block_message(
            "codex_delegate_task",
            args,
            task_id=task_id,
            session_id=session_id,
            tool_call_id=tool_call_id,
        )
    except Exception:
        logger.debug("codex_delegate_task pre_tool_call hook failed", exc_info=True)
        block_message = None

    if block_message is not None:
        result = _redacted_json({"tool": "codex_delegate_task", "status": "blocked", "error": block_message})
        return _apply_codex_delegate_hooks(args, result, duration_ms=int((time.monotonic() - start) * 1000))

    try:
        payload = _codex_delegate_task_impl(
            task_description,
            max_children=max_children,
            synchronous=synchronous,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        logger.exception("codex_delegate_task failed")
        payload = {"tool": "codex_delegate_task", "status": "error", "error": str(exc)}
    result = _redacted_json(payload)
    return _apply_codex_delegate_hooks(args, result, duration_ms=int((time.monotonic() - start) * 1000))


def _register_codex_callback_tools(mcp: Any) -> int:
    description = (
        "Kanban-backed delegate_task shim for Codex app-server. Arguments: "
        "task_description, max_children, synchronous, timeout_seconds. "
        "Synchronous mode waits and returns child results plus synthesis; "
        "async mode returns task IDs for Kanban polling."
    )
    try:
        mcp.add_tool(
            codex_delegate_task,
            name="codex_delegate_task",
            description=description,
        )
    except TypeError:
        handler = mcp.tool(name="codex_delegate_task", description=description)(codex_delegate_task)
        _ = handler
    return 1


def _build_server() -> Any:
    """Create the FastMCP server with Hermes tools attached. Lazy imports
    so the module can be imported without the mcp package installed
    (we degrade to a clear error only when actually run)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - install hint
        raise ImportError(
            f"hermes-tools MCP server requires the 'mcp' package: {exc}"
        ) from exc

    # Discover Hermes tools so dispatch works.
    from model_tools import (
        get_tool_definitions,
        handle_function_call,
    )

    mcp = FastMCP(
        "hermes-tools",
        instructions=(
            "Hermes Agent's tool surface, exposed for use inside a Codex "
            "session. Use these for capabilities Codex's built-in toolset "
            "doesn't cover: web search/extract, browser automation, "
            "Kanban-backed subagent delegation, vision, image generation, "
            "skills, and Kanban handoff."
        ),
    )

    codex_callback_count = _register_codex_callback_tools(mcp)

    # Pull authoritative Hermes tool schemas for the ones we expose, so
    # MCP clients see the same parameter docs Hermes gives the model.
    all_defs = {
        td["function"]["name"]: td["function"]
        for td in (get_tool_definitions(quiet_mode=True) or [])
        if isinstance(td, dict) and td.get("type") == "function"
    }

    exposed_count = 0

    for name in EXPOSED_TOOLS:
        spec = all_defs.get(name)
        if spec is None:
            logger.debug(
                "skipping %s — not registered in this Hermes process", name
            )
            continue

        description = spec.get("description") or f"Hermes {name} tool"
        params_schema = spec.get("parameters") or {"type": "object", "properties": {}}

        # FastMCP wants a Python callable. Build a closure that takes the
        # arguments dict, dispatches via handle_function_call, and returns
        # the result string. We use add_tool() for full control over the
        # input schema (FastMCP's @tool() decorator inspects type hints,
        # which we can't get from a JSON schema at runtime).
        def _make_handler(tool_name: str):
            def _dispatch(**kwargs: Any) -> str:
                try:
                    return handle_function_call(tool_name, kwargs or {})
                except Exception as exc:
                    logger.exception("tool %s raised", tool_name)
                    return json.dumps({"error": str(exc), "tool": tool_name})
            _dispatch.__name__ = tool_name
            _dispatch.__doc__ = description
            return _dispatch

        try:
            mcp.add_tool(
                _make_handler(name),
                name=name,
                description=description,
                # FastMCP accepts JSON schema directly via the
                # input_schema parameter on newer versions; older
                # versions use parameters_schema. Try both for compat.
            )
        except TypeError:
            # Older mcp SDK signature — fall back to decorator-style.
            handler = _make_handler(name)
            handler = mcp.tool(name=name, description=description)(handler)

        exposed_count += 1

    logger.info(
        "hermes-tools MCP server registered %d/%d registry tools and %d/%d callback tools",
        exposed_count,
        len(EXPOSED_TOOLS),
        codex_callback_count,
        len(CODEX_CALLBACK_TOOLS),
    )
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for `python -m agent.transports.hermes_tools_mcp_server`."""
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv

    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,  # MCP uses stdio for protocol — logs MUST go to stderr
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Quiet mode: keep Hermes' own banners off stdout (which is the MCP wire).
    os.environ.setdefault("HERMES_QUIET", "1")
    os.environ.setdefault("HERMES_REDACT_SECRETS", "true")

    try:
        server = _build_server()
    except ImportError as exc:
        sys.stderr.write(f"hermes-tools MCP server cannot start: {exc}\n")
        return 2

    # FastMCP runs with stdio transport by default when launched as a
    # subprocess.
    try:
        server.run()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("hermes-tools MCP server crashed")
        sys.stderr.write(f"hermes-tools MCP server error: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
