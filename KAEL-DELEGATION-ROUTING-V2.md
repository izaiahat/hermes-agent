# Kael Delegation Routing v2

This fork carries a hardcoded delegation router for Kael at `scripts/kael_delegation_router.py`.

## Live doctrine pointers
- Live skill: `~/.hermes/skills/delegation-routing-v2/SKILL.md`
- Live config: `~/.hermes/config.yaml` under `delegation` and `kael_delegation`
- Live spec: `/home/ubuntu/business/reports/delegation-system-hardcoded-2026-04-25.md`
- Prior architecture grounding: `/home/ubuntu/business/reports/subagent-kael-delegation-architecture-2026-04-24.md`
- Fork workflow: `/home/ubuntu/business/reports/hermes-fork-workflow-2026-04-25.md`

## Lanes
1. `parent` — `gpt-5.4`, final synthesis, shared-state writes, irreversible actions
2. `gpt55_specialist` — `gpt-5.5` leaf, bounded reasoning and structured output
3. `codex_cli_long_context` — Codex CLI, 5+ files / 300k+ tokens / clean-room probes
4. `gpt54_orchestrator_cli` — Hermes CLI `gpt-5.4` orchestrator child for multi-domain synthesis

## Hardcoded routing rules
- A: shared-state / irreversible action → parent
- B: single tool call or pure reasoning under 50k → parent
- C: bounded structured work under 200k → `gpt55_specialist`
- D: 5+ files, >300k tokens, or clean-room → `codex_cli_long_context`
- E: N independent subtasks → parallel fanout using child lane chosen by subtask size
- F: 2+ broad domains needing local synthesis → `gpt54_orchestrator_cli`

## Smoke test
```bash
cd ~/.hermes/hermes-agent
./scripts/kael_delegation_router.py --examples
```

The router output is intentionally machine-readable JSON so it can be used in shell checks or future wrappers.
