"""2026-08-13 starvation fix: no_agent script jobs parallelize despite workdir.

Regression for the single-thread pile-up where 34 no_agent business jobs
serialized behind multi-hour runs, starving 5/15-minute pollers (payments
poller "blind" -> urgent-alert storms). Script jobs never mutate
TERMINAL_CWD (workdir is subprocess cwd), so only AGENT workdir jobs may be
classified sequential.
"""
import re
from pathlib import Path

SCHEDULER = Path(__file__).resolve().parents[2] / "cron" / "scheduler.py"


def _partition(due_jobs):
    # Mirror of the tick partition predicate — keep in lockstep with scheduler.
    def mutates(j):
        return bool((j.get("workdir") or "").strip()) and not j.get("no_agent")
    return ([j for j in due_jobs if mutates(j)],
            [j for j in due_jobs if not mutates(j)])


def test_no_agent_workdir_jobs_go_parallel():
    jobs = [
        {"id": "script-wd", "workdir": "/home/ubuntu/business", "no_agent": True},
        {"id": "agent-wd", "workdir": "/home/ubuntu/business", "no_agent": False},
        {"id": "agent-nowd", "workdir": "", "no_agent": False},
        {"id": "script-nowd", "no_agent": True},
    ]
    sequential, parallel = _partition(jobs)
    assert [j["id"] for j in sequential] == ["agent-wd"]
    assert {j["id"] for j in parallel} == {"script-wd", "agent-nowd", "script-nowd"}


def test_scheduler_source_uses_no_agent_aware_predicate():
    src = SCHEDULER.read_text(encoding="utf-8")
    assert "_mutates_terminal_cwd" in src
    assert re.search(r"and not j\.get\(\"no_agent\"\)", src)
    # the old naive partition must be gone
    assert 'sequential_jobs = [j for j in due_jobs if (j.get("workdir") or "").strip()]' not in src
