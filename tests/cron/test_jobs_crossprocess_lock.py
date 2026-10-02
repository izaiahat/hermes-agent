"""Regression test for the jobs.json cross-process lock.

Background: ``hermes cron pause`` runs in its own process (CLI → cronjob tool →
``pause_job`` → ``update_job`` → ``save_jobs``), entirely separate from the
gateway process that also writes ``jobs.json`` (``mark_job_run`` /
``advance_next_run`` / due-fast-forward). The module's ``threading.Lock`` only
serializes writers *inside one process*, so a CLI pause issued while the gateway
was live could be silently lost to a concurrent gateway write — the job kept
firing even though the CLI reported "Paused".

``_jobs_lock()`` closes that gap with a short-held cross-process advisory file
lock. This test proves the lock actually excludes a *separate process*, which an
in-process ``threading.Lock`` cannot do.
"""

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

from cron import jobs


# Repo root (parent of the ``cron`` package) so the child process can import it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(jobs.__file__)))


@pytest.mark.skipif(jobs.fcntl is None, reason="POSIX fcntl/flock required")
def test_jobs_lock_excludes_another_process(tmp_path, monkeypatch):
    cron_dir = tmp_path / "cron"
    output_dir = cron_dir / "output"
    monkeypatch.setattr(jobs, "CRON_DIR", cron_dir)
    monkeypatch.setattr(jobs, "JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", output_dir)

    ready = tmp_path / "child_holds_lock"
    release = tmp_path / "child_may_release"
    blocker_started = tmp_path / "blocker_started"
    blocker_acquired = tmp_path / "blocker_acquired"
    holder = tmp_path / "holder.py"
    holder.write_text(
        textwrap.dedent(
            f"""
            import sys, time, pathlib
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

            with jobs._jobs_lock():
                pathlib.Path({str(ready)!r}).write_text("1")
                # Hold the lock until the parent signals (bounded so a wedged
                # test can never hang CI).
                for _ in range(1000):
                    if pathlib.Path({str(release)!r}).exists():
                        break
                    time.sleep(0.01)
            """
        )
    )

    blocker = tmp_path / "blocker.py"
    blocker.write_text(
        textwrap.dedent(
            f"""
            import sys, pathlib
            sys.path.insert(0, {_REPO_ROOT!r})
            from cron import jobs

            jobs.CRON_DIR = pathlib.Path({str(cron_dir)!r})
            jobs.JOBS_FILE = jobs.CRON_DIR / "jobs.json"
            jobs.OUTPUT_DIR = jobs.CRON_DIR / "output"

            pathlib.Path({str(blocker_started)!r}).write_text("1")
            with jobs._jobs_lock():
                pathlib.Path({str(blocker_acquired)!r}).write_text("1")
            """
        )
    )

    child = subprocess.Popen([sys.executable, str(holder)])
    blocker_child = None
    try:
        # Wait until the child is inside the critical section.
        for _ in range(1000):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists(), "child never acquired _jobs_lock()"

        # While the child holds it, a non-blocking acquire of the SAME lock file
        # from this process must fail. A threading.Lock could never block here.
        lock_file = jobs._jobs_lock_file()
        fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(OSError):
                jobs.fcntl.flock(fd, jobs.fcntl.LOCK_EX | jobs.fcntl.LOCK_NB)
        finally:
            os.close(fd)

        # A second _jobs_lock() caller in another process should block until the
        # holder releases, rather than falling through with only a process-local
        # threading lock.
        blocker_child = subprocess.Popen([sys.executable, str(blocker)])
        for _ in range(1000):
            if blocker_started.exists():
                break
            time.sleep(0.01)
        assert blocker_started.exists(), "blocker process never started"
        time.sleep(0.05)
        assert not blocker_acquired.exists(), "second process entered _jobs_lock() while held"
    finally:
        release.write_text("1")
        child.wait(timeout=15)
        if blocker_child is not None:
            blocker_child.wait(timeout=15)

    assert blocker_acquired.exists(), "second process did not acquire _jobs_lock() after release"

    # Once the child has released, the lock is freely acquirable again.
    with jobs._jobs_lock():
        pass


@pytest.mark.linux_only
def test_timed_out_writer_cannot_save_under_held_registry_lock(tmp_path, monkeypatch):
    """F11: the real 30s fallback must never authorize a registry mutation."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    row = {"id": "fixture-only", "name": "before", "enabled": False,
           "state": "paused", "prompt": "No effect fixture",
           "schedule": {"kind": "interval", "minutes": 60}, "next_run_at": None}
    jobs.save_jobs([row], replace=True)
    registry = jobs._current_cron_store().jobs_file
    before = registry.read_bytes()
    child_code = """
import json, time
from cron import jobs
start = time.monotonic()
with jobs._jobs_lock():
    cross_process = jobs._jobs_lock_state.cross_process
    rows = json.loads(jobs._current_cron_store().jobs_file.read_text())["jobs"]
    rows[0]["name"] = "ordinary-writer-landed"
    error = None
    try:
        jobs.save_jobs(rows, replace=True)
    except RuntimeError as exc:
        error = str(exc)
print(json.dumps({"cross_process": cross_process, "error": error,
                  "elapsed": time.monotonic() - start}))
"""
    with jobs._jobs_lock(require_cross_process=True):
        lock_path, lock_fd = jobs._jobs_lock_state.custody
        held = os.fstat(lock_fd.fileno())
        child = subprocess.run(
            [sys.executable, "-c", child_code], cwd=_REPO_ROOT,
            env=dict(os.environ, PYTHONPATH=_REPO_ROOT),
            capture_output=True, text=True, timeout=60, check=True,
        )
        observation = json.loads(child.stdout)
        current = lock_path.stat()
        assert jobs._jobs_lock_state.cross_process
        assert (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)
        assert observation["cross_process"] is False
        assert observation["elapsed"] >= jobs._JOBS_LOCK_TIMEOUT_SECONDS
        assert registry.read_bytes() == before, observation
        assert "cross-process lock required" in observation["error"]

    # Once custody is available, the same public saver works inside an ordinary
    # nested critical section. A refused attempt must not poison later writes.
    with jobs._jobs_lock():
        row["name"] = "legitimate-nested-write"
        jobs.save_jobs([row], replace=True)
    assert json.loads(registry.read_bytes())["jobs"] == [row]


@pytest.mark.linux_only
@pytest.mark.parametrize("failure", ["no_backend", "timeout", "denied"])
def test_degraded_registry_savers_preserve_bytes(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rows = [{"id": "fixture-only", "name": "before"}]
    jobs.save_jobs(rows, replace=True)
    registry = jobs._current_cron_store().jobs_file
    before = registry.read_bytes()
    if failure == "no_backend":
        monkeypatch.setattr(jobs, "fcntl", None)
        monkeypatch.setattr(jobs, "msvcrt", None)
    else:
        def unavailable(*args):
            if failure == "denied":
                raise PermissionError("fixture lock inaccessible")
            return False
        monkeypatch.setattr(jobs, "_acquire_flock", unavailable)

    rows[0]["name"] = "must-not-land"
    with pytest.raises(RuntimeError, match="cross-process lock required"):
        jobs.save_jobs(rows, replace=True)
    assert registry.read_bytes() == before
    with jobs._jobs_lock():
        assert jobs._jobs_lock_state.cross_process is False
        assert jobs.load_jobs() == [{"id": "fixture-only", "name": "before"}]
        for saver in (jobs.save_jobs, jobs._save_jobs_unlocked):
            with pytest.raises(RuntimeError, match="cross-process lock required"):
                saver(rows, replace=True)
            assert registry.read_bytes() == before
    assert not list(registry.parent.glob(".jobs_*.tmp"))
