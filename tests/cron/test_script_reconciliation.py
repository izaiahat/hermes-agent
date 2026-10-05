"""Authoring pins remain profile-local across the stable scheduler split."""
import hashlib

import pytest

from cron import jobs
from cron.job_definition import merge_job_definition
from cron.scheduler_script import _run_job_script


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "scripts").mkdir()
    return tmp_path


def test_script_registration_and_timeout_follow_authoring_not_runtime_drift(store):
    script = store / "scripts" / "fixture.py"
    script.write_text("print('old')\n")
    first_hash = hashlib.sha256(script.read_bytes()).hexdigest()
    job = jobs.create_job("fixture", "every 1h", script="fixture.py", script_timeout_seconds=17, paused=True, paused_reason="fixture hold")
    assert job["script_sha256"] == first_hash
    assert jobs.get_job(job["id"])["script_timeout_seconds"] == 17
    script.write_text("print('new')\n")
    unrelated = jobs.update_job(job["id"], {"name": "renamed fixture"})
    assert unrelated["script_sha256"] == first_hash
    assert unrelated["enabled"] is False
    assert unrelated["paused_reason"] == "fixture hold"
    authored = jobs.update_job(job["id"], {"script": "fixture.py", "script_timeout_seconds": 19})
    assert authored["script_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert authored["script_timeout_seconds"] == 19
    assert authored["paused_reason"] == "fixture hold"


def test_definition_replication_carries_script_authority_but_preserves_local_hold():
    local = {"id": "isolated-fixture", "enabled": False, "state": "paused", "paused_reason": "held-unknown", "next_run_at": None, "script_sha256": "old", "script_timeout_seconds": 17}
    authored = {"script": "new.py", "script_sha256": "new", "script_timeout_seconds": 19}
    merged = merge_job_definition(local, authored)
    assert merged["script_sha256"] == "new"
    assert merged["script_timeout_seconds"] == 19
    assert merged["enabled"] is False
    assert merged["paused_reason"] == "held-unknown"
    assert merged["next_run_at"] is None


def test_spawn_failure_releases_verified_snapshot_descriptor(store, monkeypatch):
    from cron import scheduler_script
    import os

    script = store / "scripts" / "fixture.py"
    script.write_text("print('fixture')\n")
    descriptors = []

    def fail_spawn(argv, **kwargs):
        descriptors.extend(kwargs["pass_fds"])
        raise OSError("isolated spawn failure")

    monkeypatch.setattr(scheduler_script.subprocess, "Popen", fail_spawn)
    ok, output = _run_job_script(str(script))
    assert not ok and "isolated spawn failure" in output
    assert descriptors
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
