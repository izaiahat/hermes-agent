"""Tests for cron job script injection feature.

Tests cover:
- Script field in job creation / storage / update
- Script execution and output injection into prompts
- Error handling (missing script, timeout, non-zero exit)
- Path resolution (absolute, relative to HERMES_HOME/scripts/)
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    (hermes_home / "scripts").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Clear cached module-level paths
    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class TestJobScriptField:
    """Test that the script field is stored and retrieved correctly."""

    def test_create_job_with_script(self, cron_env):
        from cron.jobs import create_job, get_job

        job = create_job(
            prompt="Analyze the data",
            schedule="every 30m",
            script="/path/to/monitor.py",
        )
        assert job["script"] == "/path/to/monitor.py"

        loaded = get_job(job["id"])
        assert loaded["script"] == "/path/to/monitor.py"


    def test_update_job_add_script(self, cron_env):
        from cron.jobs import create_job, update_job

        job = create_job(prompt="Hello", schedule="every 1h")
        assert job.get("script") is None

        updated = update_job(job["id"], {"script": "/new/script.py"})
        assert updated["script"] == "/new/script.py"


def test_cronjob_tool_rejects_stale_past_one_shot(cron_env, monkeypatch):
    from tools.cronjob_tools import cronjob

    now = datetime(2026, 3, 18, 4, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)
    stale = (now - timedelta(minutes=5)).isoformat()

    result = json.loads(cronjob(action="create", prompt="Too late", schedule=stale))

    assert result["success"] is False
    assert "past and cannot be scheduled" in result["error"]


class TestRunJobScript:
    """Test the _run_job_script() function."""

    def test_successful_script(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "test.py"
        script.write_text('print("hello from script")\n')

        success, output = _run_job_script(str(script))
        assert success is True
        assert output == "hello from script"

    def test_script_relative_path(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "relative.py"
        script.write_text('print("relative works")\n')

        success, output = _run_job_script("relative.py")
        assert success is True
        assert output == "relative works"


    def test_script_subprocess_env_sanitized(self, cron_env, monkeypatch):
        """Cron scripts must not inherit Hermes provider env (SECURITY.md §2.3)."""
        from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST
        from cron.scheduler import _run_job_script

        # sorted() so the probed var is deterministic across runs
        # (frozenset iteration order varies with PYTHONHASHSEED).
        blocked_var = sorted(_HERMES_PROVIDER_ENV_BLOCKLIST)[0]
        monkeypatch.setenv(blocked_var, "must_not_leak")

        script = cron_env / "scripts" / "env_probe.py"
        script.write_text(
            textwrap.dedent(
                f"""\
                import os
                key = {blocked_var!r}
                print("PRESENT" if os.environ.get(key) else "ABSENT")
                """
            )
        )

        success, output = _run_job_script("env_probe.py")
        assert success is True
        assert output == "ABSENT"

    @pytest.mark.windows_only
    def test_windows_uv_venv_python_script_bypasses_launcher(self, cron_env, tmp_path, monkeypatch):
        # Windows-only: the fake ``sys.platform`` could not reproduce the
        # ``Scripts/python.exe`` launcher layout or the CREATE_NO_WINDOW
        # creationflags this branch exists for.
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        venv = tmp_path / "venv"
        venv_scripts = venv / "Scripts"
        site_packages = venv / "Lib" / "site-packages"
        base = tmp_path / "base"
        venv_scripts.mkdir(parents=True)
        site_packages.mkdir(parents=True)
        base.mkdir()
        venv_python = venv_scripts / "python.exe"
        base_python = base / "python.exe"
        venv_python.write_text("", encoding="utf-8")
        base_python.write_text("", encoding="utf-8")
        (venv / "pyvenv.cfg").write_text(f"home = {base}\nuv = true\n", encoding="utf-8")

        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.sys, "executable", str(venv_python))
        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"] == [str(base_python), str(script.resolve())]
        assert captured["kwargs"]["creationflags"] == sched_mod.windows_hide_flags()
        env = captured["kwargs"]["env"]
        assert env["VIRTUAL_ENV"] == str(venv)
        assert str(site_packages) in env["PYTHONPATH"]


    def test_non_windows_script_preserves_default_text_decoding(self, cron_env, monkeypatch):
        # No platform patching: the Linux CI host already takes this branch.
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "probe.py"
        script.write_text('print("ok")\n')

        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script("probe.py")

        assert success is True
        assert output == "ok"
        assert captured["argv"][0] == sys.executable
        assert captured["argv"][1].startswith("/proc/self/fd/")
        assert captured["kwargs"]["pass_fds"]
        assert captured["kwargs"]["text"] is True
        assert "creationflags" not in captured["kwargs"]
        assert "encoding" not in captured["kwargs"]
        assert "errors" not in captured["kwargs"]

    def test_emoji_stdout_round_trips_through_script_capture(self, cron_env):
        """Emoji in script stdout must reach the caller intact (#42384).

        On Windows the fix is the utf-8 + errors='replace' popen kwargs
        (asserted above); on POSIX the UTF-8 locale default must already
        carry emoji through. Either way the delivery content is the real
        text, never an exception.
        """
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "emoji.py"
        script.write_text(
            'import sys\n'
            'sys.stdout.buffer.write("backup done \\N{PARTY POPPER} 日次".encode("utf-8"))\n',
            encoding="utf-8",
        )

        success, output = _run_job_script("emoji.py")

        assert success is True
        assert "backup done 🎉 日次" == output

    def test_invalid_utf8_stdout_does_not_raise(self, cron_env):
        """Truncated/invalid UTF-8 in script stdout must never escape as an
        exception (#47393) — a raised UnicodeDecodeError higher up would
        silently drop the whole delivery (#42384). The run may fail, but it
        must fail as a (False, message) result the scheduler can deliver.
        """
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "bad_bytes.py"
        # b'\xe6\x97' is the first two bytes of a three-byte CJK sequence —
        # a truncated write, exactly the shape reported in #47393.
        script.write_text(
            "import sys\n"
            "sys.stdout.buffer.write(b'partial \\xe6\\x97')\n",
            encoding="utf-8",
        )

        success, output = _run_job_script("bad_bytes.py")  # must not raise

        assert isinstance(success, bool)
        assert isinstance(output, str)
        assert output  # a message is always produced, never a silent drop

    def test_legitimate_self_edit_runs_current_script_despite_stale_registration(
        self, cron_env, monkeypatch, caplog
    ):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "pinned.py"
        script.write_text("print('current-self-edit')\n")

        def fake_run(argv, **kwargs):
            with open(argv[1], "rb") as descriptor_view:
                current = descriptor_view.read().decode()
            return SimpleNamespace(returncode=0, stdout=current, stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)
        with caplog.at_level("WARNING", logger="cron.scheduler"):
            success, output = _run_job_script(
                str(script), job={"script_sha256": "0" * 64}
            )
        assert success is True
        assert output == "print('current-self-edit')"
        assert "Cron script registration drift" in caplog.text
        assert "running the current on-disk bytes" in caplog.text

    def test_stale_registration_keeps_real_script_crash_honest(self, cron_env):
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "crash.py"
        script.write_text("raise RuntimeError('real-crash-marker')\n")
        success, output = _run_job_script(str(script), job={"script_sha256": "0" * 64})
        assert success is False
        assert "real-crash-marker" in output
        assert "success" not in output.lower()

    def test_external_effect_wrapper_still_refuses_stale_registration(
        self, cron_env, monkeypatch
    ):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "affiliate_auto_apply_daily.sh"
        script.write_text("#!/usr/bin/env bash\nprintf 'must-not-run\\n'\n")
        called = []
        monkeypatch.setattr(
            sched_mod.subprocess, "run", lambda *args, **kwargs: called.append(True)
        )
        success, output = _run_job_script(
            str(script),
            job={"id": "2dd6ae1a4db9", "script_sha256": "0" * 64},
        )
        assert success is False
        assert "cron_script_hash_mismatch" in output
        assert called == []

    def test_awin_auth_repair_wrapper_refuses_stale_registration(
        self, cron_env, monkeypatch
    ):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "awin_auth_liveness_preflight.sh"
        script.write_text("#!/usr/bin/env bash\nprintf 'must-not-run\\n'\n")
        called = []
        monkeypatch.setattr(
            sched_mod.subprocess, "run", lambda *args, **kwargs: called.append(True)
        )
        success, output = _run_job_script(
            str(script),
            job={"id": "7a269c665b12", "script_sha256": "0" * 64},
        )
        assert success is False
        assert "cron_script_hash_mismatch" in output
        assert called == []

    def test_executes_same_open_descriptor_that_was_hashed(self, cron_env, monkeypatch):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "descriptor.py"
        trusted = b"print('trusted')\n"
        script.write_bytes(trusted)
        expected = hashlib.sha256(trusted).hexdigest()

        def fake_run(argv, **kwargs):
            script.write_text("print('swapped')\n")
            with open(argv[1], "rb") as descriptor_view:
                executed = descriptor_view.read()
            return SimpleNamespace(returncode=0, stdout=executed.decode(), stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)
        success, output = _run_job_script(str(script), job={"script_sha256": expected})
        assert success is True
        assert output == trusted.decode().strip()

    def test_per_job_script_timeout_reaches_subprocess(self, cron_env, monkeypatch):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "paced.py"
        script.write_text('print("ok")\n')
        captured = {}

        def fake_run(argv, **kwargs):
            captured["timeout"] = kwargs["timeout"]
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)

        success, output = _run_job_script(
            str(script),
            job={"id": "paced", "script_timeout_seconds": 17},
        )

        assert success is True
        assert output == "ok"
        assert captured["timeout"] == 17

    def test_glp_awin_recurring_script_uses_descendant_safe_user_cgroup(
        self, cron_env, monkeypatch
    ):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "affiliate_portal_apply_daily.sh"
        script.write_text("#!/usr/bin/env bash\nprintf 'ok\\n'\n")
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)
        success, output = _run_job_script(
            str(script),
            job={
                "id": "52e2014a9579",
                "script_timeout_seconds": 1800,
                "script_sha256": digest,
            },
        )
        assert success is True and output == "ok"
        argv = captured["argv"]
        assert argv[:7] == [
            "/usr/bin/systemd-run", "--user", "--pipe", "--wait", "--collect",
            "--quiet", "--service-type=exec",
        ]
        assert "KillMode=control-group" in argv
        assert "RuntimeMaxSec=1790s" in argv
        assert captured["kwargs"]["env"]["HERMES_CRON_CGROUP_CONTAINED"] == "1"
        assert captured["kwargs"]["timeout"] == 1800

    def test_glp_awin_cgroup_executes_the_scheduler_held_verified_descriptor(
        self, cron_env, monkeypatch
    ):
        from cron import scheduler as sched_mod
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "affiliate_portal_apply_daily.sh"
        trusted = b"#!/usr/bin/env bash\nprintf 'trusted\\n'\n"
        script.write_bytes(trusted)
        digest = hashlib.sha256(trusted).hexdigest()
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            descriptor_path = next(value for value in argv if value.startswith("/proc/") and "/fd/" in value)
            script.write_text("#!/usr/bin/env bash\nprintf 'swapped\\n'\n")
            with open(descriptor_path, "rb") as descriptor_view:
                executed = descriptor_view.read()
            return SimpleNamespace(returncode=0, stdout=executed.decode(), stderr="")

        monkeypatch.setattr(sched_mod.subprocess, "run", fake_run)
        success, output = _run_job_script(
            str(script),
            job={
                "id": "52e2014a9579",
                "script_timeout_seconds": 1800,
                "script_sha256": digest,
            },
        )
        assert success is True
        assert output == trusted.decode().strip()

    @pytest.mark.parametrize("normal_parent_exit", [False, True])
    def test_real_systemd_run_command_contains_escaped_session_descendants(
        self, tmp_path, normal_parent_exit
    ):
        marker = tmp_path / f"scheduler-outer-{'normal' if normal_parent_exit else 'timeout'}"
        grandchild = (
            "import pathlib,time;time.sleep(.5);"
            f"pathlib.Path({str(marker)!r}).write_text('alive');time.sleep(60)"
        )
        child = (
            "import subprocess,sys,time;"
            f"subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True,"
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
            f"time.sleep({0 if normal_parent_exit else 60})"
        )
        unit = f"hermes-cron-script-pytest-{os.getpid()}-{time.time_ns()}"
        completed = subprocess.run([
            "/usr/bin/systemd-run", "--user", "--pipe", "--wait", "--collect", "--quiet",
            "--service-type=exec", f"--unit={unit}",
            "-p", "KillMode=control-group", "-p", "SendSIGKILL=yes",
            "-p", "TimeoutStopSec=.1s", "-p", "RuntimeMaxSec=.2s",
            "--", sys.executable, "-c", child,
        ], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
           check=False, timeout=10)
        assert completed.returncode == (0 if normal_parent_exit else 1)
        time.sleep(.7)
        assert not marker.exists()


class TestBuildJobPromptWithScript:
    """Test that script output is injected into the prompt."""

    def test_script_output_injected(self, cron_env):
        from cron.scheduler import _build_job_prompt

        script = cron_env / "scripts" / "data.py"
        script.write_text('print("new PR: #123 fix typo")\n')

        job = {
            "prompt": "Report any notable changes.",
            "script": str(script),
        }
        prompt = _build_job_prompt(job)
        assert "## Script Output" in prompt
        assert "new PR: #123 fix typo" in prompt
        assert "Report any notable changes." in prompt

    def test_script_error_injected(self, cron_env):
        from cron.scheduler import _build_job_prompt

        job = {
            "prompt": "Report status.",
            "script": "nonexistent_monitor.py",
        }
        prompt = _build_job_prompt(job)
        assert "## Script Error" in prompt
        assert "not found" in prompt.lower()
        assert "Report status." in prompt

    def test_no_script_unchanged(self, cron_env):
        from cron.scheduler import _build_job_prompt

        job = {"prompt": "Simple job."}
        prompt = _build_job_prompt(job)
        assert "## Script Output" not in prompt
        assert "Simple job." in prompt


class TestCronjobToolScript:
    """Test the cronjob tool's script parameter."""

    def test_create_persists_per_job_script_timeout(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        script = cron_env / "scripts" / "paced.py"
        script.write_text('print("ok")\n')

        result = json.loads(
            cronjob(
                action="create",
                schedule="every 1h",
                prompt="Monitor things",
                script="paced.py",
                script_timeout_seconds=17,
            )
        )

        assert result["success"] is True
        assert result["job"]["script_timeout_seconds"] == 17

    def test_clear_script(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        create_result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="some_script.py",
        ))
        job_id = create_result["job_id"]

        update_result = json.loads(cronjob(
            action="update",
            job_id=job_id,
            script="",
        ))
        assert update_result["success"] is True
        assert "script" not in update_result["job"]

    def test_list_shows_script(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="data_collector.py",
        )

        list_result = json.loads(cronjob(action="list"))
        assert list_result["success"] is True
        assert len(list_result["jobs"]) == 1
        assert list_result["jobs"][0]["script"] == "data_collector.py"


class TestScriptPathContainment:
    """Regression tests for path containment bypass in _run_job_script().

    Prior to the fix, absolute paths and ~-prefixed paths bypassed the
    scripts_dir containment check entirely, allowing arbitrary script
    execution through the cron system.
    """

    def test_absolute_path_outside_scripts_dir_blocked(self, cron_env):
        """Absolute paths outside ~/.hermes/scripts/ must be rejected."""
        from cron.scheduler import _run_job_script

        # Create a script outside the scripts dir
        outside_script = cron_env / "outside.py"
        outside_script.write_text('print("should not run")\n')

        success, output = _run_job_script(str(outside_script))
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()


    def test_tilde_path_blocked(self, cron_env):
        """~ prefixed paths must be rejected (expanduser bypasses check)."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("~/evil.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_tilde_traversal_blocked(self, cron_env):
        """~/../../../tmp/evil.py must be rejected."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("~/../../../tmp/evil.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_relative_traversal_still_blocked(self, cron_env):
        """../../etc/passwd style traversal must still be blocked."""
        from cron.scheduler import _run_job_script

        success, output = _run_job_script("../../etc/passwd")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()

    def test_relative_path_inside_scripts_dir_allowed(self, cron_env):
        """Relative paths within the scripts dir should still work."""
        from cron.scheduler import _run_job_script

        script = cron_env / "scripts" / "good.py"
        script.write_text('print("ok")\n')

        success, output = _run_job_script("good.py")
        assert success is True
        assert output == "ok"

    def test_subdirectory_inside_scripts_dir_allowed(self, cron_env):
        """Relative paths to subdirectories within scripts/ should work."""
        from cron.scheduler import _run_job_script

        subdir = cron_env / "scripts" / "monitors"
        subdir.mkdir()
        script = subdir / "check.py"
        script.write_text('print("sub ok")\n')

        success, output = _run_job_script("monitors/check.py")
        assert success is True
        assert output == "sub ok"


    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Symlinks require elevated privileges on Windows",
    )
    def test_symlink_escape_blocked(self, cron_env, tmp_path):
        """Symlinks pointing outside scripts/ must be rejected."""
        from cron.scheduler import _run_job_script

        # Create a script outside the scripts dir
        outside = tmp_path / "outside_evil.py"
        outside.write_text('print("escaped")\n')

        # Create a symlink inside scripts/ pointing outside
        link = cron_env / "scripts" / "sneaky.py"
        link.symlink_to(outside)

        success, output = _run_job_script("sneaky.py")
        assert success is False
        assert "blocked" in output.lower() or "outside" in output.lower()


class TestCronjobToolScriptValidation:
    """Test API-boundary validation of cron script paths in cronjob_tools."""


    def test_create_with_traversal_script_rejected(self, cron_env, monkeypatch):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        from tools.cronjob_tools import cronjob

        result = json.loads(cronjob(
            action="create",
            schedule="every 1h",
            prompt="Monitor things",
            script="../../etc/passwd",
        ))
        assert result["success"] is False
        assert "escapes" in result["error"].lower() or "traversal" in result["error"].lower()


class TestRunJobEnvVarCleanup:
    """Test that run_job() env vars are cleaned up even on early failure."""

    def test_env_vars_cleaned_on_early_error(self, cron_env, monkeypatch):
        """Origin env vars must be cleaned up even if run_job fails early."""
        # Ensure env vars are clean before test
        for key in (
            "HERMES_SESSION_PLATFORM",
            "HERMES_SESSION_CHAT_ID",
            "HERMES_SESSION_CHAT_NAME",
        ):
            monkeypatch.delenv(key, raising=False)

        # Build a job with origin info that will fail during execution
        # (no valid model, no API key — will raise inside try block)
        job = {
            "id": "test-envleak",
            "name": "env-leak-test",
            "prompt": "test",
            "schedule_display": "every 1h",
            "origin": {
                "platform": "telegram",
                "chat_id": "12345",
                "chat_name": "Test Chat",
            },
        }

        from cron.scheduler import run_job

        # Expect it to fail (no model/API key), but env vars must be cleaned
        try:
            run_job(job)
        except Exception:
            pass

        # Verify env vars were cleaned up by the finally block
        assert os.environ.get("HERMES_SESSION_PLATFORM") is None
        assert os.environ.get("HERMES_SESSION_CHAT_ID") is None
        assert os.environ.get("HERMES_SESSION_CHAT_NAME") is None


class TestF13InterpreterAuthority:
    def test_production_interpreter_pin_matches_exact_current_bytes(self):
        from cron import scheduler

        path = scheduler._F13_INTERPRETER_PATH
        assert path.is_file()
        value = path.lstat()
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == scheduler._F13_INTERPRETER_SHA256
        assert value.st_uid == scheduler._F13_INTERPRETER_UID
        assert value.st_gid == scheduler._F13_INTERPRETER_GID
        assert value.st_mode & 0o777 == scheduler._F13_INTERPRETER_MODE
        assert value.st_nlink == scheduler._F13_INTERPRETER_NLINK
        current = Path(path.anchor)
        for component in path.parts[1:]:
            current /= component
            assert not current.is_symlink()

    @staticmethod
    def _configure_interpreter(scheduler, tmp_path, monkeypatch):
        interpreter = tmp_path / "f13-python3.11"
        interpreter.write_bytes(b"fixture-interpreter-bytes")
        interpreter.chmod(0o700)
        monkeypatch.setattr(scheduler, "_F13_INTERPRETER_PATH", interpreter)
        monkeypatch.setattr(
            scheduler,
            "_F13_INTERPRETER_SHA256",
            hashlib.sha256(interpreter.read_bytes()).hexdigest(),
        )
        monkeypatch.setattr(scheduler, "_F13_INTERPRETER_UID", os.geteuid())
        monkeypatch.setattr(scheduler, "_F13_INTERPRETER_GID", os.getegid())
        monkeypatch.setattr(scheduler, "_F13_INTERPRETER_MODE", 0o700)
        monkeypatch.setattr(scheduler, "_F13_INTERPRETER_NLINK", 1)
        return interpreter

    def test_f13_uses_only_retained_hash_verified_interpreter_descriptor(
        self, cron_env, tmp_path, monkeypatch
    ):
        from cron import scheduler

        interpreter = self._configure_interpreter(scheduler, tmp_path, monkeypatch)
        script = cron_env / "scripts" / "f13_growth_retention.py"
        script.write_text("print('fixture')\n", encoding="utf-8")
        seen = []

        def fake_run(argv, **kwargs):
            seen.append((argv, kwargs))
            assert argv[0].startswith("/proc/self/fd/")
            assert Path(argv[0]).read_bytes() == interpreter.read_bytes()
            assert argv[1:3] == ["-I", "-S"]
            assert argv[3].startswith("/proc/self/fd/")
            descriptor = int(argv[0].rsplit("/", 1)[1])
            assert descriptor in kwargs["pass_fds"]
            assert len(kwargs["pass_fds"]) == 2
            script_descriptor = int(argv[3].rsplit("/", 1)[1])
            assert script_descriptor in kwargs["pass_fds"]
            assert Path(argv[3]).read_text(encoding="utf-8") == "print('fixture')\n"
            for key in scheduler._F13_PYTHON_STARTUP_ENV_KEYS:
                assert key not in kwargs["env"]
            return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

        monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
        ok, output = scheduler._run_job_script(
            str(script),
            job={
                "id": scheduler._F13_RETENTION_JOB_ID,
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            },
        )
        assert ok is True
        assert output == "ok"
        assert seen

    def test_f13_requires_exact_script_hash_before_subprocess(
        self, cron_env, monkeypatch
    ):
        from cron import scheduler

        script = cron_env / "scripts" / scheduler._F13_RETENTION_SCRIPT_NAME
        script.write_text("print('must-not-run')\n", encoding="utf-8")
        effects = []
        monkeypatch.setattr(
            scheduler.subprocess,
            "run",
            lambda *_args, **_kwargs: effects.append("subprocess"),
        )

        ok, output = scheduler._run_job_script(
            str(script), job={"id": scheduler._F13_RETENTION_JOB_ID}
        )

        assert ok is False
        assert "f13_exact_script_sha256_required" in output
        assert effects == []

    def test_f13_stale_script_registration_still_refuses_external_retention_run(
        self, cron_env, monkeypatch
    ):
        from cron import scheduler

        script = cron_env / "scripts" / scheduler._F13_RETENTION_SCRIPT_NAME
        script.write_text("print('must-not-run')\n", encoding="utf-8")
        effects = []
        monkeypatch.setattr(
            scheduler.subprocess,
            "run",
            lambda *_args, **_kwargs: effects.append("subprocess"),
        )

        ok, output = scheduler._run_job_script(
            str(script),
            job={
                "id": scheduler._F13_RETENTION_JOB_ID,
                "script_sha256": "0" * 64,
            },
        )

        assert ok is False
        assert "cron_script_hash_mismatch" in output
        assert effects == []

    def test_f13_real_hostile_sitecustomize_cannot_mark_or_emit_and_wrapper_runs(
        self, cron_env, tmp_path, monkeypatch
    ):
        from cron import scheduler

        hostile_dir = tmp_path / "hostile-pythonpath"
        hostile_dir.mkdir()
        marker = tmp_path / "hostile-sitecustomize-marker"
        (hostile_dir / "sitecustomize.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
            "print('HOSTILE_SITECUSTOMIZE_EXECUTED')\n",
            encoding="utf-8",
        )
        for key in scheduler._F13_PYTHON_STARTUP_ENV_KEYS:
            monkeypatch.setenv(key, str(hostile_dir))
        script = cron_env / "scripts" / scheduler._F13_RETENTION_SCRIPT_NAME
        script.write_text("print('BENIGN_F13_WRAPPER_EXECUTED')\n", encoding="utf-8")

        ok, output = scheduler._run_job_script(
            str(script),
            job={
                "id": scheduler._F13_RETENTION_JOB_ID,
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            },
        )

        assert ok is True
        assert output == "BENIGN_F13_WRAPPER_EXECUTED"
        assert "HOSTILE_SITECUSTOMIZE_EXECUTED" not in output
        assert not marker.exists()

    def test_non_f13_same_basename_preserves_existing_python_invocation_semantics(
        self, cron_env, monkeypatch
    ):
        from cron import scheduler

        script = cron_env / "scripts" / scheduler._F13_RETENTION_SCRIPT_NAME
        script.write_text("print('unrelated')\n", encoding="utf-8")
        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            return subprocess.CompletedProcess(argv, 0, stdout="unrelated\n", stderr="")

        monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
        ok, output = scheduler._run_job_script(
            str(script),
            job={
                "id": "unrelated-job",
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            },
        )

        assert ok is True
        assert output == "unrelated"
        assert captured["argv"][0] == sys.executable
        assert captured["argv"][1].startswith("/proc/self/fd/")
        assert captured["argv"][1:3] != ["-I", "-S"]

    @pytest.mark.parametrize("mutation", ["hash", "mode", "hardlink", "symlink"])
    def test_f13_interpreter_drift_blocks_before_subprocess(
        self, cron_env, tmp_path, monkeypatch, mutation
    ):
        from cron import scheduler

        interpreter = self._configure_interpreter(scheduler, tmp_path, monkeypatch)
        if mutation == "hash":
            monkeypatch.setattr(scheduler, "_F13_INTERPRETER_SHA256", "0" * 64)
        elif mutation == "mode":
            interpreter.chmod(0o722)
        elif mutation == "hardlink":
            os.link(interpreter, tmp_path / "interpreter-second-link")
        else:
            target = tmp_path / "interpreter-real"
            target.write_bytes(interpreter.read_bytes())
            target.chmod(0o700)
            interpreter.unlink()
            interpreter.symlink_to(target.name)
        script = cron_env / "scripts" / "f13_growth_retention.py"
        script.write_text("print('must-not-run')\n", encoding="utf-8")
        effects = []
        monkeypatch.setattr(
            scheduler.subprocess,
            "run",
            lambda *_args, **_kwargs: effects.append("subprocess"),
        )
        ok, output = scheduler._run_job_script(
            str(script),
            job={
                "id": scheduler._F13_RETENTION_JOB_ID,
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            },
        )
        assert ok is False
        assert "interpreter integrity" in output
        assert effects == []

    def test_f13_interpreter_path_swap_between_lstat_and_open_blocks(
        self, cron_env, tmp_path, monkeypatch
    ):
        from cron import scheduler

        interpreter = self._configure_interpreter(scheduler, tmp_path, monkeypatch)
        replacement = tmp_path / "f13-python-replacement"
        replacement.write_bytes(interpreter.read_bytes())
        replacement.chmod(0o700)
        script = cron_env / "scripts" / "f13_growth_retention.py"
        script.write_text("print('must-not-run')\n", encoding="utf-8")
        real_open = scheduler.os.open
        swapped = {"value": False}

        def hostile_open(path, flags, *args, **kwargs):
            if Path(path) == interpreter and not swapped["value"]:
                swapped["value"] = True
                os.replace(replacement, interpreter)
            return real_open(path, flags, *args, **kwargs)

        effects = []
        monkeypatch.setattr(scheduler.os, "open", hostile_open)
        monkeypatch.setattr(
            scheduler.subprocess,
            "run",
            lambda *_args, **_kwargs: effects.append("subprocess"),
        )
        ok, output = scheduler._run_job_script(
            str(script),
            job={
                "id": scheduler._F13_RETENTION_JOB_ID,
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            },
        )
        assert ok is False
        assert "identity_drift" in output
        assert effects == []
