import subprocess
from types import SimpleNamespace
from agent import anthropic_credit_guard as guard


def test_snapshot_stdin(monkeypatch):
    def run(*args, **kwargs):
        assert kwargs['stdin'] is subprocess.DEVNULL
        assert kwargs['timeout'] == 45
        return SimpleNamespace(returncode=0, stdout='{"remaining_usd":6}')
    monkeypatch.setattr(guard.subprocess, 'run', run)
    assert guard.credit_snapshot()['dispatch_allowed'] is True
