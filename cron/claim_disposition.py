"""Explicit, non-executing disposition of a paused direct run's UNKNOWN fire claim.

This is not stale-claim recovery or a retry API. The caller must first reconcile effects and
obtain the required independent approval, including the payload's complete work-lock set.
Intent is durable BEFORE the native jobs save. Once intent exists only inspection is allowed;
an uncertain save must never be replayed. The execution ledger is never updated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import sqlite3
import stat
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from cron import jobs
from hermes_constants import get_hermes_home


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _json(text):
    return json.loads(text, object_pairs_hook=_unique_object)


def _clock(value):
    result = datetime.fromisoformat(value)
    _require(result.tzinfo is not None, "timezone required")
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _home():
    home = get_hermes_home().resolve()
    _require(jobs._current_cron_store().jobs_file == home / "cron" / "jobs.json",
             "cron store/profile mismatch")
    return home


def _journal_path(home, execution_id):
    _require(isinstance(execution_id, str) and re.fullmatch(r"[0-9a-f]{32}", execution_id),
             "exact execution ID required")
    return home / "cron" / "claim-dispositions" / (execution_id + ".jsonl")


def _control_namespace(home, journal):
    """Admit only unaliased native control paths; return stable namespace identities.

    The authorized maintenance caller must exclude namespace writers throughout this
    operation. Path checks detect drift; they are not a defense against a hostile UID
    able to rename an ancestor between two syscalls.
    """
    cron = home / "cron"
    directories = [*reversed(home.parents), home, cron, journal.parent]
    files = [cron / "jobs.json", cron / "executions.db", journal]
    optional = {journal.parent, journal}
    # SQLite may open these itself. Reject aliases before it can touch another DB.
    for suffix in ("-wal", "-shm", "-journal"):
        path = cron / ("executions.db" + suffix)
        files.append(path)
        optional.add(path)
    identities = {}
    for path in directories + files:
        try:
            info = path.lstat()
        except FileNotFoundError:
            _require(path in optional, "missing control path")
            continue
        directory = path in directories
        _require(stat.S_ISDIR(info.st_mode) if directory else
                 stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                 "unsafe control path: " + str(path))
        # Sidecars legitimately come and go under SQLite's writer transaction.
        if directory or path in files[:3]:
            identities[path] = (info.st_dev, info.st_ino)
    return identities


def _check_control_namespace(home, journal, before, *, saved=False):
    current = _control_namespace(home, journal)
    for path, identity in before.items():
        if saved and path == home / "cron" / "jobs.json":
            continue  # The native atomic save intentionally replaces this inode.
        _require(current.get(path) == identity, "control namespace moved")
    return current


def disposition_paths(cron_dir):
    """Retain every exact intent, including damaged or unacknowledged ones."""
    directory = Path(cron_dir) / "claim-dispositions"
    try:
        paths = list(directory.iterdir())
    except FileNotFoundError:
        return []
    _require(not directory.is_symlink(), "pending disposition: unsafe journal directory")
    return [path for path in paths if re.fullmatch(r"[0-9a-f]{32}\.jsonl", path.name)]


def require_no_pending_disposition(job_id, cron_dir):
    """Native admission guard; caller holds the ledger writer or the job fire fence."""
    for path in disposition_paths(cron_dir):
        try:
            _require(not path.is_symlink(), "unsafe journal")
            events = [_json(line) for line in path.read_text().splitlines()]
            intent = events[0]
            _require(intent["event"] == "intent" and
                     intent["request"]["execution"]["id"] == path.stem, "invalid intent")
            if intent["request"]["job"]["id"] != job_id:
                continue
            _require(len(events) == 2 and events[1]["event"] == "verified" and
                     events[1]["request_sha256"] == _digest(intent["request"]), "missing acknowledgement")
        except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
            raise ValueError("pending disposition: native admission held") from exc


@contextmanager
def _ledger(home):
    # No schema initialization, pruning, recovery, or terminal UPDATE, even on the read path.
    conn = sqlite3.connect((home / "cron" / "executions.db").as_uri() + "?mode=rw",
                           uri=True, timeout=1)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")  # prevent a new ledger attempt during the CAS
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _rows(conn, job_id):
    return [dict(row) for row in conn.execute("SELECT * FROM executions WHERE job_id=?", (job_id,))]


def _jobs():
    # Deliberately bypass load_jobs(): its repair path may write before admission.
    path = jobs._current_cron_store().jobs_file
    data = _json(path.read_text())
    _require(isinstance(data, dict) and isinstance(data.get("jobs"), list)
             and set(data) <= {"jobs", "updated_at"}, "invalid jobs store")
    rows = data["jobs"]
    _require(all(isinstance(row, dict) and isinstance(row.get("id"), str) for row in rows),
             "invalid job record")
    _require(len({row["id"] for row in rows}) == len(rows), "duplicate job ID")
    return rows


def _validate_window(request):
    created, expires = _clock(request["created_at"]), _clock(request["expires_at"])
    finished = _clock(request["execution"]["finished_at"])
    now = datetime.now(timezone.utc)
    _require(finished <= created <= now < expires and 0 < (expires - created).total_seconds() <= 900,
             "stale or invalid evidence window")


def _validate(request, rows, ledger_rows):
    job, execution = request["job"], request["execution"]
    _require(next((row for row in rows if row["id"] == job["id"]), None) == job,
             "job preimage moved")
    _require(next((row for row in ledger_rows if row["id"] == execution["id"]), None) == execution,
             "execution preimage moved")
    _require(execution["job_id"] == job["id"] and execution["status"] == "unknown"
             and execution["source"] == "direct" and execution["handoff_pending"] == 0
             and execution["handoff_started_at"] is None, "exact UNKNOWN direct execution required")
    _require(job.get("enabled") is False and job.get("state") == "paused"
             and job.get("run_claim") is None
             and job.get("schedule", {}).get("kind") in {"cron", "interval"},
             "paused recurring job without run claim required")
    claim = job.get("fire_claim")
    _require(isinstance(claim, dict), "retained fire claim required")
    parts = str(claim.get("by", "")).split(":")
    _require(len(parts) == 3 and parts[0] == socket.gethostname()
             and parts[1] == str(execution["pid"]) and re.fullmatch(r"[0-9a-f]{32}", parts[2]),
             "fire owner/execution identity mismatch")
    _require(type(execution["pid"]) is int and execution["pid"] > 0
             and type(execution["process_started_at"]) is int and execution["process_started_at"] > 0,
             "owner PID/start identity required")
    claimed, finished = _clock(execution["claimed_at"]), _clock(execution["finished_at"])
    _require(claimed <= _clock(claim["at"]) <= finished, "claim outside exact execution")
    for row in ledger_rows:
        _require(row["status"] not in {"claimed", "running"}, "active execution exists")
        if row["id"] != execution["id"]:
            _require(_clock(row["claimed_at"]) < claimed, "newer or ambiguous execution exists")
    # Conservative: refuse even a recycled PID, mismatched start time or expired TTL.
    # Existing stale-recovery helpers intentionally accept live/wedged owners; NOT safe here.
    try:
        os.kill(execution["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("owner PID still exists")
    from cron.scheduler import is_job_running
    _require(not is_job_running(job["id"], home=Path(request["home"])), "in-process work active")
    _validate_window(request)
    _require(isinstance(request.get("reason"), str) and request["reason"].strip(), "reason required")
    _require(isinstance(request.get("evidence"), list) and request["evidence"], "evidence required")
    for item in request["evidence"]:
        path = Path(item["path"])
        _require(path.is_absolute() and path.is_file() and not path.is_symlink(), "invalid evidence path")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        _require(digest == item["sha256"], "evidence moved")
    for item in request["work_locks"]:
        current = Path(item["path"]).lstat()
        _require(stat.S_ISREG(current.st_mode) and
                 (current.st_dev, current.st_ino) == (item["device"], item["inode"]),
                 "work lock identity moved")


@contextmanager
def _work_locks(request):
    # Retention's actual common lock also excludes orphaned payloads after owner death.
    # The reviewed caller supplies the complete lock set; no arbitrary command/probe is run.
    import fcntl

    locks = request.get("work_locks")
    _require(isinstance(locks, list) and locks, "payload work locks required")
    with ExitStack() as stack:
        for item in sorted(locks, key=lambda item: item["path"]):
            path = Path(item["path"])
            _require(path.is_absolute(), "absolute work lock required")
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            before = os.fstat(fd)
            _require(stat.S_ISREG(before.st_mode) and
                     (before.st_dev, before.st_ino) == (item["device"], item["inode"]),
                     "work lock identity moved")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _require((path.stat().st_dev, path.stat().st_ino) == (before.st_dev, before.st_ino),
                     "work lock replaced")
        yield


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append(path, event, *, first=False):
    if first:
        path.parent.mkdir(mode=0o700, exist_ok=True)
        _fsync_dir(path.parent.parent)
    flags = os.O_WRONLY | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if first else os.O_APPEND)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_dir(path.parent)


def dispose_claim(request: dict, *, apply: bool = False) -> dict:
    """Preview by default; apply clears ONLY fire_claim, once, with durable custody.

    Linux/POSIX flock only. An authorized caller must hold any additional installation /
    source-reader exclusion required by its payload. Work locks must be the actual locks
    used by all payload workers, not substitute lockfiles created for this operation.
    """
    request = copy.deepcopy(request)
    home = _home()
    _require(request["home"] == str(home), "request/profile mismatch")
    journal = _journal_path(home, request["execution"]["id"])
    namespace = _control_namespace(home, journal)
    with jobs._fire_job_lock(request["job"]["id"]) as held:
        _require(held, "fire fence unavailable")
        _require(not journal.exists(), "disposition intent exists; inspect, never retry")
        with jobs._jobs_lock(require_cross_process=True), _ledger(home) as conn, _work_locks(request):
            _require(not journal.exists(), "disposition intent exists; inspect, never retry")
            namespace = _check_control_namespace(home, journal, namespace)
            jobs._require_disposition_lock_custody(request["job"]["id"])
            rows = _jobs()
            ledger_rows = _rows(conn, request["job"]["id"])
            _validate(request, rows, ledger_rows)
            after = copy.deepcopy(rows)
            next(row for row in after if row["id"] == request["job"]["id"])["fire_claim"] = None
            _check_control_namespace(home, journal, namespace)
            jobs._require_disposition_lock_custody(request["job"]["id"])
            _validate_window(request)
            if not apply:
                return {"status": "ready", "request_sha256": _digest(request)}
            intent = {"event": "intent", "at": datetime.now(timezone.utc).isoformat(),
                      "request": request, "before_jobs_sha256": _digest(rows),
                      "after_jobs_sha256": _digest(after)}
            _append(journal, intent, first=True)
            namespace = _check_control_namespace(home, journal, namespace)
            jobs._require_disposition_lock_custody(request["job"]["id"])
            # Recheck after durable intent: a slow fsync is not permission to use stale evidence.
            _require(_jobs() == rows, "jobs preimage moved")
            _validate(request, rows, _rows(conn, request["job"]["id"]))
            _check_control_namespace(home, journal, namespace)
            jobs._require_disposition_lock_custody(request["job"]["id"])
            # No rollback/retry after this point, including if save returns an exception.
            # Native saver can mutate its input; retain an independent comparison oracle.
            save_rows = copy.deepcopy(after)
            # Last admission check: hashing, fsync and lock/path checks may block.
            _validate_window(request)
            jobs.save_jobs(save_rows)
            _check_control_namespace(home, journal, namespace, saved=True)
            jobs._require_disposition_lock_custody(request["job"]["id"])
            _fsync_dir(home / "cron")
            _require(_jobs() == after and _rows(conn, request["job"]["id"]) == ledger_rows,
                     "save/readback uncertain; inspect disposition, never retry")
            _check_control_namespace(home, journal, namespace, saved=True)
            jobs._require_disposition_lock_custody(request["job"]["id"])
            _append(journal, {"event": "verified", "at": datetime.now(timezone.utc).isoformat(),
                              "request_sha256": _digest(request)})
            return {"status": "disposed", "journal": str(journal), "execution_status": "unknown"}


def inspect_disposition(execution_id: str) -> dict:
    """Read back an existing intent without saving, resuming or retrying anything.

    'applied' is a CURRENT exact readback, not a reconstructed success receipt. Movement in
    unrelated jobs also yields 'uncertain' rather than hiding a concurrent registry write.
    """
    home = _home()
    journal = _journal_path(home, execution_id)
    namespace = _control_namespace(home, journal)
    events = [_json(line) for line in journal.read_text().splitlines()]
    _require(bool(events), "incomplete disposition intent")
    intent = events[0]
    request = intent["request"]
    _require(intent["event"] == "intent" and request["home"] == str(home)
             and request["execution"]["id"] == execution_id, "invalid disposition intent")
    with jobs._fire_job_lock(request["job"]["id"]) as held:
        _require(held, "fire fence unavailable")
        with jobs._jobs_lock(require_cross_process=True), _ledger(home) as conn:
            current = _digest(_jobs())
            ledger = _rows(conn, request["job"]["id"])
            exact = next((r for r in ledger if r["id"] == execution_id), None) == request["execution"]
            status = "uncertain"
            if exact and current == intent["after_jobs_sha256"]:
                status = "applied"
            elif exact and current == intent["before_jobs_sha256"]:
                status = "not_applied"
            _check_control_namespace(home, journal, namespace)
            jobs._require_disposition_lock_custody(request["job"]["id"])
            return {"status": status, "journal": str(journal), "retry_allowed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--request", type=Path, help="Reviewed full-preimage/evidence JSON; preview by default")
    target.add_argument("--inspect", help="Exact execution ID; read back existing intent only")
    parser.add_argument("--apply", action="store_true", help="Dispose once; never run or resume")
    args = parser.parse_args(argv)
    if args.inspect and args.apply:
        parser.error("--inspect cannot be combined with --apply")
    try:
        result = (inspect_disposition(args.inspect) if args.inspect else
                  dispose_claim(_json(args.request.read_text()), apply=args.apply))
    except (ValueError, RuntimeError, OSError, sqlite3.Error, KeyError, TypeError) as exc:
        print(json.dumps({"status": "refused_or_uncertain", "error": str(exc), "retry_allowed": False}))
        return 2
    print(json.dumps(result))
    return 0 if result["status"] in {"ready", "disposed", "applied", "not_applied"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
