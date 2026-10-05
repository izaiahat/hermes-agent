"""Retained fork script/interpreter pins, ported across the scheduler split.

Keep this separate from invocation, cancellation and profile-env policy. No
registered job is run here; prepare() only pins immutable invocation inputs.
Historical authority: audited native-cutover checkpoint 96198841a570.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import stat
import sys
from pathlib import Path

logger = logging.getLogger("cron.scheduler")
_SCRIPT_HASH_ENFORCED_JOBS = {
    "69a28f878ce3": "f13_growth_retention.py",
    "f3d2db07d83f": "f12_generator_archiver.py",
    "e7499917b657": "hermes_state_db_prune.sh",
    "2dd6ae1a4db9": "affiliate_auto_apply_daily.sh",
    "52e2014a9579": "affiliate_portal_apply_daily.sh",
    "7a269c665b12": "awin_auth_liveness_preflight.sh",
    "5a2cba16e057": "affiliate_multi_network_portal_apply_daily.sh",
    "dc3649d3c84c": "glp_relationship_hygiene_daily.sh",
    "f0c6a95a371c": "glp_affiliate_reconciliation_honesty.sh",
    "567f20134cc2": "glp_price_index_weekly.sh",
}
_F13_RETENTION_JOB_ID = "69a28f878ce3"
_F13_RETENTION_SCRIPT_NAME = "f13_growth_retention.py"
_F13_INTERPRETER_PATH = Path(
    "/home/ubuntu/.hermes/hermes-agent/.hermes-runtime/python/"
    "generation-1785217502-1419311-18b71f39/cpython-3.11.15-linux-x86_64-gnu/"
    "bin/python3.11"
)
_F13_INTERPRETER_SHA256 = "8deffe5dd9ebcf98a062917a4e73bb8fbb7d5846f83dec01fb7506fd5d41c54e"
_F13_INTERPRETER_UID = 1000
_F13_INTERPRETER_GID = 1000
_F13_INTERPRETER_MODE = 0o755
_F13_INTERPRETER_NLINK = 1
_F13_INTERPRETER_MAX_BYTES = 64 * 1024 * 1024
_F13_PYTHON_STARTUP_ENV_KEYS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE")


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_uid, value.st_gid,
            stat.S_IMODE(value.st_mode), value.st_nlink, value.st_size, value.st_mtime_ns)


def _open_f13_interpreter_descriptor() -> int:
    """Hash and metadata-pin the descriptor, refusing symlinks or path swaps."""
    path = _F13_INTERPRETER_PATH
    if not path.is_absolute() or Path(os.path.normpath(str(path))) != path:
        raise RuntimeError("f13_interpreter_path_noncanonical")
    current = Path(path.anchor)
    value = None
    for component in path.parts[1:]:
        current /= component
        value = current.lstat()
        if stat.S_ISLNK(value.st_mode):
            raise RuntimeError("f13_interpreter_symlink_component")
        if current != path and not stat.S_ISDIR(value.st_mode):
            raise RuntimeError("f13_interpreter_parent_not_directory")
    if value is None or not stat.S_ISREG(value.st_mode):
        raise RuntimeError("f13_interpreter_regular_file_required")
    if (value.st_uid != _F13_INTERPRETER_UID or value.st_gid != _F13_INTERPRETER_GID
            or stat.S_IMODE(value.st_mode) != _F13_INTERPRETER_MODE
            or value.st_nlink != _F13_INTERPRETER_NLINK
            or value.st_size > _F13_INTERPRETER_MAX_BYTES
            or not (stat.S_IMODE(value.st_mode) & 0o111)):
        raise RuntimeError("f13_interpreter_owner_mode_link_or_size_invalid")
    expected = _identity(value)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(descriptor, min(1024 * 1024, opened.st_size - offset), offset)
            if not chunk:
                raise RuntimeError("f13_interpreter_short_read")
            digest.update(chunk)
            offset += len(chunk)
        if _identity(opened) != expected or _identity(path.lstat()) != expected:
            raise RuntimeError("f13_interpreter_descriptor_path_identity_drift")
        if digest.hexdigest() != _F13_INTERPRETER_SHA256:
            raise RuntimeError("f13_interpreter_hash_mismatch")
        os.set_inheritable(descriptor, True)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _snapshot_script(path: Path):
    """Seal captured bytes: a same-inode overwrite after hashing cannot change execution."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    snapshot = None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise RuntimeError("cron_script_regular_file_required")
        data = bytearray()
        while chunk := os.read(descriptor, 1024 * 1024):
            data.extend(chunk)
        factory = getattr(os, "memfd_create", None)
        if factory is None:
            import ctypes
            native = ctypes.CDLL(None, use_errno=True).memfd_create
            native.argtypes = (ctypes.c_char_p, ctypes.c_uint)
            native.restype = ctypes.c_int

            def factory(name, flags):
                fd = int(native(name.encode(), flags))
                if fd < 0:
                    number = ctypes.get_errno()
                    raise OSError(number, os.strerror(number))
                return fd
        import fcntl
        snapshot = factory(f"hermes-cron-{path.name}", getattr(os, "MFD_ALLOW_SEALING", 0x0002))
        view = memoryview(data)
        while view:
            written = os.write(snapshot, view)
            if written <= 0:
                raise RuntimeError("cron_script_snapshot_short_write")
            view = view[written:]
        os.lseek(snapshot, 0, os.SEEK_SET)
        fcntl.fcntl(snapshot, getattr(fcntl, "F_ADD_SEALS", 1033),
                    getattr(fcntl, "F_SEAL_WRITE", 0x0008) | getattr(fcntl, "F_SEAL_GROW", 0x0004)
                    | getattr(fcntl, "F_SEAL_SHRINK", 0x0002) | getattr(fcntl, "F_SEAL_SEAL", 0x0001))
        os.set_inheritable(snapshot, True)
        return snapshot, hashlib.sha256(data).hexdigest()
    except BaseException:
        if snapshot is not None:
            os.close(snapshot)
        raise
    finally:
        os.close(descriptor)


def prepare(path: Path, job: dict | None, argv: list[str]):
    """Return invocation and owned FDs. Caller closes them after the child finishes."""
    job = job or {}
    expected = str(job.get("script_sha256") or "")
    job_id = str(job.get("id") or "")
    enforced = _SCRIPT_HASH_ENFORCED_JOBS.get(job_id) == path.name
    f13 = job_id == _F13_RETENTION_JOB_ID and path.name == _F13_RETENTION_SCRIPT_NAME
    descriptors = []
    try:
        if f13 and not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError("f13_exact_script_sha256_required")
        supported = sys.platform != "win32" and os.name == "posix" and Path("/proc/self/fd").is_dir()
        if supported:
            script_fd, actual = _snapshot_script(path)
            descriptors.append(script_fd)
            execution_path = f"/proc/self/fd/{script_fd}"
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if expected else ""
            execution_path = str(path)
            if f13 or enforced:
                raise RuntimeError("cron_script_descriptor_runtime_required")
        if expected and actual != expected:
            if enforced:
                raise RuntimeError("cron_script_hash_mismatch")
            logger.warning("Cron script registration drift for %s: registered %s, current %s; "
                           "running the current on-disk bytes", path.name, expected[:16], actual[:16])
        # Keep upstream Windows .pth/bootstrap argv intact; only replace the script arg.
        argv = [execution_path if arg == str(path) else arg for arg in argv]
        if f13:
            try:
                interpreter_fd = _open_f13_interpreter_descriptor()
            except Exception as exc:
                raise RuntimeError(f"f13 interpreter integrity: {exc}") from exc
            descriptors.append(interpreter_fd)
            argv = [f"/proc/self/fd/{interpreter_fd}", "-I", "-S", execution_path]
        return argv, tuple(descriptors), f13
    except BaseException:
        for fd in descriptors:
            os.close(fd)
        raise


def contained_argv(path: Path, job: dict | None, timeout: int, argv: list[str], descriptors, env):
    """Retain exact Awin lane's pinned descendant-safe cgroup, never start it here."""
    if not (str((job or {}).get("id") or "") == "52e2014a9579"
            and path.name == "affiliate_portal_apply_daily.sh" and timeout == 1800):
        return argv
    binary = Path("/usr/bin/systemd-run")
    digest = "dbc8b988a849d5c9d7ef2de7068a6f107021bc6c11e0d7864c73f373eef726a7"
    if not sys.platform.startswith("linux") or not binary.is_file() or hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
        raise RuntimeError("GLP Awin descendant-safe systemd runtime unavailable or drifted")
    if len(argv) <= 1 or not argv[1].startswith("/proc/self/fd/"):
        raise RuntimeError("cron_script_descriptor_runtime_required")
    fd = int(argv[1].rsplit("/", 1)[1])
    if fd not in descriptors:
        raise RuntimeError("cron_script_descriptor_not_in_pass_fds")
    service_argv = [argv[0], f"/proc/{os.getpid()}/fd/{fd}", *argv[2:]]
    env["HERMES_CRON_CGROUP_CONTAINED"] = "1"
    import threading
    import time
    unit = f"hermes-cron-script-{os.getpid()}-{threading.get_ident()}-{time.time_ns()}"
    return [str(binary), "--user", "--pipe", "--wait", "--collect", "--quiet", "--service-type=exec",
            f"--unit={unit}", "-p", "KillMode=control-group", "-p", "SendSIGKILL=yes",
            "-p", "TimeoutStopSec=5s", "-p", f"RuntimeMaxSec={max(1, timeout - 10)}s", "--", *service_argv]
