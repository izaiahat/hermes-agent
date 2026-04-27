"""
Checkpoint Manager — Transparent filesystem snapshots via shadow git repos.

Creates automatic snapshots of working directories before file-mutating
operations (write_file, patch), triggered once per conversation turn.
Provides rollback to any previous checkpoint.

This is NOT a tool — the LLM never sees it.  It's transparent infrastructure
controlled by the ``checkpoints`` config flag or ``--checkpoints`` CLI flag.

Architecture:
    ~/.hermes/checkpoints/{sha256(abs_dir)[:16]}/   — shadow git repo
        HEAD, refs/, objects/                        — standard git internals
        HERMES_WORKDIR                               — original dir path
        info/exclude                                 — default excludes

The shadow repo uses GIT_DIR + GIT_WORK_TREE so no git state leaks
into the user's project directory.
"""

import hashlib
import io
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_BASE = get_hermes_home() / "checkpoints"

DEFAULT_EXCLUDES = [
    "node_modules/",
    "dist/",
    "build/",
    ".env",
    ".env.*",
    ".env.local",
    ".env.*.local",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "*.log",
    ".cache/",
    ".next/",
    ".nuxt/",
    "coverage/",
    ".pytest_cache/",
    ".venv/",
    "venv/",
    ".git/",
]

# Git subprocess timeout (seconds).
_GIT_TIMEOUT: int = max(10, min(60, int(os.getenv("HERMES_CHECKPOINT_TIMEOUT", "30"))))

# Max files to snapshot — skip huge directories to avoid slowdowns.
_MAX_FILES = 50_000

# Valid git commit hash pattern: 4–40 hex chars (short or full SHA-1/SHA-256).
_COMMIT_HASH_RE = re.compile(r'^[0-9a-fA-F]{4,64}$')


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_commit_hash(commit_hash: str) -> Optional[str]:
    """Validate a commit hash to prevent git argument injection.

    Returns an error string if invalid, None if valid.
    Values starting with '-' would be interpreted as git flags
    (e.g., '--patch', '-p') instead of revision specifiers.
    """
    if not commit_hash or not commit_hash.strip():
        return "Empty commit hash"
    if commit_hash.startswith("-"):
        return f"Invalid commit hash (must not start with '-'): {commit_hash!r}"
    if not _COMMIT_HASH_RE.match(commit_hash):
        return f"Invalid commit hash (expected 4-64 hex characters): {commit_hash!r}"
    return None


def _validate_file_path(file_path: str, working_dir: str) -> Optional[str]:
    """Validate a file path to prevent path traversal outside the working directory.

    Returns an error string if invalid, None if valid.
    """
    if not file_path or not file_path.strip():
        return "Empty file path"
    # Reject absolute paths — restore targets must be relative to the workdir
    if os.path.isabs(file_path):
        return f"File path must be relative, got absolute path: {file_path!r}"
    # Resolve and check containment within working_dir
    abs_workdir = _normalize_path(working_dir)
    resolved = (abs_workdir / file_path).resolve()
    try:
        resolved.relative_to(abs_workdir)
    except ValueError:
        return f"File path escapes the working directory via traversal: {file_path!r}"
    return None


# ---------------------------------------------------------------------------
# Shadow repo helpers
# ---------------------------------------------------------------------------

def _normalize_path(path_value: str) -> Path:
    """Return a canonical absolute path for checkpoint operations."""
    return Path(path_value).expanduser().resolve()


def _shadow_repo_path(working_dir: str) -> Path:
    """Deterministic shadow repo path: sha256(abs_path)[:16]."""
    abs_path = str(_normalize_path(working_dir))
    dir_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:16]
    return CHECKPOINT_BASE / dir_hash


def _git_env(shadow_repo: Path, working_dir: str) -> dict:
    """Build env dict that redirects git to the shadow repo.

    The shadow repo is internal Hermes infrastructure — it must NOT inherit
    the user's global or system git config.  User-level settings like
    ``commit.gpgsign = true``, signing hooks, or credential helpers would
    either break background snapshots or, worse, spawn interactive prompts
    (pinentry GUI windows) mid-session every time a file is written.

    Isolation strategy:
    * ``GIT_CONFIG_GLOBAL=<os.devnull>`` — ignore ``~/.gitconfig`` (git 2.32+).
    * ``GIT_CONFIG_SYSTEM=<os.devnull>`` — ignore ``/etc/gitconfig`` (git 2.32+).
    * ``GIT_CONFIG_NOSYSTEM=1`` — legacy belt-and-suspenders for older git.

    The shadow repo still has its own per-repo config (user.email, user.name,
    commit.gpgsign=false) set in ``_init_shadow_repo``.
    """
    normalized_working_dir = _normalize_path(working_dir)
    env = os.environ.copy()
    env["GIT_DIR"] = str(shadow_repo)
    env["GIT_WORK_TREE"] = str(normalized_working_dir)
    env.pop("GIT_INDEX_FILE", None)
    env.pop("GIT_NAMESPACE", None)
    env.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
    # Isolate the shadow repo from the user's global/system git config.
    # Prevents commit.gpgsign, hooks, aliases, credential helpers, etc. from
    # leaking into background snapshots.  Uses os.devnull for cross-platform
    # support (``/dev/null`` on POSIX, ``nul`` on Windows).
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _run_git(
    args: List[str],
    shadow_repo: Path,
    working_dir: str,
    timeout: int = _GIT_TIMEOUT,
    allowed_returncodes: Optional[Set[int]] = None,
) -> tuple:
    """Run a git command against the shadow repo.  Returns (ok, stdout, stderr).

    ``allowed_returncodes`` suppresses error logging for known/expected non-zero
    exits while preserving the normal ``ok = (returncode == 0)`` contract.
    Example: ``git diff --cached --quiet`` returns 1 when changes exist.
    """
    normalized_working_dir = _normalize_path(working_dir)
    if not normalized_working_dir.exists():
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg
    if not normalized_working_dir.is_dir():
        msg = f"working directory is not a directory: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg

    env = _git_env(shadow_repo, str(normalized_working_dir))
    cmd = ["git"] + list(args)
    allowed_returncodes = allowed_returncodes or set()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(normalized_working_dir),
        )
        ok = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if not ok and result.returncode not in allowed_returncodes:
            logger.error(
                "Git command failed: %s (rc=%d) stderr=%s",
                " ".join(cmd), result.returncode, stderr,
            )
        return ok, stdout, stderr
    except subprocess.TimeoutExpired:
        msg = f"git timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg, exc_info=True)
        return False, "", msg
    except FileNotFoundError as exc:
        missing_target = getattr(exc, "filename", None)
        if missing_target == "git":
            logger.error("Git executable not found: %s", " ".join(cmd), exc_info=True)
            return False, "", "git not found"
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command failed before execution: %s (%s)", " ".join(cmd), msg, exc_info=True)
        return False, "", msg
    except Exception as exc:
        logger.error("Unexpected git error running %s: %s", " ".join(cmd), exc, exc_info=True)
        return False, "", str(exc)


def _init_shadow_repo(shadow_repo: Path, working_dir: str) -> Optional[str]:
    """Initialise shadow repo if needed.  Returns error string or None."""
    if (shadow_repo / "HEAD").exists():
        return None

    shadow_repo.mkdir(parents=True, exist_ok=True)

    ok, _, err = _run_git(["init"], shadow_repo, working_dir)
    if not ok:
        return f"Shadow repo init failed: {err}"

    _run_git(["config", "user.email", "hermes@local"], shadow_repo, working_dir)
    _run_git(["config", "user.name", "Hermes Checkpoint"], shadow_repo, working_dir)
    # Explicitly disable commit/tag signing in the shadow repo.  _git_env
    # already isolates from the user's global config, but writing these into
    # the shadow's own config is belt-and-suspenders — it guarantees the
    # shadow repo is correct even if someone inspects or runs git against it
    # directly (without the GIT_CONFIG_* env vars).
    _run_git(["config", "commit.gpgsign", "false"], shadow_repo, working_dir)
    _run_git(["config", "tag.gpgSign", "false"], shadow_repo, working_dir)

    info_dir = shadow_repo / "info"
    info_dir.mkdir(exist_ok=True)
    (info_dir / "exclude").write_text(
        "\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8"
    )

    (shadow_repo / "HERMES_WORKDIR").write_text(
        str(_normalize_path(working_dir)) + "\n", encoding="utf-8"
    )

    logger.debug("Initialised checkpoint repo at %s for %s", shadow_repo, working_dir)
    return None


def _dir_file_count(path: str) -> int:
    """Quick file count estimate (stops early if over _MAX_FILES)."""
    count = 0
    try:
        for _ in Path(path).rglob("*"):
            count += 1
            if count > _MAX_FILES:
                return count
    except (PermissionError, OSError):
        pass
    return count


def _shadow_repo_size_bytes(shadow_repo: Path) -> int:
    """Best-effort recursive size calculation for a shadow repo."""
    total = 0
    try:
        for child in shadow_repo.rglob("*"):
            try:
                if child.is_file():
                    total += child.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _clear_worktree_except_git(worktree: Path) -> None:
    """Remove all files from a temp worktree except its ``.git`` directory."""
    for child in worktree.iterdir():
        if child.name == ".git":
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            shutil.rmtree(child, ignore_errors=True)


def _touch_tree_contents(worktree: Path) -> None:
    """Bump mtimes after archive extraction to avoid racy-git false clean reads."""
    now = time.time()
    for child in worktree.rglob("*"):
        try:
            os.utime(child, (now, now), follow_symlinks=False)
        except OSError:
            continue


def _rebuild_shadow_repo(
    source_shadow_repo: Path,
    working_dir: str,
    commit_hashes: List[str],
) -> Optional[Path]:
    """Rebuild a shadow repo containing only ``commit_hashes`` in order.

    Returns a path to the rebuilt shadow repo on success, or ``None`` on any
    failure. The rebuilt repo is created in a temporary directory and must be
    moved into place by the caller.
    """
    if not commit_hashes:
        return None

    tmp_root = Path(tempfile.mkdtemp(prefix=f"{source_shadow_repo.name}-prune-", dir=str(source_shadow_repo.parent)))
    temp_worktree = tmp_root / "worktree"
    temp_worktree.mkdir(parents=True, exist_ok=True)
    temp_shadow = tmp_root / "shadow"

    init_err = _init_shadow_repo(temp_shadow, str(temp_worktree))
    if init_err:
        shutil.rmtree(tmp_root, ignore_errors=True)
        return None

    for commit_hash in commit_hashes:
        _clear_worktree_except_git(temp_worktree)

        archive = subprocess.run(
            ["git", "archive", commit_hash],
            cwd=str(_normalize_path(working_dir)),
            env=_git_env(source_shadow_repo, working_dir),
            capture_output=True,
            timeout=_GIT_TIMEOUT * 4,
        )
        if archive.returncode != 0:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None

        try:
            with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r:") as tar:
                tar.extractall(temp_worktree)
            _touch_tree_contents(temp_worktree)
        except (tarfile.TarError, OSError):
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None

        ok, meta, _ = _run_git(
            ["show", "-s", "--format=%aI%x00%B", commit_hash],
            source_shadow_repo,
            working_dir,
        )
        if not ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None
        author_date, _, message = meta.partition("\x00")

        ok, _, _ = _run_git(["add", "-A"], temp_shadow, str(temp_worktree))
        if not ok:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None

        env = _git_env(temp_shadow, str(temp_worktree))
        env["GIT_AUTHOR_DATE"] = author_date.strip() or time.strftime("%Y-%m-%dT%H:%M:%S%z")
        env["GIT_COMMITTER_DATE"] = author_date.strip() or env["GIT_AUTHOR_DATE"]
        commit = subprocess.run(
            ["git", "commit", "--allow-empty-message", "-F", "-", "--no-gpg-sign"],
            cwd=str(_normalize_path(str(temp_worktree))),
            env=env,
            input=message,
            text=True,
            capture_output=True,
            timeout=_GIT_TIMEOUT * 4,
        )
        if commit.returncode != 0:
            shutil.rmtree(tmp_root, ignore_errors=True)
            return None

    _run_git(["gc", "--prune=now", "--aggressive"], temp_shadow, str(temp_worktree))
    return temp_shadow


def _replace_shadow_repo(shadow_repo: Path, replacement_repo: Path, working_dir: str) -> bool:
    """Atomically-ish replace ``shadow_repo`` with ``replacement_repo`` contents."""
    replacement = shadow_repo.parent / f".{shadow_repo.name}.replacement"
    backup = shadow_repo.parent / f".{shadow_repo.name}.backup"

    shutil.rmtree(replacement, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    shutil.copytree(replacement_repo, replacement)
    (replacement / "HERMES_WORKDIR").write_text(
        str(_normalize_path(working_dir)) + "\n", encoding="utf-8"
    )

    try:
        if shadow_repo.exists():
            shadow_repo.rename(backup)
        replacement.rename(shadow_repo)
        shutil.rmtree(backup, ignore_errors=True)
        return True
    except OSError:
        if shadow_repo.exists():
            shutil.rmtree(shadow_repo, ignore_errors=True)
        if backup.exists() and not shadow_repo.exists():
            backup.rename(shadow_repo)
        shutil.rmtree(replacement, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages automatic filesystem checkpoints.

    Designed to be owned by AIAgent.  Call ``new_turn()`` at the start of
    each conversation turn and ``ensure_checkpoint(dir, reason)`` before
    any file-mutating tool call.  The manager deduplicates so at most one
    snapshot is taken per directory per turn.

    Parameters
    ----------
    enabled : bool
        Master switch (from config / CLI flag).
    max_snapshots : int
        Keep at most this many checkpoints per directory.
    max_total_bytes : int
        If the shadow repo still exceeds this size after commit-count pruning,
        drop oldest checkpoints until the repo is at or under the cap, always
        preserving the most recent checkpoint.
    """

    def __init__(self, enabled: bool = False, max_snapshots: int = 10, max_total_bytes: int = 1_000_000_000):
        self.enabled = enabled
        self.max_snapshots = max(1, int(max_snapshots or 10))
        self.max_total_bytes = max(0, int(max_total_bytes or 0))
        self._checkpointed_dirs: Set[str] = set()
        self._git_available: Optional[bool] = None  # lazy probe

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def new_turn(self) -> None:
        """Reset per-turn dedup.  Call at the start of each agent iteration."""
        self._checkpointed_dirs.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        """Take a checkpoint if enabled and not already done this turn.

        Returns True if a checkpoint was taken, False otherwise.
        Never raises — all errors are silently logged.
        """
        if not self.enabled:
            return False

        # Lazy git probe
        if self._git_available is None:
            self._git_available = shutil.which("git") is not None
            if not self._git_available:
                logger.debug("Checkpoints disabled: git not found")
        if not self._git_available:
            return False

        abs_dir = str(_normalize_path(working_dir))

        # Skip root, home, and other overly broad directories
        if abs_dir in ("/", str(Path.home())):
            logger.debug("Checkpoint skipped: directory too broad (%s)", abs_dir)
            return False

        # Already checkpointed this turn?
        if abs_dir in self._checkpointed_dirs:
            return False

        self._checkpointed_dirs.add(abs_dir)

        try:
            return self._take(abs_dir, reason)
        except Exception as e:
            logger.debug("Checkpoint failed (non-fatal): %s", e)
            return False

    def list_checkpoints(self, working_dir: str) -> List[Dict]:
        """List available checkpoints for a directory.

        Returns a list of dicts with keys: hash, short_hash, timestamp, reason,
        files_changed, insertions, deletions.  Most recent first.
        """
        abs_dir = str(_normalize_path(working_dir))
        shadow = _shadow_repo_path(abs_dir)

        if not (shadow / "HEAD").exists():
            return []

        ok, stdout, _ = _run_git(
            ["log", "--format=%H|%h|%aI|%s", "-n", str(self.max_snapshots)],
            shadow, abs_dir,
        )

        if not ok or not stdout:
            return []

        results = []
        for line in stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entry = {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "timestamp": parts[2],
                    "reason": parts[3],
                    "files_changed": 0,
                    "insertions": 0,
                    "deletions": 0,
                }
                # Get diffstat for this commit
                stat_ok, stat_out, _ = _run_git(
                    ["diff", "--shortstat", f"{parts[0]}~1", parts[0]],
                    shadow, abs_dir,
                    allowed_returncodes={128, 129},  # first commit has no parent
                )
                if stat_ok and stat_out:
                    self._parse_shortstat(stat_out, entry)
                results.append(entry)
        return results

    @staticmethod
    def _parse_shortstat(stat_line: str, entry: Dict) -> None:
        """Parse git --shortstat output into entry dict."""
        m = re.search(r'(\d+) file', stat_line)
        if m:
            entry["files_changed"] = int(m.group(1))
        m = re.search(r'(\d+) insertion', stat_line)
        if m:
            entry["insertions"] = int(m.group(1))
        m = re.search(r'(\d+) deletion', stat_line)
        if m:
            entry["deletions"] = int(m.group(1))

    def diff(self, working_dir: str, commit_hash: str) -> Dict:
        """Show diff between a checkpoint and the current working tree.

        Returns dict with success, diff text, and stat summary.
        """
        # Validate commit_hash to prevent git argument injection
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))
        shadow = _shadow_repo_path(abs_dir)

        if not (shadow / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        # Verify the commit exists
        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], shadow, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found"}

        # Stage current state to compare against checkpoint
        _run_git(["add", "-A"], shadow, abs_dir, timeout=_GIT_TIMEOUT * 2)

        # Get stat summary: checkpoint vs current working tree
        ok_stat, stat_out, _ = _run_git(
            ["diff", "--stat", commit_hash, "--cached"],
            shadow, abs_dir,
        )

        # Get actual diff (limited to avoid terminal flood)
        ok_diff, diff_out, _ = _run_git(
            ["diff", commit_hash, "--cached", "--no-color"],
            shadow, abs_dir,
        )

        # Unstage to avoid polluting the shadow repo index
        _run_git(["reset", "HEAD", "--quiet"], shadow, abs_dir)

        if not ok_stat and not ok_diff:
            return {"success": False, "error": "Could not generate diff"}

        return {
            "success": True,
            "stat": stat_out if ok_stat else "",
            "diff": diff_out if ok_diff else "",
        }

    def restore(self, working_dir: str, commit_hash: str, file_path: str = None) -> Dict:
        """Restore files to a checkpoint state.

        Uses ``git checkout <hash> -- .`` (or a specific file) which restores
        tracked files without moving HEAD — safe and reversible.

        Parameters
        ----------
        file_path : str, optional
            If provided, restore only this file instead of the entire directory.

        Returns dict with success/error info.
        """
        # Validate commit_hash to prevent git argument injection
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))

        # Validate file_path to prevent path traversal outside the working dir
        if file_path:
            path_err = _validate_file_path(file_path, abs_dir)
            if path_err:
                return {"success": False, "error": path_err}

        shadow = _shadow_repo_path(abs_dir)

        if not (shadow / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        # Verify the commit exists
        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], shadow, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found", "debug": err or None}

        # Take a checkpoint of current state before restoring (so you can undo the undo)
        self._take(abs_dir, f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

        # Restore — full directory or single file
        restore_target = file_path if file_path else "."
        ok, stdout, err = _run_git(
            ["checkout", commit_hash, "--", restore_target],
            shadow, abs_dir, timeout=_GIT_TIMEOUT * 2,
        )

        if not ok:
            return {"success": False, "error": f"Restore failed: {err}", "debug": err or None}

        # Get info about what was restored
        ok2, reason_out, _ = _run_git(
            ["log", "--format=%s", "-1", commit_hash], shadow, abs_dir,
        )
        reason = reason_out if ok2 else "unknown"

        result = {
            "success": True,
            "restored_to": commit_hash[:8],
            "reason": reason,
            "directory": abs_dir,
        }
        if file_path:
            result["file"] = file_path
        return result

    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve a file path to its working directory for checkpointing.

        Walks up from the file's parent to find a reasonable project root
        (directory containing .git, pyproject.toml, package.json, etc.).
        Falls back to the file's parent directory.
        """
        path = _normalize_path(file_path)
        if path.is_dir():
            candidate = path
        else:
            candidate = path.parent

        # Walk up looking for project root markers
        markers = {".git", "pyproject.toml", "package.json", "Cargo.toml",
                    "go.mod", "Makefile", "pom.xml", ".hg", "Gemfile"}
        check = candidate
        while check != check.parent:
            if any((check / m).exists() for m in markers):
                return str(check)
            check = check.parent

        # No project root found — use the file's parent
        return str(candidate)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _take(self, working_dir: str, reason: str) -> bool:
        """Take a snapshot.  Returns True on success."""
        shadow = _shadow_repo_path(working_dir)

        # Init if needed
        err = _init_shadow_repo(shadow, working_dir)
        if err:
            logger.debug("Checkpoint init failed: %s", err)
            return False

        # Quick size guard — don't try to snapshot enormous directories
        if _dir_file_count(working_dir) > _MAX_FILES:
            logger.debug("Checkpoint skipped: >%d files in %s", _MAX_FILES, working_dir)
            return False

        # Stage everything
        ok, _, err = _run_git(
            ["add", "-A"], shadow, working_dir, timeout=_GIT_TIMEOUT * 2,
        )
        if not ok:
            logger.debug("Checkpoint git-add failed: %s", err)
            return False

        # Check if there's anything to commit
        ok_diff, diff_out, _ = _run_git(
            ["diff", "--cached", "--quiet"],
            shadow,
            working_dir,
            allowed_returncodes={1},
        )
        if ok_diff:
            # No changes to commit
            logger.debug("Checkpoint skipped: no changes in %s", working_dir)
            return False

        # Commit.  ``--no-gpg-sign`` inline covers shadow repos created before
        # the commit.gpgsign=false config was added to _init_shadow_repo — so
        # users with existing checkpoints never hit a GPG pinentry popup.
        ok, _, err = _run_git(
            ["commit", "-m", reason, "--allow-empty-message", "--no-gpg-sign"],
            shadow, working_dir, timeout=_GIT_TIMEOUT * 2,
        )
        if not ok:
            logger.debug("Checkpoint commit failed: %s", err)
            return False

        logger.debug("Checkpoint taken in %s: %s", working_dir, reason)

        # Prune old snapshots
        self._prune(shadow, working_dir)

        return True

    def _prune(self, shadow_repo: Path, working_dir: str) -> None:
        """Enforce checkpoint count and size limits on a shadow repo."""
        ok, stdout, _ = _run_git(
            ["rev-list", "--reverse", "HEAD"], shadow_repo, working_dir,
        )
        if not ok:
            return

        commits = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not commits:
            return

        repo_bytes = _shadow_repo_size_bytes(shadow_repo)
        within_count = len(commits) <= self.max_snapshots
        within_size = self.max_total_bytes <= 0 or repo_bytes <= self.max_total_bytes
        if within_count and within_size:
            return

        keep_hashes = commits[-min(len(commits), self.max_snapshots):]
        while keep_hashes:
            rebuilt = _rebuild_shadow_repo(shadow_repo, working_dir, keep_hashes)
            if rebuilt is None:
                logger.warning(
                    "Checkpoint prune skipped: failed to rebuild shadow repo for %s",
                    working_dir,
                )
                return

            rebuilt_bytes = _shadow_repo_size_bytes(rebuilt)
            size_ok = self.max_total_bytes <= 0 or rebuilt_bytes <= self.max_total_bytes
            replace_ok = size_ok or len(keep_hashes) == 1
            temp_root = rebuilt.parent

            if replace_ok:
                if _replace_shadow_repo(shadow_repo, rebuilt, working_dir):
                    logger.info(
                        "Checkpoint prune enforced for %s: commits %d→%d, bytes %d→%d",
                        working_dir,
                        len(commits),
                        len(keep_hashes),
                        repo_bytes,
                        rebuilt_bytes,
                    )
                else:
                    logger.warning(
                        "Checkpoint prune skipped: failed to replace shadow repo for %s",
                        working_dir,
                    )
                shutil.rmtree(temp_root, ignore_errors=True)
                return

            shutil.rmtree(temp_root, ignore_errors=True)
            keep_hashes = keep_hashes[1:]


def format_checkpoint_list(checkpoints: List[Dict], directory: str) -> str:
    """Format checkpoint list for display to user."""
    if not checkpoints:
        return f"No checkpoints found for {directory}"

    lines = [f"📸 Checkpoints for {directory}:\n"]
    for i, cp in enumerate(checkpoints, 1):
        # Parse ISO timestamp to something readable
        ts = cp["timestamp"]
        if "T" in ts:
            ts = ts.split("T")[1].split("+")[0].split("-")[0][:5]  # HH:MM
            date = cp["timestamp"].split("T")[0]
            ts = f"{date} {ts}"

        # Build change summary
        files = cp.get("files_changed", 0)
        ins = cp.get("insertions", 0)
        dele = cp.get("deletions", 0)
        if files:
            stat = f"  ({files} file{'s' if files != 1 else ''}, +{ins}/-{dele})"
        else:
            stat = ""

        lines.append(f"  {i}. {cp['short_hash']}  {ts}  {cp['reason']}{stat}")

    lines.append("\n  /rollback <N>             restore to checkpoint N")
    lines.append("  /rollback diff <N>        preview changes since checkpoint N")
    lines.append("  /rollback <N> <file>      restore a single file from checkpoint N")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-maintenance (issue #3015 follow-up)
# ---------------------------------------------------------------------------
#
# Every working directory the agent has ever touched gets its own shadow
# repo under CHECKPOINT_BASE. Per-repo ``_prune`` now enforces commit-count
# and size caps for active repos, but abandoned repos (deleted projects,
# one-off tmp dirs, long-stale work trees) can still accumulate forever
# without a base-directory sweep. Field reports put the typical offender at
# 1000+ repos / ~12 GB on active contributor machines.
#
# ``prune_checkpoints`` sweeps CHECKPOINT_BASE at startup, deleting shadow
# repos that match either criterion:
#   * orphan:  the ``HERMES_WORKDIR`` path no longer exists on disk
#   * stale:   the repo's newest mtime is older than ``retention_days``
#
# ``maybe_auto_prune_checkpoints`` wraps it with an idempotency marker
# (``CHECKPOINT_BASE/.last_prune``) so calling it on every CLI/gateway
# startup is free after the first run of the day.  Opt-in via
# ``checkpoints.auto_prune`` in config.yaml — default off so users who
# rely on ``/rollback`` against long-ago sessions never lose data
# silently.

_PRUNE_MARKER_NAME = ".last_prune"


def _read_workdir_marker(shadow_repo: Path) -> Optional[str]:
    """Read ``HERMES_WORKDIR`` from a shadow repo, or None if missing/unreadable."""
    try:
        return (shadow_repo / "HERMES_WORKDIR").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _shadow_repo_newest_mtime(shadow_repo: Path) -> float:
    """Return newest mtime across the shadow repo (walks objects/refs/HEAD).

    We walk instead of trusting the directory mtime because git's pack
    operations can leave the top-level dir untouched while refs/objects
    inside get updated.  Best-effort — returns 0.0 on any error.
    """
    newest = 0.0
    try:
        for p in shadow_repo.rglob("*"):
            try:
                m = p.stat().st_mtime
                if m > newest:
                    newest = m
            except OSError:
                continue
    except OSError:
        pass
    return newest


def prune_checkpoints(
    retention_days: int = 7,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
) -> Dict[str, int]:
    """Delete stale/orphan shadow repos under ``checkpoint_base``.

    A shadow repo is deleted when either:

    * ``delete_orphans=True`` and its ``HERMES_WORKDIR`` path no longer
      exists on disk (the original project was deleted / moved); OR
    * its newest in-repo mtime is older than ``retention_days`` days.

    Returns a dict with counts ``{"scanned", "deleted_orphan",
    "deleted_stale", "errors", "bytes_freed"}``.

    Never raises — maintenance must never block interactive startup.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    result = {
        "scanned": 0,
        "deleted_orphan": 0,
        "deleted_stale": 0,
        "errors": 0,
        "bytes_freed": 0,
    }
    if not base.exists():
        return result

    cutoff = 0.0
    if retention_days > 0:
        import time as _time
        cutoff = _time.time() - retention_days * 86400

    for child in base.iterdir():
        if not child.is_dir():
            continue
        # Protect the marker file and anything that isn't a real shadow
        # repo (no HEAD = not initialised, leave alone).
        if not (child / "HEAD").exists():
            continue
        result["scanned"] += 1

        reason: Optional[str] = None
        if delete_orphans:
            workdir = _read_workdir_marker(child)
            if workdir is None or not Path(workdir).exists():
                reason = "orphan"

        if reason is None and retention_days > 0:
            newest = _shadow_repo_newest_mtime(child)
            if newest > 0 and newest < cutoff:
                reason = "stale"

        if reason is None:
            continue

        # Measure size before delete (best-effort)
        try:
            size = sum(p.stat().st_size for p in child.rglob("*") if p.is_file())
        except OSError:
            size = 0
        try:
            shutil.rmtree(child)
            result["bytes_freed"] += size
            if reason == "orphan":
                result["deleted_orphan"] += 1
            else:
                result["deleted_stale"] += 1
            logger.debug("Pruned %s checkpoint repo: %s (%d bytes)", reason, child.name, size)
        except OSError as exc:
            result["errors"] += 1
            logger.warning("Failed to prune checkpoint repo %s: %s", child.name, exc)

    return result


def maybe_auto_prune_checkpoints(
    retention_days: int = 7,
    min_interval_hours: int = 24,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
) -> Dict[str, object]:
    """Idempotent wrapper around ``prune_checkpoints`` for startup hooks.

    Writes ``CHECKPOINT_BASE/.last_prune`` on completion so subsequent
    calls within ``min_interval_hours`` short-circuit.  Designed to be
    called once per CLI/gateway process startup; the marker keeps costs
    bounded regardless of how many times hermes is invoked per day.

    Returns ``{"skipped": bool, "result": prune_checkpoints-dict,
    "error": optional str}``.
    """
    import time as _time
    base = checkpoint_base or CHECKPOINT_BASE
    out: Dict[str, object] = {"skipped": False}

    try:
        if not base.exists():
            out["result"] = {
                "scanned": 0, "deleted_orphan": 0, "deleted_stale": 0,
                "errors": 0, "bytes_freed": 0,
            }
            return out

        marker = base / _PRUNE_MARKER_NAME
        now = _time.time()
        if marker.exists():
            try:
                last_ts = float(marker.read_text(encoding="utf-8").strip())
                if now - last_ts < min_interval_hours * 3600:
                    out["skipped"] = True
                    return out
            except (OSError, ValueError):
                pass  # corrupt marker — treat as no prior run

        result = prune_checkpoints(
            retention_days=retention_days,
            delete_orphans=delete_orphans,
            checkpoint_base=base,
        )
        out["result"] = result

        try:
            marker.write_text(str(now), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write checkpoint prune marker: %s", exc)

        total = result["deleted_orphan"] + result["deleted_stale"]
        if total > 0:
            logger.info(
                "checkpoint auto-maintenance: pruned %d repo(s) "
                "(%d orphan, %d stale), reclaimed %.1f MB",
                total,
                result["deleted_orphan"],
                result["deleted_stale"],
                result["bytes_freed"] / (1024 * 1024),
            )
    except Exception as exc:
        logger.warning("checkpoint auto-maintenance failed: %s", exc)
        out["error"] = str(exc)

    return out

