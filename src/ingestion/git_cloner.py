import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_REPO_FILES = int(os.getenv("MAX_REPO_FILES", "5000"))
MAX_REPO_BYTES = int(os.getenv("MAX_REPO_BYTES", str(200 * 1024 * 1024)))
CLONE_TIMEOUT_SECONDS = int(os.getenv("CLONE_TIMEOUT_SECONDS", "120"))


class GitCloneError(Exception):
    """Raised when `git clone` fails or times out."""


class RepoTooLargeError(Exception):
    """Raised when a cloned repo exceeds MAX_REPO_FILES / MAX_REPO_BYTES."""


def clone_repo(url: str) -> Path:
    """Shallow-clone a repo into a fresh temp directory and enforce size limits.

    Caller owns the returned directory and must delete it when done — the
    directory (and any partial clone) is removed automatically if this
    function raises.

    Args:
        url: Git URL of the repository to clone.

    Returns:
        Path of the temp directory holding the clone.

    Raises:
        GitCloneError: If `git clone` fails or times out.
        RepoTooLargeError: If the clone exceeds MAX_REPO_FILES or
            MAX_REPO_BYTES.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="codebase-rag-clone-"))
    try:
        _run_clone(url, temp_dir)
        _enforce_size_limits(temp_dir)
        return temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _run_clone(url: str, dest: Path) -> None:
    """Run a depth-1, single-branch `git clone`.

    Args:
        url: Git URL of the repository to clone.
        dest: Directory to clone into.

    Raises:
        GitCloneError: If git exits non-zero or exceeds
            CLONE_TIMEOUT_SECONDS.
    """
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)],
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitCloneError(f"git clone timed out after {CLONE_TIMEOUT_SECONDS}s") from exc

    if result.returncode != 0:
        raise GitCloneError(f"git clone failed: {result.stderr.strip()}")


def _enforce_size_limits(repo_dir: Path) -> None:
    """Walk a clone (ignoring .git) and stop as soon as a size cap is exceeded.

    Args:
        repo_dir: Root of the cloned repository.

    Raises:
        RepoTooLargeError: If the file count exceeds MAX_REPO_FILES or the
            total size exceeds MAX_REPO_BYTES.
    """
    total_files = 0
    total_bytes = 0

    for path in repo_dir.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue

        total_files += 1
        total_bytes += path.stat().st_size

        if total_files > MAX_REPO_FILES or total_bytes > MAX_REPO_BYTES:
            raise RepoTooLargeError(
                f"Repository exceeds ingestion limits "
                f"(> {MAX_REPO_FILES} files or > {MAX_REPO_BYTES} bytes)"
            )
