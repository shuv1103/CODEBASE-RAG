"""Shallow, size-limited cloning of public GitHub repositories."""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from utils.config import settings
from utils.repo_id import normalize_github_url

logger = logging.getLogger(__name__)

_GITHUB_API_TIMEOUT_SECONDS = 10

# git runs on untrusted URLs: no prompts, no stored credentials (no private
# repos), no LFS (bypasses size checks), https only.
_GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "",
    "GIT_ALLOW_PROTOCOL": "https",
    "GIT_LFS_SKIP_SMUDGE": "1",
}

# All git needs from the environment; secrets never reach the subprocess.
_INHERITED_ENV_VARS = (
    "PATH",
    "HOME",
    "USERPROFILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "GIT_SSL_CAINFO",
)


def git_env() -> dict[str, str]:
    """Build the git subprocess's environment.

    Returns:
        The allowed variables from os.environ plus the hardening _GIT_ENV.
    """
    inherited = {
        name: os.environ[name]
        for name in _INHERITED_ENV_VARS
        if name in os.environ
    }
    return {**inherited, **_GIT_ENV}


class GitCloneError(Exception):
    """Raised when a repo can't be cloned; the message is user-facing."""


class RepoTooLargeError(Exception):
    """Raised when a repo exceeds a MAX_REPO_* limit (user-facing message)."""


def clone_repo(url: str) -> Path:
    """Shallow-clone a public GitHub repo into a temp dir, within size limits.

    The caller must delete the returned dir; on failure it is removed here.

    Args:
        url: Canonical https://github.com/<org>/<repo> URL (re-validated).

    Returns:
        Path of the temp directory holding the clone.

    Raises:
        GitCloneError: If the URL isn't canonical, the repo is missing or
            private, or git fails or times out.
        RepoTooLargeError: If a size limit is exceeded.
    """
    try:
        canonical = normalize_github_url(url)
    except ValueError as exc:
        raise GitCloneError(f"Not a GitHub repository URL: {exc}") from exc
    if canonical != url:
        raise GitCloneError(
            "Repository URL must be in canonical "
            "https://github.com/<org>/<repo> form"
        )

    check_remote_size(url)

    temp_dir = Path(tempfile.mkdtemp(prefix="codebase-rag-clone-"))
    try:
        run_clone(url, temp_dir)
        enforce_size_limits(temp_dir)
        return temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def fetch_repo_metadata(org: str, repo: str) -> dict[str, Any]:
    """Read a public repo's metadata from the GitHub REST API.

    Args:
        org: Owner (user or organization).
        repo: Repository name.

    Returns:
        The parsed JSON of GET /repos/{org}/{repo}; "size" is in KB.

    Raises:
        urllib.error.HTTPError: On a non-2xx response (404: missing or
            private).
        OSError: If GitHub can't be reached.
        ValueError: If the body isn't JSON.
    """
    request = urllib.request.Request(
        f"https://api.github.com/repos/{org}/{repo}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "codebase-rag",
        },
    )
    with urllib.request.urlopen(
        request, timeout=_GITHUB_API_TIMEOUT_SECONDS
    ) as response:
        return json.loads(response.read())


def check_remote_size(url: str) -> None:
    """Reject a missing, private or oversized repo before downloading it.

    If GitHub can't answer (rate limit, outage), the clone proceeds and the
    post-clone limits still apply.

    Args:
        url: Canonical GitHub URL.

    Raises:
        GitCloneError: If GitHub reports the repo as not found.
        RepoTooLargeError: If it exceeds MAX_REPO_DOWNLOAD_BYTES.
    """
    org, repo = urlparse(url).path.strip("/").split("/")
    try:
        metadata = fetch_repo_metadata(org, repo)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise GitCloneError("Repository not found or not public") from exc
        logger.warning(
            "GitHub size check unavailable (HTTP %s); "
            "relying on post-clone limits",
            exc.code,
        )
        return
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning(
            "GitHub size check unavailable (%s); relying on post-clone limits",
            exc,
        )
        return

    size_bytes = int(metadata.get("size") or 0) * 1024
    if size_bytes > settings.max_repo_download_bytes:
        raise RepoTooLargeError(
            "Repository is too large to download "
            f"({size_bytes // (1024 * 1024)} MB; "
            f"limit {settings.max_repo_download_bytes // (1024 * 1024)} MB)"
        )


def run_clone(url: str, dest: Path) -> None:
    """Run a depth-1, single-branch `git clone` with no credentials or prompts.

    Args:
        url: Canonical GitHub URL.
        dest: Directory to clone into.

    Raises:
        GitCloneError: If git fails or exceeds CLONE_TIMEOUT_SECONDS.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-c",
                "credential.helper=",
                "-c",
                "core.symlinks=false",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                "--no-tags",
                "--",
                url,
                str(dest),
            ],
            capture_output=True,
            text=True,
            # The returncode is turned into a GitCloneError below.
            check=False,
            timeout=settings.clone_timeout_seconds,
            env=git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise GitCloneError(
            f"git clone timed out after {settings.clone_timeout_seconds}s"
        ) from exc

    if result.returncode != 0:
        # stderr exposes server paths: log it, never show it to clients.
        logger.warning(
            "git clone of %s failed (exit %d): %s",
            url,
            result.returncode,
            result.stderr.strip(),
        )
        raise GitCloneError(
            "git clone failed (repository unavailable or network error)"
        )


def enforce_size_limits(repo_dir: Path) -> None:
    """Walk a clone (ignoring .git) and stop at the first exceeded limit.

    Args:
        repo_dir: Root of the clone.

    Raises:
        RepoTooLargeError: Past MAX_REPO_FILES files or MAX_REPO_BYTES.
    """
    total_files = 0
    total_bytes = 0

    for path in repo_dir.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue

        total_files += 1
        total_bytes += path.stat().st_size

        if (
            total_files > settings.max_repo_files
            or total_bytes > settings.max_repo_bytes
        ):
            raise RepoTooLargeError(
                f"Repository exceeds ingestion limits "
                f"(> {settings.max_repo_files} files or "
                f"> {settings.max_repo_bytes} bytes)"
            )
