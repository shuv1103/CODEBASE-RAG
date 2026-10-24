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
from urllib.parse import urlparse

from utils.config import settings
from utils.repo_id import normalize_github_url

logger = logging.getLogger(__name__)

_GITHUB_API_TIMEOUT_SECONDS = 10

# git runs on untrusted input (any public repo URL), so it gets no way to
# prompt, no stored credentials (a private repo the server's account could
# read must not be cloneable), no Git LFS downloads (an LFS-backed repo could
# pull gigabytes past the size checks) and https as the only transport.
_GIT_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_ASKPASS": "",
    "GIT_ALLOW_PROTOCOL": "https",
    "GIT_LFS_SKIP_SMUDGE": "1",
}

# The only inherited variables git needs: finding binaries, a home for its
# config, temp space, Windows system paths, proxies and CA certificates.
# Everything else (API keys, the Redis password) stays out of the subprocess.
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
    """Build the environment for the git subprocess.

    Returns:
        The allowed variables from os.environ, plus the hardening _GIT_ENV.
    """
    inherited = {
        name: os.environ[name]
        for name in _INHERITED_ENV_VARS
        if name in os.environ
    }
    return {**inherited, **_GIT_ENV}


class GitCloneError(Exception):
    """Raised when a repo can't be cloned.

    Causes: a bad URL, a missing or private repo, a git failure or a timeout.
    """


class RepoTooLargeError(Exception):
    """Raised when a repo exceeds a size limit.

    The limits are MAX_REPO_DOWNLOAD_BYTES, MAX_REPO_FILES and MAX_REPO_BYTES.
    """


def clone_repo(url: str) -> Path:
    """Shallow-clone a public GitHub repo to a temp dir, enforcing size limits.

    The caller owns the returned directory and must delete it; if this
    raises, the directory and any partial clone are removed here. The URL is
    re-validated because the worker takes it from the task queue and must
    not trust it.

    Args:
        url: Canonical https://github.com/<org>/<repo> URL.

    Returns:
        Path of the temp directory holding the clone.

    Raises:
        GitCloneError: If the URL isn't a canonical GitHub repo URL, the repo
            is missing or private, or `git clone` fails or times out.
        RepoTooLargeError: If the repo exceeds MAX_REPO_DOWNLOAD_BYTES (before
            cloning) or MAX_REPO_FILES / MAX_REPO_BYTES (after).
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


def fetch_repo_metadata(org: str, repo: str) -> dict:
    """Read a public repo's metadata from the GitHub REST API.

    Args:
        org: Owner (user or organization).
        repo: Repository name.

    Returns:
        The parsed JSON body of GET /repos/{org}/{repo}.

    Raises:
        urllib.error.HTTPError: On a non-2xx response (404 = missing or
            private).
        urllib.error.URLError / OSError / ValueError: If GitHub can't be reached
            or the body isn't JSON.
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
    """Reject a missing/private or oversized repo before downloading anything.

    GitHub's "size" (KB) covers the whole packed repository, history
    included, so it bounds what a shallow clone can download. When the API
    can't answer (rate limit, outage) the clone proceeds and the post-clone
    checks still apply.

    Args:
        url: Canonical https://github.com/<org>/<repo> URL.

    Raises:
        GitCloneError: If GitHub reports the repo as not found (missing or
            private).
        RepoTooLargeError: If the repo is larger than MAX_REPO_DOWNLOAD_BYTES.
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
    """Run a depth-1, single-branch `git clone` with credentials/prompts off.

    Args:
        url: Canonical GitHub URL of the repository to clone.
        dest: Directory to clone into.

    Raises:
        GitCloneError: If git exits non-zero or exceeds
            CLONE_TIMEOUT_SECONDS.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-c", "credential.helper=",
                "-c", "core.symlinks=false",
                "clone", "--depth", "1", "--single-branch", "--no-tags",
                "--", url, str(dest),
            ],
            capture_output=True,
            text=True,
            timeout=settings.clone_timeout_seconds,
            env=git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise GitCloneError(
            f"git clone timed out after {settings.clone_timeout_seconds}s"
        ) from exc

    if result.returncode != 0:
        raise GitCloneError(f"git clone failed: {result.stderr.strip()}")


def enforce_size_limits(repo_dir: Path) -> None:
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

        if (
            total_files > settings.max_repo_files
            or total_bytes > settings.max_repo_bytes
        ):
            raise RepoTooLargeError(
                f"Repository exceeds ingestion limits "
                f"(> {settings.max_repo_files} files or "
                f"> {settings.max_repo_bytes} bytes)"
            )
