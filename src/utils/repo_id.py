import hashlib
from urllib.parse import urlparse


def normalize_github_url(url: str) -> str:
    """Validate a github.com repository URL and return its canonical form.

    The canonical form is https://github.com/<org>/<repo>, with no trailing
    slash, `.git` suffix, query string, or fragment, so the same repo always
    normalizes to the same string regardless of how the user typed it.

    Args:
        url: User-supplied GitHub repository URL.

    Returns:
        The canonical https://github.com/<org>/<repo> URL.

    Raises:
        ValueError: If the URL is not an http(s) github.com URL, or is
            missing the org/user or repo segment.

    Example:
        >>> normalize_github_url("https://github.com/org/repo.git/")
        'https://github.com/org/repo'
    """
    parsed = urlparse((url or "").strip())

    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() != "github.com":
        raise ValueError("URL must be a https://github.com/<org>/<repo> repository URL")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("URL must include both an org/user and a repo name")

    org, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]

    if not org or not repo:
        raise ValueError("URL must include both an org/user and a repo name")

    return f"https://github.com/{org}/{repo}"


def compute_repo_id(normalized_url: str) -> str:
    """Compute a stable short id for a normalized repo URL.

    The same URL always hashes to the same id.

    Args:
        normalized_url: URL already passed through normalize_github_url.

    Returns:
        The first 12 hex characters of the URL's SHA-1 digest.

    Example:
        >>> compute_repo_id("https://github.com/org/repo")
        'c3460b930023'
    """
    return hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()[:12]
