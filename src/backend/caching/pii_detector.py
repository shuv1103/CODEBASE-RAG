"""Detection and redaction of PII and credentials in user text."""

import re

_REDACTED = "[REDACTED]"

_PII_PATTERNS = [
    # Email address
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # US phone number
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    # 16-digit card number, optional space/dash separators
    re.compile(r"\b(?:\d[ -]?){15}\d\b"),
]

# People paste credentials into code questions; treat them like PII.
_SECRET_PATTERNS = [
    # OpenAI / Anthropic-style keys
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),
    # LangSmith
    re.compile(r"\blsv2_[a-z]{2}_[A-Za-z0-9_]{20,}"),
    # AWS access key id
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # GitHub tokens
    re.compile(
        r"\bgh[pousr]_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{22,}"
    ),
    # Slack tokens
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    # Google API key
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    # JWT
    re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    ),
    # PEM private key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # password = "...", api_key: ..., token=...
    re.compile(
        r"(?i:\b(?:password|passwd|pwd|secret|api[_-]?key"
        r"|access[_-]?token|auth[_-]?token)"
        r"\s*[:=]\s*['\"]?[^\s'\"]{6,})"
    ),
]

# Runs on every chat message: one alternation scans in a single pass.
_SENSITIVE_RE = re.compile(
    "|".join(f"(?:{p.pattern})" for p in _PII_PATTERNS + _SECRET_PATTERNS)
)


def contains_pii(text: str) -> bool:
    """Detect PII or credentials that must never reach the shared cache.

    Matches emails, US phone and card numbers, provider API keys and tokens,
    JWTs, private keys and `password=`-style values. Deliberately broad: a
    false positive only costs a cache miss.

    Args:
        text: The chat message to scan.

    Returns:
        True if any pattern matches; False for clean or empty text.

    Example:
        >>> contains_pii("mail me at a@b.com")
        True
    """
    return _SENSITIVE_RE.search(text) is not None


def redact_pii(text: str) -> str:
    """Replace every PII or credential match with "[REDACTED]".

    Args:
        text: Text about to leave the process (e.g. a LangSmith trace).

    Returns:
        The text with each match replaced.

    Example:
        >>> redact_pii("mail a@b.com about it")
        'mail [REDACTED] about it'
    """
    return _SENSITIVE_RE.sub(_REDACTED, text)
