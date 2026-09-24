import re

# Compiled once at import time — this runs on every chat message before any
# cache operation, so it must stay cheap.
_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # US phone number
    re.compile(r"\b(?:\d[ -]?){15}\d\b"),  # 16-digit card number, optional space/dash separators
]


def contains_pii(text: str) -> bool:
    """Detect PII that must never reach the semantic cache.

    Checks for an email address, a US phone number, or a 16-digit card-like
    number. Deliberately conservative and lightweight — no ML/NER model, no
    Luhn check on card numbers. False positives just mean an extra cache
    miss (safe); false negatives are the real risk, so keep patterns broad
    rather than narrow. Callers must treat any exception from this function
    as a detection failure and bypass the cache, never as "no PII found".

    Args:
        text: The chat message to scan.

    Returns:
        True if any PII pattern matches, False otherwise (including for
        empty text).

    Example:
        >>> contains_pii("mail me at a@b.com")
        True
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in _PII_PATTERNS)
