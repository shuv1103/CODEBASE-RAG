"""Chat query normalization applied before any cache lookup."""

import re

_WHITESPACE_RUN = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([?!.,])")


def normalize_query(text: str) -> str:
    """Normalize a chat query before it touches the cache (both tiers).

    Trim the ends, collapse whitespace runs to one space, drop spaces before
    ? ! . , and strip trailing ? ! . so "do ?", "do?" and "do" share one
    exact key. Case and
    symbols are kept: in code questions, identifiers like "JWTMiddleware" or
    "validate_token" carry meaning. LangCache's exact match already ignores
    case.

    Args:
        text: The raw chat query.

    Returns:
        The whitespace-normalized query.

    Example:
        >>> normalize_query("  How does   JWTMiddleware work ? ")
        'How does JWTMiddleware work'
    """
    collapsed = _WHITESPACE_RUN.sub(" ", text.strip())
    return _SPACE_BEFORE_PUNCTUATION.sub(r"\1", collapsed).rstrip("?!.")
