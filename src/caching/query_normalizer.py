import re

_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Normalize a chat query before it touches the cache (both tiers).

    Trims leading/trailing whitespace and collapses repeated whitespace to a
    single space. Deliberately does not lowercase, stem, or strip
    punctuation/symbols — this is a code-question cache, and "JWTMiddleware"
    / "validate_token" style identifiers must survive normalization
    unchanged, since altering case or symbols would change the technical
    meaning of the query. (LangCache's exact strategy is already
    case-insensitive.)

    Args:
        text: The raw chat query.

    Returns:
        The whitespace-normalized query.

    Example:
        >>> normalize_query("  How does   JWTMiddleware work? ")
        'How does JWTMiddleware work?'
    """
    return _WHITESPACE_RUN.sub(" ", text.strip())
