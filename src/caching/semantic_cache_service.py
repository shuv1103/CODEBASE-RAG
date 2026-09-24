import logging
from dataclasses import dataclass

from .pii_detector import contains_pii
from .query_normalizer import normalize_query
from .semantic_cache_repository import SemanticCacheRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheLookupResult:
    """The outcome of a two-tier cache lookup.

    Attributes:
        answer: The cached response, or None on any miss/bypass.
        outcome: One of "exact_hit" | "semantic_hit" | "miss" | "blocked" —
            kept distinct so callers (e.g. tracing.log_cache_event) can tell
            which tier resolved a hit instead of collapsing both into a
            generic "hit".
        nearest_similarity: Similarity of the matched entry (rounded to 2
            decimal places; 1.0 for an exact hit). None on a miss — the
            cache only returns entries above its similarity threshold — and
            for "blocked".
    """

    answer: str | None
    outcome: str
    nearest_similarity: float | None = None


class SemanticCacheService:
    """Orchestrates a two-tier cache around one chat turn.

    Both tiers are served by one repository call (LangCacheRepository):
    1. Exact-match tier — the stored question matches this one exactly
       (case-insensitive, after whitespace normalization). No false positives.
    2. Semantic tier — vector similarity, so it also catches genuine
       paraphrases the exact tier can't.

    Both tiers are PII-gated by the same check before the cache is touched,
    and both are scoped per repo_id. Every method is fail-safe — a cache
    problem costs a cache miss, never a failed chat turn.

    Args:
        repository: Backing store for both tiers.

    Example:
        >>> service = SemanticCacheService(LangCacheRepository())
        >>> result = service.lookup("c3460b930023", "How does auth work?")
        >>> result.outcome in {"exact_hit", "semantic_hit", "miss", "blocked"}
        True
    """

    def __init__(self, repository: SemanticCacheRepository) -> None:
        self._repository = repository

    def lookup(self, repo_id: str, message: str) -> CacheLookupResult:
        """Look up a message; an exact match wins over a semantic one.

        Args:
            repo_id: Repository the lookup is scoped to.
            message: The raw chat message.

        Returns:
            CacheLookupResult describing which tier (if any) answered.
        """
        if self._is_blocked(repo_id, message):
            logger.info(
                "Semantic cache bypassed (PII detected)",
                extra={"cache_event": "blocked", "repo_id": repo_id},
            )
            return CacheLookupResult(answer=None, outcome="blocked")

        result = self._repository.lookup(repo_id, normalize_query(message))
        if result.hit is not None and result.hit.strategy == "exact":
            logger.info("Exact cache hit", extra={"cache_event": "exact_hit", "repo_id": repo_id})
            return CacheLookupResult(
                answer=result.hit.response, outcome="exact_hit", nearest_similarity=result.nearest_similarity
            )
        if result.hit is not None:
            logger.info(
                "Semantic cache hit",
                extra={
                    "cache_event": "semantic_hit",
                    "repo_id": repo_id,
                    "nearest_similarity": result.nearest_similarity,
                },
            )
            return CacheLookupResult(
                answer=result.hit.response,
                outcome="semantic_hit",
                nearest_similarity=result.nearest_similarity,
            )

        logger.info(
            "Semantic cache miss",
            extra={
                "cache_event": "miss",
                "repo_id": repo_id,
                "nearest_similarity": result.nearest_similarity,
            },
        )
        return CacheLookupResult(answer=None, outcome="miss", nearest_similarity=result.nearest_similarity)

    def store(self, repo_id: str, message: str, response: str) -> None:
        """Cache a response (serves both tiers), unless the message contains PII.

        Never raises.

        Args:
            repo_id: Repository the entry is scoped to.
            message: The raw chat message.
            response: The agent's answer to cache.
        """
        if self._is_blocked(repo_id, message):
            return
        self._repository.store(repo_id, normalize_query(message), response)

    def invalidate(self, repo_id: str) -> None:
        """Drop every cached answer for a repo. Never raises.

        Call this before a re-ingestion replaces the repo's underlying data,
        so no stale answer grounded in the old code survives it.

        Args:
            repo_id: Repository whose cached answers should be dropped.
        """
        self._repository.clear(repo_id)

    @staticmethod
    def _is_blocked(repo_id: str, message: str) -> bool:
        """Decide whether a message must bypass the cache.

        A PII-detection failure is treated as blocked (fail-safe).

        Args:
            repo_id: Repository id, used only for logging.
            message: The raw chat message.

        Returns:
            True if the message contains PII or detection failed.
        """
        try:
            return contains_pii(message)
        except Exception as exc:  # noqa: BLE001 - fail-safe boundary, see pii_detector docstring
            logger.warning(
                "PII detection failed, bypassing cache",
                extra={"cache_event": "pii_check_error", "repo_id": repo_id, "error": str(exc)},
            )
            return True
