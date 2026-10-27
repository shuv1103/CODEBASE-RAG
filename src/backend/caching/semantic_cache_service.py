"""PII-gated answer cache in front of the cache repository."""

import logging
from dataclasses import dataclass

from .pii_detector import contains_pii
from .query_normalizer import normalize_query
from .semantic_cache_repository import SemanticCacheRepository

logger = logging.getLogger(__name__)

# Follow-up answers live under "<repo_id>:followup", apart from first turns.
FOLLOW_UP_SCOPE_SUFFIX = ":followup"


@dataclass(frozen=True)
class CacheLookupResult:
    """The outcome of a two-tier cache lookup.

    Attributes:
        answer: The cached response, or None on any miss/bypass.
        outcome: One of "exact_hit" | "semantic_hit" | "miss" | "blocked",
            so tracing can tell which tier answered.
        nearest_similarity: Similarity of the matched entry, rounded to 2
            places (1.0 for an exact hit); None on a miss or "blocked".
    """

    answer: str | None
    outcome: str
    nearest_similarity: float | None = None


class SemanticCacheService:
    """Two-tier answer cache for chat turns, gated by a PII check.

    The exact tier matches the normalized question and never gives a false
    positive; the semantic tier also catches paraphrases. Both are scoped
    per repo_id. Every method is fail-safe: a cache problem costs a miss,
    never a failed chat turn.

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

    def lookup(
        self, repo_id: str, message: str, previous_question: str | None = None
    ) -> CacheLookupResult:
        """Look up a message; an exact match wins over a semantic one.

        A follow-up (previous_question set) is searched only among other
        follow-ups (see cache_scope), and only for an exact match: its key
        is mostly the previous question, so a semantic search would match
        any follow-up to a similar first question.

        Args:
            repo_id: Repository the lookup is scoped to.
            message: The raw chat message.
            previous_question: The session's previous question, for a
                follow-up turn; None on a session's first turn.

        Returns:
            CacheLookupResult describing which tier (if any) answered.
        """
        scope, key = self.cache_scope(repo_id, message, previous_question)
        if self.is_blocked(repo_id, key):
            logger.info(
                "Semantic cache bypassed (PII detected)",
                extra={
                    "cache_event": "blocked",
                    "repo_id": repo_id,
                    "scope": scope,
                },
            )
            return CacheLookupResult(answer=None, outcome="blocked")

        result = self._repository.lookup(
            scope, normalize_query(key), semantic=scope == repo_id
        )
        if result.hit is not None and result.hit.strategy == "exact":
            logger.info(
                "Exact cache hit",
                extra={
                    "cache_event": "exact_hit",
                    "repo_id": repo_id,
                    "scope": scope,
                },
            )
            return CacheLookupResult(
                answer=result.hit.response,
                outcome="exact_hit",
                nearest_similarity=result.nearest_similarity,
            )
        if result.hit is not None:
            logger.info(
                "Semantic cache hit",
                extra={
                    "cache_event": "semantic_hit",
                    "repo_id": repo_id,
                    "scope": scope,
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
                "scope": scope,
                "nearest_similarity": result.nearest_similarity,
            },
        )
        return CacheLookupResult(
            answer=None,
            outcome="miss",
            nearest_similarity=result.nearest_similarity,
        )

    def store(
        self,
        repo_id: str,
        message: str,
        response: str,
        previous_question: str | None = None,
    ) -> None:
        """Cache a response (both tiers) unless the message contains PII.

        Never raises.

        Args:
            repo_id: Repository the entry is scoped to.
            message: The raw chat message.
            response: The agent's answer to cache.
            previous_question: The session's previous question, for a
                follow-up turn; None on a session's first turn.
        """
        scope, key = self.cache_scope(repo_id, message, previous_question)
        if self.is_blocked(repo_id, key):
            return
        self._repository.store(scope, normalize_query(key), response)

    @staticmethod
    def cache_scope(
        repo_id: str, message: str, previous_question: str | None
    ) -> tuple[str, str]:
        """Return the (scope, key) a turn is cached under.

        A first turn is keyed by the question in the repo's scope. A
        follow-up's answer can depend on the conversation, so it is keyed by
        the previous question plus the message, in a separate follow-up
        scope: first-turn lookups never see follow-up answers.

        Args:
            repo_id: Repository id.
            message: The raw chat message.
            previous_question: The session's previous question, or None.

        Returns:
            (scope, key): the repository scope and the text to cache under.
        """
        if previous_question is None:
            return repo_id, message
        return (
            f"{repo_id}{FOLLOW_UP_SCOPE_SUFFIX}",
            f"{previous_question}\n{message}",
        )

    def invalidate(self, repo_id: str) -> None:
        """Drop every cached answer for a repo. Never raises.

        Call this before a re-ingestion, so no answer grounded in the old
        code survives it.

        Args:
            repo_id: Repository whose cached answers should be dropped.
        """
        self._repository.clear(repo_id)
        self._repository.clear(f"{repo_id}{FOLLOW_UP_SCOPE_SUFFIX}")

    @staticmethod
    def is_blocked(repo_id: str, message: str) -> bool:
        """Decide whether a message must bypass the cache.

        A failed PII check counts as blocked.

        Args:
            repo_id: Repository id, used only for logging.
            message: The raw chat message.

        Returns:
            True if the message contains PII or detection failed.
        """
        try:
            return contains_pii(message)
        # Fail-safe: a failed check must not let PII into the shared cache.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PII detection failed, bypassing cache",
                extra={
                    "cache_event": "pii_check_error",
                    "repo_id": repo_id,
                    "error": str(exc),
                },
            )
            return True
