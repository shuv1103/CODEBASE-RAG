"""PII-gated answer cache in front of the cache repository, and its keys."""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Literal

from caching.pii_detector import contains_pii
from caching.semantic_cache_repository import SemanticCacheRepository

logger = logging.getLogger(__name__)

# Follow-up answers live under "<repo_id>:followup", apart from first turns.
FOLLOW_UP_SCOPE_SUFFIX = ":followup"

CacheOutcome = Literal["exact_hit", "semantic_hit", "miss", "blocked"]

# Conversational padding that never changes what is being asked.
FILLER_PHRASES = ("can you", "could you", "please", "tell me", "explain")
_FILLER = re.compile(
    r"\b(?:" + "|".join(map(re.escape, FILLER_PHRASES)) + r")\b"
)
# Only sentence punctuation: ".", "+", "#", "/", "-" inside code tokens
# distinct identifiers never share a key. A wrong cache hit is worse than a miss.
_SENTENCE_PUNCTUATION = re.compile(
    r"[?!\"`()\u201c\u201d\u2018]|[.,;:]+(?=\s|$)"
)
_APOSTROPHES = str.maketrans("", "", "'\u2019")
_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Strip presentation noise from a chat query, keeping its intent.

    Casefolds, drops sentence punctuation and apostrophes, removes filler
    phrases and collapses whitespace. Numbers, versions and code tokens are
    kept as typed; there are no synonyms or stopword lists.

    Args:
        text: The raw chat query.

    Returns:
        The cache key text; a query of only filler keeps its words.

    Example:
        >>> normalize_query("Can you please explain what's user.name in v1.2?")
        'whats user.name in v1.2'
    """
    query = _SENTENCE_PUNCTUATION.sub(
        " ", text.casefold().translate(_APOSTROPHES)
    )
    without_filler = _WHITESPACE.sub(" ", _FILLER.sub(" ", query)).strip()
    return without_filler or _WHITESPACE.sub(" ", query).strip()


@dataclass(frozen=True)
class CacheLookupResult:
    """Outcome of a cache lookup.

    Attributes:
        answer: The cached answer, or None on a miss or bypass.
        outcome: "exact_hit", "semantic_hit", "miss" or "blocked" (PII).
        nearest_similarity: Matched entry's similarity; None without a hit.
    """

    answer: str | None
    outcome: CacheOutcome
    nearest_similarity: float | None = None


class SemanticCacheService:
    """Answer cache for chat turns, per repo, bypassed for PII.

    Fail-safe: a cache problem costs a miss, never a failed chat turn.

    Args:
        repository: Backing store, e.g. LangCacheRepository.
    """

    def __init__(self, repository: SemanticCacheRepository) -> None:
        self._repository = repository

    def lookup(
        self,
        repo_id: str,
        message: str,
        previous_question: str | None = None,
        *,
        previous_answer: str = "",
    ) -> CacheLookupResult:
        """Look a chat turn up in the cache.

        A first turn matches exactly or semantically. A follow-up matches only
        other follow-ups, exactly: its key is mostly the previous question, so
        a semantic match would be another question's answer.

        Args:
            repo_id: Repo the chat is about.
            message: The raw chat message.
            previous_question: The previous question, or None on a first turn.
            previous_answer: The answer the session last received.

        Returns:
            CacheLookupResult naming the tier that answered, if any.
        """
        scope, key = self.cache_scope(
            repo_id, message, previous_question, previous_answer
        )
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

        hit = self._repository.lookup(
            scope, normalize_query(key), semantic=scope == repo_id
        )
        if hit is None:
            result = CacheLookupResult(answer=None, outcome="miss")
        else:
            result = CacheLookupResult(
                answer=hit.response,
                outcome=(
                    "exact_hit" if hit.strategy == "exact" else "semantic_hit"
                ),
                nearest_similarity=hit.similarity,
            )
        logger.info(
            "Cache lookup: %s",
            result.outcome,
            extra={
                "cache_event": result.outcome,
                "repo_id": repo_id,
                "scope": scope,
                "nearest_similarity": result.nearest_similarity,
            },
        )
        return result

    def store(
        self,
        repo_id: str,
        message: str,
        response: str,
        previous_question: str | None = None,
        *,
        previous_answer: str = "",
    ) -> None:
        """Cache an agent answer unless the turn contains PII. Never raises.

        Args:
            repo_id: Repo the chat is about.
            message: The raw chat message.
            response: The agent's answer.
            previous_question: The previous question, or None on a first turn.
            previous_answer: The answer the session last received.
        """
        scope, key = self.cache_scope(
            repo_id, message, previous_question, previous_answer
        )
        if self.is_blocked(repo_id, key):
            return
        self._repository.store(scope, normalize_query(key), response)

    @staticmethod
    def cache_scope(
        repo_id: str,
        message: str,
        previous_question: str | None,
        previous_answer: str = "",
    ) -> tuple[str, str]:
        r"""Return the (scope, key) a chat turn is cached under.

        A follow-up's answer depends on the answer before it, which can differ
        between users who asked the same question, so that answer is in the
        key: only identical conversations share follow-up entries.

        Args:
            repo_id: Repo the chat is about.
            message: The raw chat message.
            previous_question: The previous question, or None on a first turn.
            previous_answer: The answer the session last received.

        Returns:
            (repo_id, message) for a first turn, or ("<repo_id>:followup",
            "<previous_question>\n<answer sha256[:16]>\n<message>").
        """
        if previous_question is None:
            return repo_id, message
        answer_digest = hashlib.sha256(previous_answer.encode()).hexdigest()
        return (
            f"{repo_id}{FOLLOW_UP_SCOPE_SUFFIX}",
            f"{previous_question}\n{answer_digest[:16]}\n{message}",
        )

    def invalidate(self, repo_id: str) -> None:
        """Drop every cached answer for a repo, first turns and follow-ups.

        Args:
            repo_id: Repo being re-ingested.
        """
        self._repository.clear(repo_id)
        self._repository.clear(f"{repo_id}{FOLLOW_UP_SCOPE_SUFFIX}")

    @staticmethod
    def is_blocked(repo_id: str, message: str) -> bool:
        """Check whether a message must bypass the cache.

        Args:
            repo_id: Repo id, used only for logging.
            message: Text that would become the cache key.

        Returns:
            True if it contains PII, or if the PII check itself failed.
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
