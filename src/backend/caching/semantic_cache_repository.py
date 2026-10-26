"""Answer cache backends, including Redis LangCache."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from langcache import LangCache
from langcache.models import CacheEntry, SearchStrategy

from utils.config import settings

logger = logging.getLogger(__name__)

# A slow cache must not hold up a chat turn; a timeout counts as a miss.
_REQUEST_TIMEOUT_MS = 5000
_MAX_RESULTS = 3
_REPO_ATTRIBUTE = "repo_id"


@dataclass(frozen=True)
class CacheHit:
    """A cache match.

    Attributes:
        response: The previously generated response.
        similarity: Similarity of the match, rounded to 2 decimal places —
            higher is more similar (1.0 for an exact match).
        strategy: Which search strategy matched: "exact" or "semantic".
    """

    response: str
    similarity: float
    strategy: str


@dataclass(frozen=True)
class SemanticLookupResult:
    """Result of a cache lookup.

    Attributes:
        hit: The best match for this repo_id, or None on a miss.
        nearest_similarity: Similarity of that match; None on a miss,
            since the cache only returns entries above its threshold.
    """

    hit: CacheHit | None
    nearest_similarity: float | None


# Frozen, so every miss path can share one instance.
_MISS = SemanticLookupResult(hit=None, nearest_similarity=None)


class SemanticCacheRepository(ABC):
    """Interface for a store of cached answers (exact and semantic match)."""

    @abstractmethod
    def lookup(
        self, repo_id: str, normalized_query: str, semantic: bool = True
    ) -> SemanticLookupResult:
        """Find a cached answer for a query, exact match first. Never raises.

        Args:
            repo_id: Repository the cache is scoped to.
            normalized_query: Query already passed through normalize_query.
            semantic: False restricts the search to exact matches.

        Returns:
            SemanticLookupResult — see its docstring.
        """

    @abstractmethod
    def store(self, repo_id: str, normalized_query: str, response: str) -> None:
        """Cache a response for a normalized query. Never raises.

        Args:
            repo_id: Repository the cache is scoped to.
            normalized_query: Query already passed through normalize_query.
            response: The answer to cache.
        """

    @abstractmethod
    def clear(self, repo_id: str) -> None:
        """Delete every cached entry for a repo. Never raises.

        Call this before a re-ingestion so no answer grounded in the old
        data survives it.

        Args:
            repo_id: Repository whose entries should be deleted.
        """


def strategy_of(entry: CacheEntry) -> str:
    """Return which LangCache search strategy produced a result.

    Args:
        entry: A LangCache search result (CacheEntry).

    Returns:
        "exact" or "semantic" (semantic when the field is missing).
    """
    return str(
        getattr(entry.search_strategy, "value", entry.search_strategy)
        or "semantic"
    )


class LangCacheRepository(SemanticCacheRepository):
    """Answer cache on Redis LangCache, serving the exact and semantic tiers.

    One search runs both strategies: `exact` (case-insensitive prompt match)
    and `semantic` (vector similarity, embedded by LangCache itself); an
    exact match wins. Every entry, search and delete is scoped by a `repo_id`
    attribute, since the same question means something different per
    codebase. That attribute must be defined on the cache in the Redis Cloud
    console (LangCache → your service → Attributes).

    Fail-safe: when LangCache is disabled, unconfigured, unreachable or
    erroring, lookups report a miss and stores and clears do nothing.

    Args:
        server_url: LangCache API base URL. Defaults to LANGCACHE_SERVER_URL.
        cache_id: LangCache cache ID. Defaults to LANGCACHE_CACHE_ID.
        api_key: LangCache service API key. Defaults to LANGCACHE_API_KEY.
        ttl_seconds: Entry lifetime. Defaults to LANGCACHE_TTL (3600).
        similarity_threshold: Minimum similarity (0.5–1.0) for a semantic
            hit. Defaults to LANGCACHE_SIMILARITY_THRESHOLD, or the cache's
            own threshold setting when unset.

    Example:
        >>> repo = LangCacheRepository()
        >>> repo.store("c3460b930023", "How does auth work?", "It uses JWTs...")
        >>> repo.lookup("c3460b930023", "how does auth work?").hit.strategy
        'exact'
    """

    def __init__(
        self,
        server_url: str | None = None,
        cache_id: str | None = None,
        api_key: str | None = None,
        ttl_seconds: int | None = None,
        similarity_threshold: float | None = None,
    ) -> None:
        server_url = server_url or settings.langcache_server_url
        cache_id = cache_id or settings.langcache_cache_id
        api_key = api_key or settings.langcache_api_key.get_secret_value()
        ttl = ttl_seconds if ttl_seconds is not None else settings.langcache_ttl
        self._ttl_millis = ttl * 1000
        self._similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.langcache_similarity_threshold
        )

        self._client: LangCache | None = None
        if not settings.langcache_enabled:
            logger.info(
                "LangCache disabled (LANGCACHE_ENABLED=false)",
                extra={"cache_event": "disabled"},
            )
        elif not (server_url and cache_id and api_key):
            logger.warning(
                "LangCache not configured (need LANGCACHE_SERVER_URL, "
                "LANGCACHE_CACHE_ID, LANGCACHE_API_KEY) — caching off",
                extra={"cache_event": "not_configured"},
            )
        else:
            if not server_url.startswith(("http://", "https://")):
                server_url = f"https://{server_url}"
            self._client = LangCache(
                server_url=server_url,
                cache_id=cache_id,
                api_key=api_key,
                timeout_ms=_REQUEST_TIMEOUT_MS,
            )

    def lookup(
        self, repo_id: str, normalized_query: str, semantic: bool = True
    ) -> SemanticLookupResult:
        """Search this repo's cached answers, preferring an exact match.

        Args:
            repo_id: Repository the cache is scoped to.
            normalized_query: Query already passed through normalize_query.
            semantic: False restricts the search to exact matches.

        Returns:
            SemanticLookupResult with the best hit, or a miss when nothing
            matched, when disabled, or on any error.
        """
        if self._client is None or not repo_id:
            return _MISS
        try:
            response = self._client.search(
                prompt=normalized_query,
                search_strategies=(
                    [SearchStrategy.EXACT, SearchStrategy.SEMANTIC]
                    if semantic
                    else [SearchStrategy.EXACT]
                ),
                similarity_threshold=self._similarity_threshold,
                attributes={_REPO_ATTRIBUTE: repo_id},
                max_results=_MAX_RESULTS,
            )
        # Fail-safe: a cache problem must never fail a chat turn.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LangCache lookup failed, bypassing cache",
                extra={
                    "cache_event": "lookup_error",
                    "repo_id": repo_id,
                    "error": str(exc),
                },
            )
            return _MISS

        entries = response.data or []
        if not entries:
            return _MISS

        best = next(
            (e for e in entries if strategy_of(e) == "exact"), None
        ) or max(entries, key=lambda e: e.similarity or 0.0)
        similarity = (
            round(best.similarity, 2) if best.similarity is not None else 1.0
        )
        return SemanticLookupResult(
            hit=CacheHit(
                response=best.response,
                similarity=similarity,
                strategy=strategy_of(best),
            ),
            nearest_similarity=similarity,
        )

    def store(self, repo_id: str, normalized_query: str, response: str) -> None:
        """Cache a non-empty answer for this repo until the TTL. Never raises.

        Args:
            repo_id: Repository the cache is scoped to.
            normalized_query: Query already passed through normalize_query.
            response: The answer to cache; empty responses are skipped.
        """
        if self._client is None or not repo_id or not response:
            return
        try:
            self._client.set(
                prompt=normalized_query,
                response=response,
                attributes={_REPO_ATTRIBUTE: repo_id},
                ttl_millis=self._ttl_millis,
            )
        # Fail-safe: a cache problem must never fail a chat turn.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LangCache store failed",
                extra={
                    "cache_event": "store_error",
                    "repo_id": repo_id,
                    "error": str(exc),
                },
            )

    def clear(self, repo_id: str) -> None:
        """Delete this repo's cached answers. Never raises.

        Refuses an empty repo_id: LangCache deletes the WHOLE cache when a
        delete has no attributes.

        Args:
            repo_id: Repository whose entries should be deleted.
        """
        if self._client is None or not repo_id:
            return
        try:
            self._client.delete_query(attributes={_REPO_ATTRIBUTE: repo_id})
        # Fail-safe: a cache problem must never fail a chat turn.
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LangCache clear failed",
                extra={
                    "cache_event": "clear_error",
                    "repo_id": repo_id,
                    "error": str(exc),
                },
            )
