"""Answer cache backends, including Redis LangCache."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from langcache import LangCache
from langcache.models import CacheEntry, SearchStrategy

from utils.config import settings

logger = logging.getLogger(__name__)

# A slow cache must not hold up a chat turn; a timeout counts as a miss.
_REQUEST_TIMEOUT_MS = 5000
_MAX_RESULTS = 3
_REPO_ATTRIBUTE = "repo_id"

CacheStrategy = Literal["exact", "semantic"]


@dataclass(frozen=True)
class CacheHit:
    """A cached answer that matched a lookup.

    Attributes:
        response: The cached answer text.
        similarity: Match similarity rounded to 2 places; 1.0 for exact.
        strategy: "exact" or "semantic".
    """

    response: str
    similarity: float
    strategy: CacheStrategy


class SemanticCacheRepository(ABC):
    """Store of cached answers, searched exactly or semantically per repo."""

    @abstractmethod
    def lookup(
        self, repo_id: str, normalized_query: str, semantic: bool = True
    ) -> CacheHit | None:
        """Find the best cached answer for a query. Never raises.

        Args:
            repo_id: Cache scope (a repo id, or its follow-up scope).
            normalized_query: Query already passed through normalize_query.
            semantic: False restricts the search to exact matches.

        Returns:
            The best match, exact before semantic, or None on a miss.
        """

    @abstractmethod
    def store(
        self, repo_id: str, normalized_query: str, response: str
    ) -> None:
        """Cache an answer for a normalized query. Never raises.

        Args:
            repo_id: Cache scope.
            normalized_query: Query already passed through normalize_query.
            response: The answer to cache.
        """

    @abstractmethod
    def clear(self, repo_id: str) -> None:
        """Delete every cached entry in a scope. Never raises.

        Args:
            repo_id: Cache scope to empty.
        """


def strategy_of(entry: CacheEntry) -> CacheStrategy:
    """Return which LangCache search strategy produced an entry.

    Args:
        entry: A LangCache search result.

    Returns:
        "exact", or "semantic" for anything else.
    """
    value = getattr(entry.search_strategy, "value", entry.search_strategy)
    return "exact" if value == "exact" else "semantic"


class LangCacheRepository(SemanticCacheRepository):
    """Answer cache on Redis LangCache, scoped by a repo_id attribute.

    The repo_id attribute must be defined on the cache in the Redis Cloud
    console. Fail-safe: when disabled, unconfigured or erroring, lookups miss
    and stores and clears do nothing.

    Args:
        server_url: LangCache API URL. Defaults to LANGCACHE_SERVER_URL.
        cache_id: Defaults to LANGCACHE_CACHE_ID.
        api_key: Defaults to LANGCACHE_API_KEY.
        ttl_seconds: Entry lifetime. Defaults to LANGCACHE_TTL.
        similarity_threshold: Minimum semantic similarity (0.5-1.0). Defaults
            to LANGCACHE_SIMILARITY_THRESHOLD, else the cache's own setting.
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
        ttl = (
            ttl_seconds if ttl_seconds is not None else settings.langcache_ttl
        )
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
    ) -> CacheHit | None:
        """Search a scope's cached answers in one call, exact match first.

        Args:
            repo_id: Cache scope; an empty one always misses.
            normalized_query: Query already passed through normalize_query.
            semantic: False restricts the search to exact matches.

        Returns:
            The best match, or None on a miss, when disabled, or on any error.
        """
        if self._client is None or not repo_id:
            return None
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
            return None

        entries = response.data or []
        if not entries:
            return None

        best = next(
            (e for e in entries if strategy_of(e) == "exact"), None
        ) or max(entries, key=lambda e: e.similarity or 0.0)
        return CacheHit(
            response=best.response,
            similarity=(
                round(best.similarity, 2)
                if best.similarity is not None
                else 1.0
            ),
            strategy=strategy_of(best),
        )

    def store(
        self, repo_id: str, normalized_query: str, response: str
    ) -> None:
        """Cache a non-empty answer until the TTL. Never raises.

        Args:
            repo_id: Cache scope; an empty one is skipped.
            normalized_query: Query already passed through normalize_query.
            response: The answer to cache; an empty one is skipped.
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
        """Delete a scope's cached answers. Never raises.

        Args:
            repo_id: Cache scope; an empty one is refused, since LangCache
                would delete the whole cache.
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
