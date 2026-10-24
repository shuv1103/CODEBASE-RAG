"""Redis-backed repo status store shared by the API and workers."""

from collections.abc import Iterator
from typing import Any

import redis

from utils.config import settings

from .status_store import RepoStatus, StatusStore

_KEY_PREFIX = "codebase-rag:repo-status:"
_POOL_WAIT_SECONDS = 10
# Redis drops idle connections; re-check one idle this long before reuse.
_HEALTH_CHECK_SECONDS = 30


class RedisStatusStore(StatusStore):
    """StatusStore in Redis, shared by the API and Celery worker processes.

    Workers report progress from other processes, so the record must live
    outside any one process. Connections come from a blocking pool capped at
    REDIS_STATUS_MAX_CONNECTIONS per process, keeping the whole stack within
    a managed Redis's client limit (Redis Cloud free plan: 30).

    Args:
        redis_url: Redis connection URL. Defaults to REDIS_URL.
        ttl_seconds: Record lifetime. Defaults to REPO_STATUS_TTL_SECONDS;
            0 or unset means records never expire.
    """

    def __init__(
        self, redis_url: str | None = None, ttl_seconds: int | None = None
    ) -> None:
        pool = redis.BlockingConnectionPool.from_url(
            redis_url or settings.redis_url,
            max_connections=settings.redis_status_max_connections,
            timeout=_POOL_WAIT_SECONDS,
            health_check_interval=_HEALTH_CHECK_SECONDS,
            socket_keepalive=True,
            decode_responses=True,
        )
        self._redis = redis.Redis(connection_pool=pool)
        self._ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else (settings.repo_status_ttl_seconds or None)
        )

    @staticmethod
    def key(repo_id: str) -> str:
        """Return the Redis key for a repo's status record.

        Args:
            repo_id: Repo the key belongs to.

        Returns:
            "codebase-rag:repo-status:<repo_id>".
        """
        return f"{_KEY_PREFIX}{repo_id}"

    def get(self, repo_id: str) -> RepoStatus | None:
        """Read and deserialize a repo's status record.

        Args:
            repo_id: Repo to look up.

        Returns:
            The status record, or None if the key is absent.
        """
        raw = self._redis.get(self.key(repo_id))
        return RepoStatus.model_validate_json(raw) if raw else None

    def create_if_absent(
        self, repo_id: str, url: str
    ) -> tuple[RepoStatus, bool]:
        """Atomically create a pending record using Redis SET NX.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created); created=False means another caller's record
            already existed and is returned unchanged.
        """
        record = RepoStatus.new_pending(repo_id, url)

        # SET NX is atomic: of concurrent requests for a new repo, one wins.
        created = self._redis.set(
            self.key(repo_id),
            record.model_dump_json(),
            nx=True,
            ex=self._ttl_seconds,
        )
        if created:
            return record, True

        existing = self.get(repo_id)
        if existing is None:
            # The record expired between the failed SET NX and this read.
            return self.create_if_absent(repo_id, url)
        return existing, False

    def update(self, repo_id: str, **fields: Any) -> None:
        """Merge fields into a repo's record and rewrite it. No-op if unknown.

        Args:
            repo_id: Repo whose record should be updated.
            **fields: RepoStatus fields to overwrite.
        """
        existing = self.get(repo_id)
        if existing is None:
            return
        updated = existing.with_updates(**fields)
        self._redis.set(
            self.key(repo_id), updated.model_dump_json(), ex=self._ttl_seconds
        )

    def iter_records(self) -> Iterator[RepoStatus]:
        """Yield every stored status record.

        Uses SCAN, so a large keyspace never blocks Redis.

        Yields:
            Each repo's status record.
        """
        for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
            if record := self.get(key.removeprefix(_KEY_PREFIX)):
                yield record

    def update_if_attempt(
        self, repo_id: str, expected_attempt_id: str | None, **fields: Any
    ) -> bool:
        """Merge fields into a repo's record only if its attempt still matches.

        Uses WATCH/MULTI, so a concurrent write between the read and the
        write makes this retry against the new record instead of
        overwriting it.

        Args:
            repo_id: Repo whose record should be updated.
            expected_attempt_id: Attempt the caller belongs to.
            **fields: RepoStatus fields to overwrite.

        Returns:
            True if the record was updated; False if it is unknown or now
            belongs to another attempt.
        """
        key = self.key(repo_id)
        with self._redis.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    existing = (
                        RepoStatus.model_validate_json(raw) if raw else None
                    )
                    if (
                        existing is None
                        or existing.attempt_id != expected_attempt_id
                    ):
                        return False
                    updated = existing.with_updates(**fields)
                    pipe.multi()
                    pipe.set(
                        key, updated.model_dump_json(), ex=self._ttl_seconds
                    )
                    pipe.execute()
                    return True
                except redis.WatchError:
                    continue
