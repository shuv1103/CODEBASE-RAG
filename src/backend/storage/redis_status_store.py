"""Redis-backed repo status store shared by the API and workers."""

from collections.abc import Iterator
from typing import Unpack

import redis

from storage.status_store import RepoStatus, StatusFields, StatusStore
from utils.config import settings

_KEY_PREFIX = "codebase-rag:repo-status:"
_POOL_WAIT_SECONDS = 10
# Redis drops idle connections; re-check one idle this long before reuse.
_HEALTH_CHECK_SECONDS = 30
# Writers per record are few; this many lost races means something is wrong.
_MAX_CAS_ATTEMPTS = 10


class RedisStatusStore(StatusStore):
    """StatusStore in Redis, shared by the API and the Celery workers.

    Records are JSON at "codebase-rag:repo-status:<repo_id>"; connections come
    from a pool capped at REDIS_STATUS_MAX_CONNECTIONS.

    Args:
        redis_url: Defaults to REDIS_URL.
        ttl_seconds: Record lifetime. Defaults to REPO_STATUS_TTL_SECONDS;
            0 means records never expire.
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
        """Return the Redis key of a repo's status record.

        Args:
            repo_id: Repo the key belongs to.

        Returns:
            "codebase-rag:repo-status:<repo_id>".
        """
        return f"{_KEY_PREFIX}{repo_id}"

    def get(self, repo_id: str) -> RepoStatus | None:
        """Read a repo's status record.

        Args:
            repo_id: Repo to look up.

        Returns:
            The record, or None if the key is absent.
        """
        raw = self._redis.get(self.key(repo_id))
        return RepoStatus.model_validate_json(raw) if raw else None

    def create_if_absent(
        self, repo_id: str, url: str
    ) -> tuple[RepoStatus, bool]:
        """Atomically create a pending record with SET NX.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created); created is False when another caller's record
            already existed.
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

    def iter_records(self) -> Iterator[RepoStatus]:
        """Yield every stored status record, via non-blocking SCAN.

        Yields:
            Each repo's RepoStatus.
        """
        for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
            if record := self.get(key.removeprefix(_KEY_PREFIX)):
                yield record

    def update_if_attempt(
        self,
        repo_id: str,
        expected_attempt_id: str | None,
        **fields: Unpack[StatusFields],
    ) -> bool:
        """Merge fields while the attempt still matches, using WATCH/MULTI.

        A concurrent write in between makes it retry on the new record,
        up to _MAX_CAS_ATTEMPTS times.

        Args:
            repo_id: Repo whose record to update.
            expected_attempt_id: Attempt the caller belongs to.
            **fields: RepoStatus fields to overwrite.

        Returns:
            True if updated; False if unknown or another attempt's.

        Raises:
            redis.WatchError: If every attempt lost a race.
        """
        key = self.key(repo_id)
        with self._redis.pipeline() as pipe:
            for _ in range(_MAX_CAS_ATTEMPTS):
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
        raise redis.WatchError(
            f"{key} kept changing; gave up after {_MAX_CAS_ATTEMPTS} attempts"
        )
