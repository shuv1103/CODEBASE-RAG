import os
from typing import Optional

import redis
from dotenv import load_dotenv

from .status_store import RepoStatus, StatusStore, utc_now_iso

load_dotenv()

_KEY_PREFIX = "codebase-rag:repo-status:"
# Seconds a caller waits for a free pooled connection before redis-py raises.
_POOL_WAIT_SECONDS = 10
# Ping a pooled connection before reuse if it has been idle this long.
_HEALTH_CHECK_SECONDS = 30


class RedisStatusStore(StatusStore):
    """Redis-backed StatusStore shared by the API and Celery worker processes.

    The shared source of truth for repo ingestion status between the FastAPI
    API process and Celery worker process(es).

    This is what makes the Celery migration correct rather than just present:
    InMemoryStatusStore's dict lives in one process's memory, so a worker
    process running `ingest_repo_task` (src/worker/tasks.py) could never
    report progress back to it. Redis is external to both, so every process
    reads/writes the same record (CLAUDE.md §10.3/§10.9 step 9).

    Connections come from a bounded, blocking pool: at most
    REDIS_STATUS_MAX_CONNECTIONS per process, and a caller waits for a free
    one instead of opening another. That keeps each process within a fixed
    share of a managed Redis's client limit (Redis Cloud free: 30). Pooled
    connections are health-checked before reuse, so ones the server dropped
    while idle are replaced instead of failing a request.

    Args:
        redis_url: Redis connection URL. Defaults to REDIS_URL.
        ttl_seconds: Record lifetime. Defaults to REPO_STATUS_TTL_SECONDS;
            0 or unset means records never expire.
    """

    def __init__(self, redis_url: str | None = None, ttl_seconds: int | None = None) -> None:
        pool = redis.BlockingConnectionPool.from_url(
            redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            max_connections=int(os.getenv("REDIS_STATUS_MAX_CONNECTIONS", "5")),
            timeout=_POOL_WAIT_SECONDS,
            health_check_interval=_HEALTH_CHECK_SECONDS,
            socket_keepalive=True,
            decode_responses=True,
        )
        self._redis = redis.Redis(connection_pool=pool)
        env_ttl = int(os.getenv("REPO_STATUS_TTL_SECONDS", "0"))
        self._ttl_seconds = ttl_seconds if ttl_seconds is not None else (env_ttl or None)

    @staticmethod
    def _key(repo_id: str) -> str:
        """Return the Redis key for a repo's status record.

        Args:
            repo_id: Repo the key belongs to.

        Returns:
            "codebase-rag:repo-status:<repo_id>".
        """
        return f"{_KEY_PREFIX}{repo_id}"

    def get(self, repo_id: str) -> Optional[RepoStatus]:
        """Read and deserialize a repo's status record.

        Args:
            repo_id: Repo to look up.

        Returns:
            The status record, or None if the key is absent.
        """
        raw = self._redis.get(self._key(repo_id))
        return RepoStatus.model_validate_json(raw) if raw else None

    def create_if_absent(self, repo_id: str, url: str) -> tuple[RepoStatus, bool]:
        """Atomically create a pending record using Redis SET NX.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created); created=False means another caller's record
            already existed and is returned unchanged.
        """
        record = RepoStatus(
            repo_id=repo_id,
            url=url,
            status="pending",
            created_at=utc_now_iso(),
        )

        # SET ... NX is atomic in Redis, so this is safe under concurrent
        # requests for the same brand-new repo_id — exactly one caller wins.
        created = self._redis.set(
            self._key(repo_id), record.model_dump_json(), nx=True, ex=self._ttl_seconds
        )
        if created:
            return record, True

        existing = self.get(repo_id)
        if existing is None:
            # The record expired (TTL) in the gap between the failed NX-set
            # and this read — safe to retry as a fresh create.
            return self.create_if_absent(repo_id, url)
        return existing, False

    def update(self, repo_id: str, **fields) -> None:
        """Merge fields into a repo's record and rewrite it. No-op if unknown.

        Args:
            repo_id: Repo whose record should be updated.
            **fields: RepoStatus fields to overwrite.
        """
        existing = self.get(repo_id)
        if existing is None:
            return
        updated = existing.model_copy(update=fields)
        self._redis.set(self._key(repo_id), updated.model_dump_json(), ex=self._ttl_seconds)
