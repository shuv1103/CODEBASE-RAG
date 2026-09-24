import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        Timezone-aware ISO timestamp, e.g. "2026-07-17T10:02:11.123456+00:00".
    """
    return datetime.now(timezone.utc).isoformat()


class RepoStatus(BaseModel):
    """Ingestion status record for one repository.

    Attributes:
        repo_id: Stable id derived from the normalized repo URL.
        url: Normalized repository URL.
        status: One of pending | cloning | chunking | embedding | ready |
            failed.
        chunks_total: Total chunks to embed, once chunking has finished.
        chunks_embedded: Chunks embedded so far.
        error: Failure reason, set only when status is "failed".
        created_at: UTC ISO timestamp of record creation.
        ready_at: UTC ISO timestamp when status became "ready".
    """

    repo_id: str
    url: str
    status: str  # pending | cloning | chunking | embedding | ready | failed
    chunks_total: Optional[int] = None
    chunks_embedded: Optional[int] = None
    error: Optional[str] = None
    created_at: str
    ready_at: Optional[str] = None


class StatusStore(ABC):
    """Abstract store for repo ingestion status.

    Repository Pattern: the abstraction the rest of the app codes against for
    repo ingestion status, so the backing store can be swapped without
    touching any caller (CLAUDE.md §10.3/§10.9 step 9).
    """

    @abstractmethod
    def get(self, repo_id: str) -> Optional[RepoStatus]:
        """Return the current status record for a repo.

        Args:
            repo_id: Repo to look up.

        Returns:
            The status record, or None if the repo is unknown.
        """

    @abstractmethod
    def create_if_absent(self, repo_id: str, url: str) -> tuple[RepoStatus, bool]:
        """Atomically create a pending record for repo_id if none exists yet.

        Callers use this single atomic operation (rather than a separate
        get()-then-create()) so that two concurrent POST /repos calls for the
        same brand-new repo can't both decide "this is new" and both enqueue
        an ingestion.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created). created=False means a record already existed
            and is returned unchanged.
        """

    @abstractmethod
    def update(self, repo_id: str, **fields) -> None:
        """Merge fields into the existing record for a repo. No-op if unknown.

        Args:
            repo_id: Repo whose record should be updated.
            **fields: RepoStatus fields to overwrite.
        """


class InMemoryStatusStore(StatusStore):
    """Thread-safe in-memory repo status tracker.

    Single-process only: a Celery worker process cannot see updates made
    here, since it doesn't share this object's memory. Kept for local dev /
    tests that don't want a Redis dependency — the Celery-backed production
    path (src/worker/) always uses RedisStatusStore instead.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, RepoStatus] = {}

    def get(self, repo_id: str) -> Optional[RepoStatus]:
        """Return the status record for a repo, or None if unknown.

        Args:
            repo_id: Repo to look up.

        Returns:
            The status record, or None.
        """
        with self._lock:
            return self._records.get(repo_id)

    def create_if_absent(self, repo_id: str, url: str) -> tuple[RepoStatus, bool]:
        """Create a pending record under the lock if none exists yet.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created); created=False means the existing record was
            returned unchanged.
        """
        with self._lock:
            existing = self._records.get(repo_id)
            if existing is not None:
                return existing, False

            record = RepoStatus(
                repo_id=repo_id,
                url=url,
                status="pending",
                created_at=utc_now_iso(),
            )
            self._records[repo_id] = record
            return record, True

    def update(self, repo_id: str, **fields) -> None:
        """Merge fields into a repo's record. No-op if unknown.

        Args:
            repo_id: Repo whose record should be updated.
            **fields: RepoStatus fields to overwrite.
        """
        with self._lock:
            existing = self._records.get(repo_id)
            if existing is None:
                return
            self._records[repo_id] = existing.model_copy(update=fields)
