"""Repo status record, the status store interface and its in-memory store."""

import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from utils.time_utils import utc_now_iso

IN_FLIGHT_STATUSES = frozenset({"pending", "cloning", "chunking", "embedding"})


class PublicRepoStatus(BaseModel):
    """The fields of a repo's status record that API clients may see.

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
    status: str
    chunks_total: int | None = None
    chunks_embedded: int | None = None
    error: str | None = None
    created_at: str
    ready_at: str | None = None


class RepoStatus(PublicRepoStatus):
    """Ingestion status record for a repo: the API's fields plus internal ones.

    Attributes:
        updated_at: UTC ISO timestamp of the last write; None on records
            written before this field existed (created_at is used instead).
        attempt_id: Id of the current ingestion attempt (internal, not
            returned by the API); a queued task with another id is stale.
    """

    updated_at: str | None = None
    attempt_id: str | None = None

    @classmethod
    def new_pending(cls, repo_id: str, url: str) -> "RepoStatus":
        """Build the record for a repo's first ingestion attempt.

        Args:
            repo_id: Repo the record belongs to.
            url: Normalized repository URL.

        Returns:
            A "pending" record with a fresh attempt_id.
        """
        now = utc_now_iso()
        return cls(
            repo_id=repo_id,
            url=url,
            status="pending",
            created_at=now,
            updated_at=now,
            attempt_id=uuid.uuid4().hex,
        )

    def with_updates(self, **fields: Any) -> "RepoStatus":
        """Return a copy with fields overwritten and updated_at set to now.

        Args:
            **fields: RepoStatus fields to overwrite.

        Returns:
            The updated copy.
        """
        return self.model_copy(update={**fields, "updated_at": utc_now_iso()})

    def is_abandoned(
        self,
        stale_after_seconds: int,
        now: datetime | None = None,
        pending_stale_after_seconds: int | None = None,
    ) -> bool:
        """Check whether an unfinished ingestion went too long without update.

        Args:
            stale_after_seconds: Age beyond which no live task can own a
                running (cloning, chunking or embedding) record.
            now: Current time (UTC); defaults to the real clock.
            pending_stale_after_seconds: Age beyond which a "pending" record
                counts as lost. Defaults to stale_after_seconds.

        Returns:
            True for an in-flight record older than its threshold.
        """
        if self.status not in IN_FLIGHT_STATUSES:
            return False
        # A pending task may still be waiting in the queue behind others,
        # so its staleness can't be judged by the running-task threshold.
        threshold = (
            pending_stale_after_seconds
            if self.status == "pending" and pending_stale_after_seconds
            is not None
            else stale_after_seconds
        )
        last_write = datetime.fromisoformat(self.updated_at or self.created_at)
        age = (now or datetime.now(timezone.utc)) - last_write
        return age.total_seconds() > threshold


class StatusStore(ABC):
    """Interface for storing repo ingestion status records."""

    @abstractmethod
    def get(self, repo_id: str) -> RepoStatus | None:
        """Return the current status record for a repo.

        Args:
            repo_id: Repo to look up.

        Returns:
            The status record, or None if the repo is unknown.
        """

    @abstractmethod
    def create_if_absent(
        self, repo_id: str, url: str
    ) -> tuple[RepoStatus, bool]:
        """Atomically create a pending record for repo_id if none exists yet.

        One atomic call, not get()-then-create(), so two concurrent requests
        for the same new repo can't both enqueue an ingestion.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created). created=False means a record already existed
            and is returned unchanged.
        """

    @abstractmethod
    def update(self, repo_id: str, **fields: Any) -> None:
        """Merge fields into the existing record for a repo. No-op if unknown.

        Args:
            repo_id: Repo whose record should be updated.
            **fields: RepoStatus fields to overwrite.
        """

    @abstractmethod
    def update_if_attempt(
        self, repo_id: str, expected_attempt_id: str | None, **fields: Any
    ) -> bool:
        """Atomically merge fields only if the record's attempt_id matches.

        Lets a stale ingestion attempt (or a lost race between two retries)
        fail its write instead of overwriting the current attempt's record.

        Args:
            repo_id: Repo whose record should be updated.
            expected_attempt_id: Attempt the caller belongs to.
            **fields: RepoStatus fields to overwrite.

        Returns:
            True if the record was updated; False if it is unknown or now
            belongs to another attempt.
        """


class InMemoryStatusStore(StatusStore):
    """Thread-safe in-memory StatusStore for tests and local development.

    Single-process only: Celery workers can't see it, so production always
    uses RedisStatusStore.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, RepoStatus] = {}

    def get(self, repo_id: str) -> RepoStatus | None:
        """Return the status record for a repo, or None if unknown.

        Args:
            repo_id: Repo to look up.

        Returns:
            The status record, or None.
        """
        with self._lock:
            return self._records.get(repo_id)

    def create_if_absent(
        self, repo_id: str, url: str
    ) -> tuple[RepoStatus, bool]:
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

            record = RepoStatus.new_pending(repo_id, url)
            self._records[repo_id] = record
            return record, True

    def update(self, repo_id: str, **fields: Any) -> None:
        """Merge fields into a repo's record. No-op if unknown.

        Args:
            repo_id: Repo whose record should be updated.
            **fields: RepoStatus fields to overwrite.
        """
        with self._lock:
            existing = self._records.get(repo_id)
            if existing is None:
                return
            self._records[repo_id] = existing.with_updates(**fields)

    def update_if_attempt(
        self, repo_id: str, expected_attempt_id: str | None, **fields: Any
    ) -> bool:
        """Merge fields under the lock only if the attempt still matches.

        Args:
            repo_id: Repo whose record should be updated.
            expected_attempt_id: Attempt the caller belongs to.
            **fields: RepoStatus fields to overwrite.

        Returns:
            True if the record was updated; False if it is unknown or now
            belongs to another attempt.
        """
        with self._lock:
            existing = self._records.get(repo_id)
            if existing is None or existing.attempt_id != expected_attempt_id:
                return False
            self._records[repo_id] = existing.with_updates(**fields)
            return True
