"""Repo status record, the status store interface and its in-memory store."""

import threading
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Literal, TypedDict, Unpack

from pydantic import BaseModel

from utils.time_utils import utc_now_iso

# A typo like "redy" is a type error, and Pydantic rejects it on read.
RepoStatusValue = Literal[
    "pending", "cloning", "chunking", "embedding", "ready", "failed"
]

IN_FLIGHT_STATUSES = frozenset({"pending", "cloning", "chunking", "embedding"})


class StatusFields(TypedDict, total=False):
    """Fields a status write may set; a misspelt key is a type error."""

    url: str
    status: RepoStatusValue
    chunks_total: int | None
    chunks_embedded: int | None
    error: str | None
    created_at: str
    ready_at: str | None
    updated_at: str | None
    attempt_id: str | None


class PublicRepoStatus(BaseModel):
    """The repo status fields API clients see.

    Attributes:
        status: pending, cloning, chunking, embedding, ready or failed.
        chunks_total: Chunks to embed, once chunking has finished.
        chunks_embedded: Chunks embedded so far.
        error: User-facing failure reason, set only when "failed".
        created_at: UTC ISO 8601 creation time.
        ready_at: UTC ISO 8601 time the repo became "ready".
    """

    repo_id: str
    url: str
    status: RepoStatusValue
    chunks_total: int | None = None
    chunks_embedded: int | None = None
    error: str | None = None
    created_at: str
    ready_at: str | None = None


class RepoStatus(PublicRepoStatus):
    """A repo's ingestion status record, including internal fields.

    Attributes:
        updated_at: UTC ISO time of the last write; None on old records.
        attempt_id: Current ingestion attempt; a task with another id is
            stale.
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

    def with_updates(self, **fields: Unpack[StatusFields]) -> "RepoStatus":
        """Return a copy with fields overwritten and updated_at set to now.

        Args:
            **fields: RepoStatus fields to overwrite.
        """
        return self.model_copy(update={**fields, "updated_at": utc_now_iso()})

    def is_abandoned(
        self,
        stale_after_seconds: int,
        now: datetime | None = None,
        pending_stale_after_seconds: int | None = None,
    ) -> bool:
        """Check whether an in-flight record went too long without a write.

        Args:
            stale_after_seconds: Threshold for a running record.
            now: Current UTC time; defaults to the real clock.
            pending_stale_after_seconds: Threshold for a "pending" record,
                which may wait in the queue. Defaults to stale_after_seconds.

        Returns:
            True for an in-flight record older than its threshold.
        """
        if self.status not in IN_FLIGHT_STATUSES:
            return False
        # A pending task may legitimately wait in the queue for a long time.
        threshold = (
            pending_stale_after_seconds
            if self.status == "pending"
            and pending_stale_after_seconds is not None
            else stale_after_seconds
        )
        last_write = datetime.fromisoformat(self.updated_at or self.created_at)
        age = (now or datetime.now(UTC)) - last_write
        return age.total_seconds() > threshold


class StatusStore(ABC):
    """Storage interface for repo ingestion status records."""

    @abstractmethod
    def get(self, repo_id: str) -> RepoStatus | None:
        """Return a repo's status record.

        Args:
            repo_id: Repo to look up.

        Returns:
            The record, or None if the repo is unknown.
        """

    @abstractmethod
    def create_if_absent(
        self, repo_id: str, url: str
    ) -> tuple[RepoStatus, bool]:
        """Atomically create a pending record unless one exists.

        One atomic call, so concurrent requests for a new repo enqueue it once.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created); created is False when an existing record is
            returned unchanged.
        """

    @abstractmethod
    def update_if_attempt(
        self,
        repo_id: str,
        expected_attempt_id: str | None,
        **fields: Unpack[StatusFields],
    ) -> bool:
        """Atomically merge fields while the record's attempt_id still matches.

        Args:
            repo_id: Repo whose record to update.
            expected_attempt_id: Attempt the caller belongs to.
            **fields: RepoStatus fields to overwrite.

        Returns:
            True if updated; False if the record is unknown or now belongs to
            another attempt.
        """


class InMemoryStatusStore(StatusStore):
    """Thread-safe single-process StatusStore for tests and local runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, RepoStatus] = {}

    def get(self, repo_id: str) -> RepoStatus | None:
        """Return a repo's status record.

        Args:
            repo_id: Repo to look up.

        Returns:
            The record, or None if unknown.
        """
        with self._lock:
            return self._records.get(repo_id)

    def create_if_absent(
        self, repo_id: str, url: str
    ) -> tuple[RepoStatus, bool]:
        """Create a pending record under the lock unless one exists.

        Args:
            repo_id: Repo to create a record for.
            url: Normalized repository URL.

        Returns:
            (record, created).
        """
        with self._lock:
            existing = self._records.get(repo_id)
            if existing is not None:
                return existing, False

            record = RepoStatus.new_pending(repo_id, url)
            self._records[repo_id] = record
            return record, True

    def update(self, repo_id: str, **fields: Unpack[StatusFields]) -> None:
        """Merge fields unconditionally (test helper; no-op if unknown).

        Args:
            repo_id: Repo whose record to update.
            **fields: RepoStatus fields to overwrite.
        """
        with self._lock:
            existing = self._records.get(repo_id)
            if existing is None:
                return
            self._records[repo_id] = existing.with_updates(**fields)

    def update_if_attempt(
        self,
        repo_id: str,
        expected_attempt_id: str | None,
        **fields: Unpack[StatusFields],
    ) -> bool:
        """Merge fields under the lock while the attempt still matches.

        Args:
            repo_id: Repo whose record to update.
            expected_attempt_id: Attempt the caller belongs to.
            **fields: RepoStatus fields to overwrite.

        Returns:
            True if updated; False if unknown or another attempt's.
        """
        with self._lock:
            existing = self._records.get(repo_id)
            if existing is None or existing.attempt_id != expected_attempt_id:
                return False
            self._records[repo_id] = existing.with_updates(**fields)
            return True
