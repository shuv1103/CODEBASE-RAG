"""Celery ingestion task and its crash handler."""

import functools
import logging
from typing import Any, Unpack

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure

from ingestion.git_cloner import GitCloneError, RepoTooLargeError
from ingestion.ingest_pipeline import IngestionError, ingest_repo
from storage.redis_status_store import RedisStatusStore
from storage.status_store import IN_FLIGHT_STATUSES, StatusFields
from utils.time_utils import utc_now_iso

from .celery_app import celery_app

logger = logging.getLogger(__name__)

# Only these messages are written for users; any other error could leak
# internal hosts, keys or stack details, so it gets a generic message.
_USER_FACING_ERRORS = (GitCloneError, RepoTooLargeError, IngestionError)
_INTERNAL_ERROR = (
    "Ingestion failed due to an internal error. Please retry later."
)
_WORKER_CRASHED_ERROR = (
    "Ingestion worker crashed. Submit the repository again to retry."
)


# One store, and so one bounded connection pool, per worker process. Built
# on first use because the API imports this module only to enqueue tasks.
@functools.cache
def get_status_store() -> RedisStatusStore:
    """Return this process's shared RedisStatusStore, building it on first call.

    Returns:
        The process-wide RedisStatusStore.
    """
    return RedisStatusStore()


class AttemptSupersededError(Exception):
    """Raised when the record now belongs to a newer ingestion attempt."""


def write_status(
    store: RedisStatusStore,
    repo_id: str,
    attempt_id: str | None,
    **fields: Unpack[StatusFields],
) -> bool:
    """Write status fields, but only while the record is still this attempt's.

    Args:
        store: The status store.
        repo_id: Repo whose record should be updated.
        attempt_id: Attempt the task belongs to. None (a message queued
            before attempt ids existed) matches only a record that has none.
        **fields: RepoStatus fields to overwrite.

    Returns:
        False if the record has moved on to another attempt.
    """
    return store.update_if_attempt(repo_id, attempt_id, **fields)


def public_error(exc: Exception) -> str:
    """Return the failure message that is safe to show API clients.

    Args:
        exc: The exception that ended the ingestion.

    Returns:
        The exception's own message for user-facing failures, otherwise a
        generic message (the details are only logged).
    """
    return (
        str(exc) if isinstance(exc, _USER_FACING_ERRORS) else _INTERNAL_ERROR
    )


@celery_app.task(name="worker.tasks.ingest_repo_task")
def ingest_repo_task(
    repo_id: str, source_url: str, attempt_id: str | None = None
) -> None:
    """Run the full ingestion pipeline for one repo (Celery task).

    There is no Celery-level retry: the pipeline already retries OpenAI
    429s, and an outer retry would re-clone and re-embed the whole repo
    after any failure, even a permanent one, at the user's expense. A
    failed repo is retried when the user submits it again.

    Do nothing when the repo is already ready or failed, or the record has
    moved on to a newer attempt, so a redelivered, abandoned or superseded
    message can't ingest a repo twice. Every status write is tied to this
    attempt, and the run stops at its next progress write once superseded.

    Args:
        repo_id: Repo to ingest.
        source_url: Normalized repository URL to clone.
        attempt_id: The record's attempt id when the task was queued; None
            for messages queued before attempt ids existed.
    """
    status_store = get_status_store()
    record = status_store.get(repo_id)
    # A "failed" record was given up on (for example marked abandoned while
    # this message waited); running now would silently flip it back.
    if (
        record is None
        or record.status in ("ready", "failed")
        or record.attempt_id != attempt_id
    ):
        logger.info(
            "Skipping ingest_repo_task for repo_id=%s: "
            "already finished or superseded",
            repo_id,
        )
        return
    logger.info(
        "Ingestion started for repo_id=%s attempt_id=%s", repo_id, attempt_id
    )

    def on_progress(fields: StatusFields) -> None:
        """Write a progress update; stop the run once it is superseded."""
        if not write_status(status_store, repo_id, attempt_id, **fields):
            raise AttemptSupersededError(repo_id)

    try:
        ingest_repo(source_url, repo_id, on_progress=on_progress)
        write_status(
            status_store,
            repo_id,
            attempt_id,
            status="ready",
            ready_at=utc_now_iso(),
            error=None,
        )
    except AttemptSupersededError:
        logger.info(
            "Stopped ingest_repo_task for repo_id=%s: superseded by a newer "
            "attempt",
            repo_id,
        )
    except SoftTimeLimitExceeded:
        logger.exception("Ingestion timed out for repo_id=%s", repo_id)
        write_status(
            status_store,
            repo_id,
            attempt_id,
            status="failed",
            error="Ingestion exceeded the configured time limit",
        )
    except Exception as exc:
        logger.exception("Ingestion failed for repo_id=%s", repo_id)
        write_status(
            status_store,
            repo_id,
            attempt_id,
            status="failed",
            error=public_error(exc),
        )


@task_failure.connect(sender=ingest_repo_task)
def on_ingest_task_failure(
    task_id: str | None = None,
    exception: BaseException | None = None,
    args: tuple[Any, ...] | None = None,
    **_kwargs: Any,
) -> None:
    """Mark a repo "failed" when its ingestion worker process dies outright.

    A dead process (a native crash, or the hard time limit's SIGKILL) never
    runs the task's own error handling, which would leave the status stuck.
    Under a prefork pool this signal fires in the surviving parent. Under
    --pool=solo the whole worker dies, and the stuck record is failed by
    the API's staleness check instead.

    Args:
        task_id: Id of the failed task.
        exception: The exception Celery reported.
        args: The task's positional args: (repo_id, source_url[,
            attempt_id]).
        **_kwargs: Other signal arguments (ignored).
    """
    if not args:
        return
    repo_id = args[0]
    attempt_id = args[2] if len(args) > 2 else None
    logger.error(
        "ingest_repo_task[%s] failed outside its own error handling "
        "(repo_id=%s): %s",
        task_id,
        repo_id,
        exception,
    )
    status_store = get_status_store()
    existing = status_store.get(repo_id)
    if existing and existing.status in IN_FLIGHT_STATUSES:
        write_status(
            status_store,
            repo_id,
            attempt_id,
            status="failed",
            error=_WORKER_CRASHED_ERROR,
        )
