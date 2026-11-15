"""Celery ingestion task and its crash handler."""

import functools
import logging
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure

from caching.semantic_cache_repository import LangCacheRepository
from caching.semantic_cache_service import SemanticCacheService
from ingestion.git_cloner import GitCloneError, RepoTooLargeError
from ingestion.ingest_pipeline import IngestionError, ingest_repo
from storage.redis_status_store import RedisStatusStore
from storage.status_store import IN_FLIGHT_STATUSES, StatusFields
from utils.time_utils import utc_now_iso
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Only these messages are user-facing; others could leak internals.
_USER_FACING_ERRORS = (GitCloneError, RepoTooLargeError, IngestionError)
_INTERNAL_ERROR = (
    "Ingestion failed due to an internal error. Please retry later."
)
_WORKER_CRASHED_ERROR = (
    "Ingestion worker crashed. Submit the repository again to retry."
)


# Lazy: the API imports this module only to enqueue tasks.
@functools.cache
def get_status_store() -> RedisStatusStore:
    """Return this process's RedisStatusStore (one pool), built once."""
    return RedisStatusStore()


@functools.cache
def get_answer_cache() -> SemanticCacheService:
    """Return this process's answer cache client, built once."""
    return SemanticCacheService(LangCacheRepository())


class AttemptSupersededError(Exception):
    """Raised when a newer ingestion attempt now owns the repo's record."""


def public_error(exc: Exception) -> str:
    """Return the failure message that is safe to show API clients.

    Args:
        exc: The exception that ended the ingestion.

    Returns:
        Its own message for GitCloneError, RepoTooLargeError and
        IngestionError; a generic message for anything else.
    """
    return (
        str(exc) if isinstance(exc, _USER_FACING_ERRORS) else _INTERNAL_ERROR
    )


@celery_app.task(name="worker.tasks.ingest_repo_task")
def ingest_repo_task(
    repo_id: str, source_url: str, attempt_id: str | None = None
) -> None:
    """Run the ingestion pipeline for one repo (Celery task).

    No Celery-level retry: 429s are retried inside the pipeline, and an outer
    retry would re-clone and re-embed after permanent failures. A repo that
    is finished, or an attempt that is superseded, is skipped.

    Args:
        repo_id: Repo to ingest.
        source_url: Normalized repository URL to clone.
        attempt_id: Attempt the task belongs to; None for messages queued
            before attempt ids existed.
    """
    status_store = get_status_store()
    record = status_store.get(repo_id)
    # A "failed" record was given up on; running would flip it back.
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
        """Write a progress update, stopping the run once superseded.

        Raises:
            AttemptSupersededError: If a newer attempt owns the record.
        """
        if not status_store.update_if_attempt(repo_id, attempt_id, **fields):
            raise AttemptSupersededError(repo_id)

    try:
        ingest_repo(source_url, repo_id, on_progress=on_progress)
        status_store.update_if_attempt(
            repo_id,
            attempt_id,
            status="ready",
            ready_at=utc_now_iso(),
            error=None,
        )
        # Chat ran on the old index during a re-ingest; drop those answers.
        get_answer_cache().invalidate(repo_id)
    except AttemptSupersededError:
        logger.info(
            "Stopped ingest_repo_task for repo_id=%s: superseded by a newer "
            "attempt",
            repo_id,
        )
    except SoftTimeLimitExceeded:
        logger.exception("Ingestion timed out for repo_id=%s", repo_id)
        status_store.update_if_attempt(
            repo_id,
            attempt_id,
            status="failed",
            error="Ingestion exceeded the configured time limit",
        )
    except Exception as exc:
        logger.exception("Ingestion failed for repo_id=%s", repo_id)
        status_store.update_if_attempt(
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
    """Mark a repo "failed" when its worker process died mid-task (signal).

    Fires under a prefork pool; under --pool=solo the API's staleness check
    fails the record instead.

    Args:
        task_id: Id of the failed task.
        exception: The exception Celery reported.
        args: The task's args: (repo_id, source_url[, attempt_id]).
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
        status_store.update_if_attempt(
            repo_id,
            attempt_id,
            status="failed",
            error=_WORKER_CRASHED_ERROR,
        )
