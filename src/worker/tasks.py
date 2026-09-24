import logging

from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import task_failure

from api.redis_status_store import RedisStatusStore
from api.status_store import utc_now_iso
from pipeline.ingest_pipeline import ingest_repo

from .celery_app import celery_app

logger = logging.getLogger(__name__)

# One store (and so one bounded connection pool) per worker process, reused
# by every task. Built on first use rather than at import, because the API
# process imports this module only to enqueue tasks and never needs it.
_status_store: RedisStatusStore | None = None


def _get_status_store() -> RedisStatusStore:
    """Return this process's shared RedisStatusStore, building it on first call.

    Returns:
        The module-level RedisStatusStore.
    """
    global _status_store
    if _status_store is None:
        _status_store = RedisStatusStore()
    return _status_store


@celery_app.task(name="worker.tasks.ingest_repo_task")
def ingest_repo_task(repo_id: str, source_url: str) -> None:
    """Celery task running the full ingestion pipeline for one repo.

    Runs the clone -> chunk -> embed -> store pipeline in a worker process,
    instead of blocking the API process the
    way FastAPI's BackgroundTasks did (CLAUDE.md §10.5's "production
    upgrade" path).

    No automatic Celery-level retry on failure by design: `ingest_repo`
    already retries the transient, expected failure mode internally
    (OpenAI 429s, via ExponentialBackoff around embed_documents). A retry at
    this outer level would silently re-clone and re-embed an entire repo
    after any failure, including non-transient ones (bad URL, repo too
    large), burning OpenAI spend without the user asking for it. Instead
    this reports status="failed" and leaves retrying to the existing
    dedup-aware re-POST path (RagWorkflowFacade.start_ingestion,
    CLAUDE.md §10.4: "If found with status == failed, allow a retry").

    Status is reported into RedisStatusStore, not the API process's
    StatusStore instance — this task runs in a separate worker process, so
    only a store backed by something external to both processes (Redis) is
    visible to both.

    Args:
        repo_id: Repo to ingest.
        source_url: Normalized repository URL to clone.
    """
    status_store = _get_status_store()

    def on_progress(fields: dict) -> None:
        status_store.update(repo_id, **fields)

    try:
        ingest_repo(source_url, repo_id, on_progress=on_progress)
        status_store.update(repo_id, status="ready", ready_at=utc_now_iso(), error=None)
    except SoftTimeLimitExceeded:
        logger.exception("Ingestion timed out for repo_id=%s", repo_id)
        status_store.update(repo_id, status="failed", error="Ingestion exceeded the configured time limit")
    except Exception as exc:
        logger.exception("Ingestion failed for repo_id=%s", repo_id)
        status_store.update(repo_id, status="failed", error=str(exc))


@task_failure.connect(sender=ingest_repo_task)
def _on_ingest_task_failure(task_id=None, exception=None, args=None, **_kwargs) -> None:
    """Mark a repo "failed" when its ingestion worker process dies outright.

    Safety net for the one failure mode ingest_repo_task's own try/except
    above can never catch: the worker *process* dying outright (a
    WorkerLostError — e.g. a native SIGSEGV inside a C extension, or a hard
    time-limit SIGKILL). When that happens the task's except block never
    runs, because the process that would have run it is gone, so
    RedisStatusStore is never told and the frontend polls a status stuck at
    whatever stage was last reported, forever. This signal fires from the
    worker's parent process instead, which is still alive, so it's the only
    place left that can still report the failure.

    Args:
        task_id: Id of the failed task.
        exception: The exception Celery reported.
        args: The task's positional args; args[0] is the repo_id.
        **_kwargs: Other signal arguments (ignored).
    """
    if not args:
        return
    repo_id = args[0]
    logger.error("ingest_repo_task[%s] failed outside its own error handling (repo_id=%s): %s", task_id, repo_id, exception)
    status_store = _get_status_store()
    existing = status_store.get(repo_id)
    if existing and existing.status not in ("ready", "failed"):
        status_store.update(repo_id, status="failed", error=f"Ingestion worker crashed: {exception}")
