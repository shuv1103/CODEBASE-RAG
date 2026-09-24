import os

from celery import Celery
from dotenv import load_dotenv

load_dotenv()


def create_celery_app() -> Celery:
    """Create and configure a Celery application instance.

    Broker and result backend come from CELERY_BROKER_URL /
    CELERY_RESULT_BACKEND (falling back to REDIS_URL), and task time limits
    are derived from INGESTION_TIMEOUT_SECONDS.

    Returns:
        The configured Celery app, with worker.tasks registered.
    """

    default_broker = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    broker_url = os.getenv("CELERY_BROKER_URL", default_broker)
    result_backend = os.getenv("CELERY_RESULT_BACKEND", default_broker)

    app = Celery(
        "codebase_rag",
        broker=broker_url,
        backend=result_backend,
        include=["worker.tasks"],
    )

    ingestion_timeout = int(os.getenv("INGESTION_TIMEOUT_SECONDS", "900"))

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Long-running I/O-bound tasks (git clone + OpenAI calls): ack only
        # after the task finishes, so a worker crash mid-ingestion redelivers
        # the job instead of silently losing it; keep prefetch at 1 so one
        # worker doesn't hoard several long tasks while a sibling sits idle.
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        # Soft limit lets the task catch SoftTimeLimitExceeded and report
        # status="failed" cleanly; hard limit is a slightly longer backstop
        # that SIGKILLs the worker child if the soft handler itself hangs.
        task_soft_time_limit=ingestion_timeout,
        task_time_limit=ingestion_timeout + 60,
        # Bound the Redis connections each process opens, so the API plus
        # every worker stay within a managed Redis's client limit (Redis
        # Cloud free: 30). broker_pool_limit caps pooled producer
        # connections (publishers wait for a free one); redis_max_connections
        # caps the result backend's pool.
        broker_pool_limit=int(os.getenv("CELERY_BROKER_POOL_LIMIT", "2")),
        redis_max_connections=int(os.getenv("CELERY_REDIS_MAX_CONNECTIONS", "4")),
        # Detect connections the server dropped while idle, instead of
        # failing on the next write.
        redis_backend_health_check_interval=30,
        redis_socket_keepalive=True,
    )

    return app


celery_app = create_celery_app()
