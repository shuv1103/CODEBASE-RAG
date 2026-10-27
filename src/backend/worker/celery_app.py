"""Celery application configuration."""

import logging
import ssl

from celery import Celery
from celery.signals import setup_logging

from utils.config import settings
from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


@setup_logging.connect
def configure_worker_logging(**_kwargs: object) -> None:
    """Use the app's log format in the worker instead of Celery's.

    Connecting this signal stops Celery hijacking the root logger, which
    would drop extra={...} fields and ignore LOG_LEVEL (--loglevel is then
    unused). Only the worker fires it; the API imports this module freely.
    """
    configure_logging(service="worker")


# Celery refuses a rediss:// URL whose SSL options don't say how to verify
# the server certificate.
_VERIFIED_TLS = {"ssl_cert_reqs": ssl.CERT_REQUIRED}


def create_celery_app() -> Celery:
    """Create and configure a Celery application instance.

    Broker and result backend come from CELERY_BROKER_URL /
    CELERY_RESULT_BACKEND (falling back to REDIS_URL), and task time limits
    are derived from INGESTION_TIMEOUT_SECONDS. rediss:// URLs connect over
    TLS with certificate verification.

    Returns:
        The configured Celery app, with worker.tasks registered.
    """
    app = Celery(
        "codebase_rag",
        broker=settings.celery_broker_url,
        backend=settings.celery_result_backend,
        include=["worker.tasks"],
    )

    if settings.plaintext_remote_redis_urls:
        logger.warning(
            "%s reach a remote Redis without TLS (redis://); use rediss:// "
            "so the password and data are encrypted",
            ", ".join(settings.plaintext_remote_redis_urls),
        )

    app.conf.update(
        broker_use_ssl=(
            _VERIFIED_TLS
            if settings.celery_broker_url.startswith("rediss://")
            else None
        ),
        redis_backend_use_ssl=(
            _VERIFIED_TLS
            if settings.celery_result_backend.startswith("rediss://")
            else None
        ),
        # An unacked task is redelivered after this timeout. Redis's default
        # (1 h) can be shorter than an ingestion, which would hand a running
        # task to a second worker; this one outlasts any run.
        broker_transport_options={
            "visibility_timeout": settings.celery_visibility_timeout_seconds
        },
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Ack after the task finishes, so a crash mid-ingestion redelivers
        # the job instead of losing it. Prefetch 1 stops one worker hoarding
        # long tasks while another sits idle.
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        # Enforced only by a prefork pool. Under --pool=solo (docker-compose,
        # Windows) they are ignored, and ingest_repo's own timeout checks
        # after every batch stop the run instead.
        task_soft_time_limit=settings.ingestion_timeout_seconds,
        task_time_limit=settings.ingestion_hard_time_limit_seconds,
        # Keep the API and every worker within a managed Redis's client
        # limit (Redis Cloud free plan: 30). Publishers wait for a free
        # pooled connection instead of opening another.
        broker_pool_limit=settings.celery_broker_pool_limit,
        redis_max_connections=settings.celery_redis_max_connections,
        # Redis drops idle connections; detect that before the next write.
        redis_backend_health_check_interval=30,
        redis_socket_keepalive=True,
    )

    return app


celery_app = create_celery_app()
