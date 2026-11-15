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
    """Use the app's log format in the worker (Celery setup_logging signal).

    Connecting this signal also stops Celery hijacking the root logger.
    """
    configure_logging(service="worker")


# Celery refuses rediss:// unless the SSL options say how to verify.
_VERIFIED_TLS = {"ssl_cert_reqs": ssl.CERT_REQUIRED}


def create_celery_app() -> Celery:
    """Create the Celery app on Redis, with TLS verification for rediss://.

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
        # Outlasts any run, so a running task is never redelivered.
        broker_transport_options={
            "visibility_timeout": settings.celery_visibility_timeout_seconds
        },
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Ack late so a crash redelivers; prefetch 1 so no worker hoards.
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        # Prefork only; under --pool=solo the pipeline's own checks apply.
        task_soft_time_limit=settings.ingestion_timeout_seconds,
        task_time_limit=settings.ingestion_hard_time_limit_seconds,
        # Stay within Redis Cloud's 30-client limit.
        broker_pool_limit=settings.celery_broker_pool_limit,
        redis_max_connections=settings.celery_redis_max_connections,
        # Redis drops idle connections; detect that before the next write.
        redis_backend_health_check_interval=30,
        redis_socket_keepalive=True,
    )

    return app


celery_app = create_celery_app()
