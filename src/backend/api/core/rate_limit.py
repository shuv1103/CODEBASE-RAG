"""slowapi rate limiter and its per-IP and global keys."""

import logging
import time

from limits import parse
from slowapi import Limiter
from starlette.requests import Request

from api.core.exceptions import IngestionCapError
from utils.config import settings

logger = logging.getLogger(__name__)

_INGESTION_CAP_KEY = "codebase-rag-ratelimit/global-new-ingestions"


def global_key() -> str:
    """Return the rate-limit key every client shares (one global bucket)."""
    return "global"


def client_ip(request: Request) -> str:
    """Return the client's IP as the rate-limit key, unspoofable by clients.

    Takes the X-Forwarded-For entry TRUSTED_PROXY_HOPS from the right (written
    by our own proxy); entries further left are client-controlled.

    Args:
        request: The incoming request.

    Returns:
        The client IP, or the TCP peer address when no proxy is trusted or the
        header is short.
    """
    hops = settings.trusted_proxy_hops
    if hops:
        forwarded = [
            entry.strip()
            for entry in request.headers.get("x-forwarded-for", "").split(",")
            if entry.strip()
        ]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return request.client.host if request.client else "127.0.0.1"


# "memory://" counts per process; several instances need a Redis URI.
limiter = Limiter(
    key_func=client_ip,
    storage_uri=settings.rate_limit_storage_uri,
    key_prefix="codebase-rag-ratelimit",
    in_memory_fallback_enabled=True,
    # A limiter fault must never turn into a 500.
    swallow_errors=True,
)


def admit_new_ingestion() -> None:
    """Count one new ingestion against GLOBAL_CREATE_REPO_RATE_LIMIT.

    Fails open (admits) if the limiter's storage is unreachable.

    Raises:
        IngestionCapError: If the cap is reached, with the seconds until reset.
    """
    item = parse(settings.global_create_repo_rate_limit)
    try:
        if limiter.limiter.hit(item, _INGESTION_CAP_KEY):
            return
        reset_at, _ = limiter.limiter.get_window_stats(
            item, _INGESTION_CAP_KEY
        )
    except Exception:  # noqa: BLE001
        logger.warning("Ingestion cap storage unavailable; admitting request")
        return
    raise IngestionCapError(max(1, int(reset_at - time.time())))
