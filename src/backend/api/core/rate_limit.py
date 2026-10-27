"""slowapi rate limiter and its per-IP and global keys."""

import logging
import time

from limits import parse
from slowapi import Limiter
from starlette.requests import Request

from utils.config import settings

from .exceptions import IngestionCapError

logger = logging.getLogger(__name__)

_INGESTION_CAP_KEY = "codebase-rag-ratelimit/global-new-ingestions"

# Chat turns and new repos cost OpenAI calls, so both are limited per IP.
# Per-IP limits can't cap the total (many IPs, or one attacker rotating
# them), so chat has one global bucket shared by every client, and new
# ingestions are capped globally by admit_new_ingestion.


def global_key() -> str:
    """Return the rate-limit key every client shares (a global bucket)."""
    return "global"


def client_ip(request: Request) -> str:
    """Return the client's IP as the rate-limit key, unspoofable by clients.

    Each trusted proxy appends its peer to X-Forwarded-For, so the entry
    `settings.trusted_proxy_hops` from the right was written by our outermost
    proxy. Entries further left come from the client and are ignored: the
    leftmost entry (uvicorn's --forwarded-allow-ips=*) is client-controlled.

    Args:
        request: The incoming request.

    Returns:
        The client IP, or the TCP peer address when no proxy is trusted or
        the header is missing/short.
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


# "memory://" counts per process; several API instances need a shared Redis
# URI. If that storage is down, counting falls back to memory, not failures.
limiter = Limiter(
    key_func=client_ip,
    storage_uri=settings.rate_limit_storage_uri,
    key_prefix="codebase-rag-ratelimit",
    in_memory_fallback_enabled=True,
    # A limiter fault must never turn into a 500.
    swallow_errors=True,
)


def admit_new_ingestion() -> None:
    """Count one new ingestion against the global cap.

    Only requests that start work are counted (GLOBAL_CREATE_REPO_RATE_LIMIT),
    so reopening an already ingested repo never uses up the cap. Fails open,
    like the limiter itself, if its storage is unreachable.

    Raises:
        IngestionCapError: If the cap is reached.
    """
    item = parse(settings.global_create_repo_rate_limit)
    try:
        if limiter.limiter.hit(item, _INGESTION_CAP_KEY):
            return
        reset_at, _ = limiter.limiter.get_window_stats(
            item, _INGESTION_CAP_KEY
        )
    # Fails open, like slowapi: any storage error admits the request.
    except Exception:  # noqa: BLE001
        logger.warning("Ingestion cap storage unavailable; admitting request")
        return
    raise IngestionCapError(max(1, int(reset_at - time.time())))
