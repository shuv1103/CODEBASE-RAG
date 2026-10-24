"""API core: error mapping, request limits and LLM concurrency control.

api.core.dependencies is not re-exported: importing it builds the Redis
status store, the answer cache client, the agent registry and the Celery app.
"""

from .body_limit import BodySizeLimitMiddleware
from .concurrency import LLMConcurrencyLimiter
from .exceptions import (
    AgentResponseError,
    IngestionCapError,
    InvalidRepoUrlError,
    LLMBusyError,
    RepoNotFoundError,
    RepoNotReadyError,
    register_exception_handlers,
)
from .rate_limit import (
    admit_new_ingestion,
    client_ip,
    global_key,
    limiter,
)

__all__ = [
    "AgentResponseError",
    "BodySizeLimitMiddleware",
    "IngestionCapError",
    "InvalidRepoUrlError",
    "LLMBusyError",
    "LLMConcurrencyLimiter",
    "RepoNotFoundError",
    "RepoNotReadyError",
    "admit_new_ingestion",
    "client_ip",
    "global_key",
    "limiter",
    "register_exception_handlers",
]
