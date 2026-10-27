"""Domain exceptions and their mapping to JSON HTTP error responses."""

import logging
import time

import redis
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

# A Redis outage blocks every repo request but is usually brief.
_STORE_UNAVAILABLE_RETRY_AFTER_SECONDS = 5


class InvalidRepoUrlError(Exception):
    """Raised when a submitted URL is not a valid github.com repository URL."""


class RepoNotFoundError(Exception):
    """Raised when a repo_id has no status record (mapped to HTTP 404).

    Args:
        repo_id: The unknown repo id.

    Attributes:
        repo_id: The unknown repo id.
    """

    def __init__(self, repo_id: str) -> None:
        self.repo_id = repo_id
        super().__init__(f"Unknown repo_id: {repo_id}")


class RepoNotReadyError(Exception):
    """Raised when chatting with a repo that is not "ready" (HTTP 409).

    Args:
        repo_id: The repo that was queried.
        status: Its current ingestion status.

    Attributes:
        repo_id: The repo that was queried.
        status: Its current ingestion status.
    """

    def __init__(self, repo_id: str, status: str) -> None:
        self.repo_id = repo_id
        self.status = status
        super().__init__(f"Repo {repo_id} is not ready (status={status})")


class AgentResponseError(Exception):
    """Raised when the agent run completes without producing a final answer."""


class LLMBusyError(Exception):
    """Raised when no LLM slot freed up within the queue timeout (HTTP 503).

    Args:
        retry_after_seconds: Suggested wait before the client retries.

    Attributes:
        retry_after_seconds: Suggested wait before the client retries.
    """

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            "The assistant is busy right now. Please retry in a few seconds."
        )


class IngestionCapError(Exception):
    """Raised when the global cap on new ingestions is reached (HTTP 429).

    Args:
        retry_after_seconds: Seconds until the cap's window resets.

    Attributes:
        retry_after_seconds: Seconds until the cap's window resets.
    """

    def __init__(self, retry_after_seconds: int) -> None:
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            "Too many new repositories are being ingested right now. "
            "Try again later."
        )


# Domain errors whose message is safe to show the client as-is.
_CLIENT_ERROR_STATUS: dict[type[Exception], int] = {
    InvalidRepoUrlError: 400,
    RepoNotFoundError: 404,
    RepoNotReadyError: 409,
}


def retry_after_seconds(request: Request, exc: RateLimitExceeded) -> int:
    """Return the seconds until the exceeded rate-limit window resets.

    Args:
        request: The rejected request (slowapi stored the hit limit on it).
        exc: The rate-limit error.

    Returns:
        Seconds until reset, or the limit's whole window if the limiter's
        storage can't be read.
    """
    try:
        limit_item, keys = request.state.view_rate_limit
        reset_at, _ = request.app.state.limiter.limiter.get_window_stats(
            limit_item, *keys
        )
        return max(1, int(reset_at - time.time()))
    # The limiter's storage (memory or Redis) can fail in many ways; the
    # whole window is a safe, slightly pessimistic Retry-After.
    except Exception:  # noqa: BLE001
        return exc.limit.limit.get_expiry()


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain exceptions to JSON HTTP error responses.

    InvalidRepoUrlError -> 400, RepoNotFoundError -> 404,
    RepoNotReadyError -> 409, RateLimitExceeded and IngestionCapError -> 429
    (with Retry-After),
    LLMBusyError -> 503 (with Retry-After), Redis unreachable -> 503 (with
    Retry-After), AgentResponseError -> 502, and any other exception -> 500
    with a generic message.

    Args:
        app: The FastAPI application to register handlers on.
    """

    async def client_error(request: Request, exc: Exception) -> JSONResponse:
        # Starlette also routes subclasses here, so walk the MRO.
        status_code = next(
            _CLIENT_ERROR_STATUS[cls]
            for cls in type(exc).__mro__
            if cls in _CLIENT_ERROR_STATUS
        )
        return JSONResponse(
            status_code=status_code, content={"detail": str(exc)}
        )

    for exc_class in _CLIENT_ERROR_STATUS:
        app.add_exception_handler(exc_class, client_error)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limited(
        request: Request, exc: RateLimitExceeded
    ) -> JSONResponse:
        # The frontend shows "detail", so every error uses that shape.
        detail = (
            f"Too many requests: limit is {exc.detail}. Try again shortly."
        )
        return JSONResponse(
            status_code=429,
            content={"detail": detail},
            headers={"Retry-After": str(retry_after_seconds(request, exc))},
        )

    @app.exception_handler(IngestionCapError)
    async def ingestion_cap(
        request: Request, exc: IngestionCapError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={"detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(LLMBusyError)
    async def llm_busy(request: Request, exc: LLMBusyError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(redis.exceptions.ConnectionError)
    @app.exception_handler(redis.exceptions.TimeoutError)
    async def store_unavailable(
        request: Request, exc: redis.exceptions.RedisError
    ) -> JSONResponse:
        # The cause names the Redis host, so it is logged, never returned.
        logger.error(
            "Status store unreachable in %s: %s", request.url.path, exc
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Service temporarily unavailable. Please retry shortly."
                )
            },
            headers={
                "Retry-After": str(_STORE_UNAVAILABLE_RETRY_AFTER_SECONDS)
            },
        )

    @app.exception_handler(AgentResponseError)
    async def agent_error(
        request: Request, exc: AgentResponseError
    ) -> JSONResponse:
        logger.exception(
            "Agent failed to produce a response in %s", request.url.path
        )
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error in %s", request.url.path)
        return JSONResponse(
            status_code=500, content={"detail": "Internal server error"}
        )
