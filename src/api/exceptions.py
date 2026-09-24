import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class InvalidRepoUrlError(Exception):
    """Raised when a submitted URL is not a valid github.com repository URL."""


class RepoNotFoundError(Exception):
    """Raised when a repo_id has no status record (mapped to HTTP 404).

    Args:
        repo_id: The unknown repo id.

    Attributes:
        repo_id: The unknown repo id.
    """

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        super().__init__(f"Unknown repo_id: {repo_id}")


class RepoNotReadyError(Exception):
    """Raised when chatting with a repo that is not "ready" (mapped to HTTP 409).

    Args:
        repo_id: The repo that was queried.
        status: Its current ingestion status.

    Attributes:
        repo_id: The repo that was queried.
        status: Its current ingestion status.
    """

    def __init__(self, repo_id: str, status: str):
        self.repo_id = repo_id
        self.status = status
        super().__init__(f"Repo {repo_id} is not ready (status={status})")


class AgentResponseError(Exception):
    """Raised when the agent run completes without producing a final answer."""


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain exceptions to JSON HTTP error responses.

    InvalidRepoUrlError -> 400, RepoNotFoundError -> 404,
    RepoNotReadyError -> 409, AgentResponseError -> 502, and any other
    exception -> 500 with a generic message.

    Args:
        app: The FastAPI application to register handlers on.
    """
    @app.exception_handler(InvalidRepoUrlError)
    async def _invalid_url(request: Request, exc: InvalidRepoUrlError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RepoNotFoundError)
    async def _not_found(request: Request, exc: RepoNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(RepoNotReadyError)
    async def _not_ready(request: Request, exc: RepoNotReadyError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(AgentResponseError)
    async def _agent_error(request: Request, exc: AgentResponseError) -> JSONResponse:
        logger.exception("Agent failed to produce a response")
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error in %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
