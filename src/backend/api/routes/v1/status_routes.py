"""v1 API routes: health, repo ingestion, status and chat."""

from fastapi import APIRouter, Depends, Request, Response, status

from utils.config import settings

from ...core.dependencies import RagWorkflowFacade, get_rag_workflow
from ...core.rate_limit import admit_new_ingestion, global_key, limiter
from ...schemas.status_schemas import (
    ChatRequest,
    ChatResponse,
    CreateRepoRequest,
    CreateRepoResponse,
    RepoStatusResponse,
)

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the API process is alive.

    Returns:
        {"status": "ok"}.
    """
    return {"status": "ok"}


@router.post("/repos", response_model=CreateRepoResponse)
@limiter.limit(settings.create_repo_rate_limit)
def create_repo(
    request: Request,
    payload: CreateRepoRequest,
    response: Response,
    workflow: RagWorkflowFacade = Depends(get_rag_workflow),
) -> CreateRepoResponse:
    """Start (or reuse) an ingestion for a GitHub repository.

    Rate-limited per client IP (CREATE_REPO_RATE_LIMIT). New ingestions are
    also capped across all clients (GLOBAL_CREATE_REPO_RATE_LIMIT), which
    bounds total ingestion spend however many IPs submit repos; reopening
    an existing repo doesn't count against that cap.

    Args:
        request: Incoming request; the rate limiter reads the client IP.
        payload: Request body holding the repo URL.
        response: Outgoing response, used to set the status code.
        workflow: Injected workflow facade.

    Returns:
        202 with status "pending" when a new ingestion was enqueued, or 200
        with the existing status when the repo is already ready/in flight.

    Raises:
        InvalidRepoUrlError: If the URL is not a github.com repository URL.
        IngestionCapError: If the global cap on new ingestions is reached.
    """
    repo_id, repo_status, is_new, normalized_url, attempt_id = (
        workflow.start_ingestion(payload.url, admit_new_ingestion)
    )

    if not is_new:
        response.status_code = status.HTTP_200_OK
        return CreateRepoResponse(repo_id=repo_id, status=repo_status)

    # A Celery worker runs it, so it survives an API restart and scales out.
    workflow.enqueue_ingestion(repo_id, normalized_url, attempt_id)

    response.status_code = status.HTTP_202_ACCEPTED
    return CreateRepoResponse(repo_id=repo_id, status="pending")


@router.get("/repos/{repo_id}", response_model=RepoStatusResponse)
def get_repo(
    repo_id: str, workflow: RagWorkflowFacade = Depends(get_rag_workflow)
) -> RepoStatusResponse:
    """Return a repo's current ingestion status.

    Args:
        repo_id: Repo to look up.
        workflow: Injected workflow facade.

    Returns:
        The repo's status record.

    Raises:
        RepoNotFoundError: If the repo_id is unknown.
    """
    # response_model hides the internal updated_at and attempt_id fields.
    return workflow.get_status(repo_id)


@router.post("/repos/{repo_id}/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
@limiter.limit(settings.global_chat_rate_limit, key_func=global_key)
async def chat(
    request: Request,
    repo_id: str,
    payload: ChatRequest,
    workflow: RagWorkflowFacade = Depends(get_rag_workflow),
) -> ChatResponse:
    """Ask a question about an ingested repo.

    Rate-limited per client IP (CHAT_RATE_LIMIT) and across all clients
    (GLOBAL_CHAT_RATE_LIMIT).

    Args:
        request: Incoming request; the rate limiter reads the client IP.
        repo_id: Repo to chat about; must be "ready".
        payload: Request body holding the message and optional session_id.
        workflow: Injected workflow facade.

    Returns:
        The answer, the session id to reuse, and whether it was cached.

    Raises:
        RepoNotFoundError: If the repo_id is unknown.
        RepoNotReadyError: If the repo is not "ready".
        AgentResponseError: If the agent produced no final answer.
    """
    session_id, answer, cached = await workflow.chat(
        repo_id, payload.message, payload.session_id
    )
    return ChatResponse(session_id=session_id, answer=answer, cached=cached)
