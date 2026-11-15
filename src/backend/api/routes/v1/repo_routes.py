"""v1 API routes: health, repo ingestion, status and chat."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request, Response, status

from api.core.dependencies import RagWorkflowFacade, get_rag_workflow
from api.core.rate_limit import admit_new_ingestion, global_key, limiter
from api.schemas.repo_schemas import (
    ChatRequest,
    ChatResponse,
    CreateRepoRequest,
    CreateRepoResponse,
    RepoStatusResponse,
)
from storage.status_store import RepoStatus
from utils.config import settings

router = APIRouter(prefix="/api")

Workflow = Annotated[RagWorkflowFacade, Depends(get_rag_workflow)]

# compute_repo_id's 12 hex chars; anything else is a 422 before any I/O.
RepoId = Annotated[str, Path(pattern=r"^[0-9a-f]{12}$")]


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
    workflow: Workflow,
) -> CreateRepoResponse:
    """Start an ingestion for a GitHub repo, or report the existing one.

    Limited per IP (CREATE_REPO_RATE_LIMIT); new ingestions also count
    against the global GLOBAL_CREATE_REPO_RATE_LIMIT.

    Args:
        request: Incoming request; the limiter reads the client IP from it.
        payload: Body {"url": "https://github.com/<org>/<repo>"}.
        response: Outgoing response, used to set the status code.
        workflow: Injected workflow facade.

    Returns:
        {repo_id, status}: 202 with "pending" when an ingestion was queued, or
        200 with the current status of an existing repo.

    Raises:
        InvalidRepoUrlError: If the URL is not a github.com repository URL.
        IngestionCapError: If the global cap on new ingestions is reached.
    """
    started = workflow.start_ingestion(payload.url, admit_new_ingestion)

    if not started.is_new:
        response.status_code = status.HTTP_200_OK
        return CreateRepoResponse(
            repo_id=started.repo_id, status=started.status
        )

    workflow.enqueue_ingestion(
        started.repo_id, started.normalized_url, started.attempt_id
    )

    response.status_code = status.HTTP_202_ACCEPTED
    return CreateRepoResponse(repo_id=started.repo_id, status="pending")


@router.get("/repos/{repo_id}", response_model=RepoStatusResponse)
def get_repo(repo_id: RepoId, workflow: Workflow) -> RepoStatus:
    """Return a repo's ingestion status and progress.

    Args:
        repo_id: 12-hex repo id.
        workflow: Injected workflow facade.

    Returns:
        The record; response_model hides updated_at and attempt_id.

    Raises:
        RepoNotFoundError: If repo_id is unknown.
    """
    return workflow.get_status(repo_id)


@router.post("/repos/{repo_id}/chat", response_model=ChatResponse)
@limiter.limit(settings.chat_rate_limit)
@limiter.limit(settings.global_chat_rate_limit, key_func=global_key)
async def chat(
    request: Request,
    repo_id: RepoId,
    payload: ChatRequest,
    workflow: Workflow,
) -> ChatResponse:
    """Answer a question about a ready repo.

    Limited per IP (CHAT_RATE_LIMIT) and globally (GLOBAL_CHAT_RATE_LIMIT).

    Args:
        request: Incoming request; the limiter reads the client IP from it.
        repo_id: 12-hex id of a "ready" repo.
        payload: Body {"message": str, "session_id": str | null}.
        workflow: Injected workflow facade.

    Returns:
        {session_id, answer, cache_hit}; send session_id on the next turn.

    Raises:
        RepoNotFoundError: If repo_id is unknown.
        RepoNotReadyError: If the repo is not "ready".
        AgentResponseError: If the agent produced no final answer.
    """
    turn = await workflow.chat(repo_id, payload.message, payload.session_id)
    return ChatResponse(**turn._asdict())
