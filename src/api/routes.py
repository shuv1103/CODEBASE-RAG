from fastapi import APIRouter, Depends, Response, status

from .dependencies import RagWorkflowFacade, get_rag_workflow
from .schemas import (
    ChatRequest,
    ChatResponse,
    CreateRepoRequest,
    CreateRepoResponse,
    RepoStatusResponse,
)

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict:
    """Liveness check.

    Returns:
        {"status": "ok"}.
    """
    return {"status": "ok"}


@router.post("/repos", response_model=CreateRepoResponse)
def create_repo(
    payload: CreateRepoRequest,
    response: Response,
    workflow: RagWorkflowFacade = Depends(get_rag_workflow),
) -> CreateRepoResponse:
    """Start (or reuse) an ingestion for a GitHub repository.

    Args:
        payload: Request body holding the repo URL.
        response: Outgoing response, used to set the status code.
        workflow: Injected workflow facade.

    Returns:
        202 with status "pending" when a new ingestion was enqueued, or 200
        with the existing status when the repo is already ready/in flight.

    Raises:
        InvalidRepoUrlError: If the URL is not a github.com repository URL.
    """
    repo_id, repo_status, is_new, normalized_url = workflow.start_ingestion(payload.url)

    if not is_new:
        # Already ready, or already in flight — either way, no new ingestion.
        # See CLAUDE.md §10.4 (cost/dedup guardrail) and §10.7 (duplicate-ingestion guardrail).
        response.status_code = status.HTTP_200_OK
        return CreateRepoResponse(repo_id=repo_id, status=repo_status)

    # Enqueued onto Celery (via Redis) instead of FastAPI's BackgroundTasks —
    # runs in a separate worker process, survives an API restart, and scales
    # to multiple concurrent ingestions across worker replicas.
    workflow.enqueue_ingestion(repo_id, normalized_url)

    response.status_code = status.HTTP_202_ACCEPTED
    return CreateRepoResponse(repo_id=repo_id, status="pending")


@router.get("/repos/{repo_id}", response_model=RepoStatusResponse)
def get_repo(repo_id: str, workflow: RagWorkflowFacade = Depends(get_rag_workflow)) -> RepoStatusResponse:
    """Return a repo's current ingestion status.

    Args:
        repo_id: Repo to look up.
        workflow: Injected workflow facade.

    Returns:
        The repo's status record.

    Raises:
        RepoNotFoundError: If the repo_id is unknown.
    """
    record = workflow.get_status(repo_id)
    return RepoStatusResponse(**record.model_dump())


@router.post("/repos/{repo_id}/chat", response_model=ChatResponse)
async def chat(
    repo_id: str,
    payload: ChatRequest,
    workflow: RagWorkflowFacade = Depends(get_rag_workflow),
) -> ChatResponse:
    """Ask a question about an ingested repo.

    Args:
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
    session_id, answer, cached = await workflow.chat(repo_id, payload.message, payload.session_id)
    return ChatResponse(session_id=session_id, answer=answer, cached=cached)
