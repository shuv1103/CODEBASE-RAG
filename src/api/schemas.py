from typing import Optional

from pydantic import BaseModel, Field, field_validator


class CreateRepoRequest(BaseModel):
    """Request body for POST /api/repos.

    Attributes:
        url: GitHub repository URL to ingest (whitespace-stripped).
    """

    url: str = Field(..., min_length=1, examples=["https://github.com/org/repo"])

    @field_validator("url")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject whitespace-only URLs and strip surrounding whitespace.

        Args:
            value: The submitted URL.

        Returns:
            The stripped URL.

        Raises:
            ValueError: If the URL is blank after stripping.
        """
        if not value.strip():
            raise ValueError("url cannot be blank")
        return value.strip()


class CreateRepoResponse(BaseModel):
    """Response body for POST /api/repos.

    Attributes:
        repo_id: Stable id derived from the normalized repo URL.
        status: Current ingestion status of the repo.
    """

    repo_id: str
    status: str


class RepoStatusResponse(BaseModel):
    """Response body for GET /api/repos/{repo_id}.

    Attributes:
        repo_id: Stable id derived from the normalized repo URL.
        url: Normalized repository URL.
        status: One of pending | cloning | chunking | embedding | ready |
            failed.
        chunks_total: Total chunks to embed, once chunking has finished.
        chunks_embedded: Chunks embedded so far.
        error: Failure reason, set only when status is "failed".
        created_at: UTC ISO timestamp of record creation.
        ready_at: UTC ISO timestamp when status became "ready".
    """

    repo_id: str
    url: str
    status: str
    chunks_total: Optional[int] = None
    chunks_embedded: Optional[int] = None
    error: Optional[str] = None
    created_at: str
    ready_at: Optional[str] = None


class ChatRequest(BaseModel):
    """Request body for POST /api/repos/{repo_id}/chat.

    Attributes:
        message: The user's question (whitespace-stripped).
        session_id: Existing ADK session to continue, or None to start one.
    """

    message: str = Field(..., min_length=1)
    session_id: Optional[str] = None

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        """Reject whitespace-only messages and strip surrounding whitespace.

        Args:
            value: The submitted message.

        Returns:
            The stripped message.

        Raises:
            ValueError: If the message is blank after stripping.
        """
        if not value.strip():
            raise ValueError("message cannot be blank")
        return value.strip()


class ChatResponse(BaseModel):
    """Response body for POST /api/repos/{repo_id}/chat.

    Attributes:
        session_id: ADK session id to send back on the next turn.
        answer: The agent's (or cache's) answer text.
        cached: True if the answer was served from the cache.
    """

    session_id: str
    answer: str
    cached: bool
