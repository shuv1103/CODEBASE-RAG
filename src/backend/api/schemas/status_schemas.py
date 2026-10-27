"""Request and response models for the v1 API."""

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from storage.status_store import PublicRepoStatus

# A message goes straight into the LLM prompt, so its cap is also a cost
# limit. Real GitHub URLs and session ids (UUIDs) are far shorter.
MAX_URL_LENGTH = 200
MAX_MESSAGE_LENGTH = 2000
MAX_SESSION_ID_LENGTH = 100


def strip_not_blank(value: str, info: ValidationInfo) -> str:
    """Strip surrounding whitespace and reject blank input (field validator).

    Args:
        value: The submitted field value.
        info: Pydantic validation context (supplies the field name).

    Returns:
        The stripped value.

    Raises:
        ValueError: If the value is blank after stripping.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{info.field_name} cannot be blank")
    return stripped


class CreateRepoRequest(BaseModel):
    """Request body for POST /api/repos.

    Attributes:
        url: GitHub repository URL to ingest (whitespace-stripped), at most
            MAX_URL_LENGTH characters.
    """

    url: str = Field(
        ...,
        min_length=1,
        max_length=MAX_URL_LENGTH,
        examples=["https://github.com/org/repo"],
    )

    _url_not_blank = field_validator("url")(strip_not_blank)


class CreateRepoResponse(BaseModel):
    """Response body for POST /api/repos.

    Attributes:
        repo_id: Stable id derived from the normalized repo URL.
        status: Current ingestion status of the repo.
    """

    repo_id: str
    status: str


class RepoStatusResponse(PublicRepoStatus):
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


class ChatRequest(BaseModel):
    """Request body for POST /api/repos/{repo_id}/chat.

    Attributes:
        message: The user's question (whitespace-stripped), at most
            MAX_MESSAGE_LENGTH characters.
        session_id: Existing ADK session to continue, or None to start one;
            at most MAX_SESSION_ID_LENGTH characters.
    """

    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(None, max_length=MAX_SESSION_ID_LENGTH)

    _message_not_blank = field_validator("message")(strip_not_blank)


class ChatResponse(BaseModel):
    """Response body for POST /api/repos/{repo_id}/chat.

    Attributes:
        session_id: ADK session id to send back on the next turn.
        answer: The agent's (or cache's) answer text.
        cache_hit: "exact" or "semantic" when the cache answered, None when
            the agent did.
    """

    session_id: str
    answer: str
    cache_hit: Literal["exact", "semantic"] | None
