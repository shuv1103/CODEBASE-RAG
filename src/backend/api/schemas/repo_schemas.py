"""Request and response models for the v1 API."""

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from caching.semantic_cache_repository import CacheStrategy
from storage.status_store import PublicRepoStatus, RepoStatusValue

# The message cap is also an LLM cost cap.
MAX_URL_LENGTH = 200
MAX_MESSAGE_LENGTH = 2000
MAX_SESSION_ID_LENGTH = 100


def strip_not_blank(value: str, info: ValidationInfo) -> str:
    """Strip surrounding whitespace and reject a blank value (field validator).

    Args:
        value: The submitted field value.
        info: Pydantic context; supplies the field name for the error.

    Returns:
        The stripped value.

    Raises:
        ValueError: If the value is blank.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{info.field_name} cannot be blank")
    return stripped


class CreateRepoRequest(BaseModel):
    """Body of POST /api/repos: the GitHub repo URL (at most 200 chars)."""

    url: str = Field(
        ...,
        min_length=1,
        max_length=MAX_URL_LENGTH,
        examples=["https://github.com/org/repo"],
    )

    _url_not_blank = field_validator("url")(strip_not_blank)


class CreateRepoResponse(BaseModel):
    """Body returned by POST /api/repos."""

    repo_id: str
    status: RepoStatusValue


class RepoStatusResponse(PublicRepoStatus):
    """Body returned by GET /api/repos/{repo_id}."""


class ChatRequest(BaseModel):
    """Body of POST /api/repos/{repo_id}/chat.

    Attributes:
        message: The question, at most 2000 chars.
        session_id: Session to continue; omit to start a new one.
    """

    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    session_id: str | None = Field(None, max_length=MAX_SESSION_ID_LENGTH)

    _message_not_blank = field_validator("message")(strip_not_blank)


class ChatResponse(BaseModel):
    """Body returned by chat.

    Attributes:
        session_id: Session to send on the next turn.
        answer: The answer text.
        cache_hit: "exact" or "semantic" when the cache answered, else None.
    """

    session_id: str
    answer: str
    cache_hit: CacheStrategy | None
