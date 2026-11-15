"""Models for retrieved chunks and retrieval responses."""

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """One search hit.

    Attributes:
        content: The stored text (location headers plus code).
        similarity_score: Cosine similarity; higher is closer.
    """

    chunk_id: str
    relative_path: str
    file_name: str
    language: str
    chunk_type: str
    start_line: int | None = None
    end_line: int | None = None
    content: str | None = None
    similarity_score: float


class RetrievalResponse(BaseModel):
    """Ranked results for one query.

    Attributes:
        results: Retrieved chunks, most relevant first.
    """

    results: list[RetrievedChunk] = Field(default_factory=list)
