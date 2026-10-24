"""Model for a chunk with its embedding vector."""

from pydantic import BaseModel, Field

from utils.time_utils import utc_now_iso


class EmbeddedChunk(BaseModel):
    """A code chunk together with its embedding vector, ready to upsert.

    Attributes:
        chunk_id: Unique id of the chunk.
        relative_path: File path relative to the repository root.
        file_name: Base name of the source file.
        language: Detected programming language.
        chunk_type: Chunking strategy that produced the chunk.
        symbol_name: Function/class name, if known.
        parent_symbol: Enclosing symbol name, if known.
        start_line: First line of the chunk, if known.
        end_line: Last line of the chunk, if known.
        repo_id: Id of the repository the chunk belongs to.
        source_url: Normalized URL of the source repository.
        content: Raw chunk text (kept for debugging).
        embedding_input_text: The exact text that was embedded.
        embedding_model: Name of the embedding model used.
        embedding_dimension: Length of the vector.
        vector: The embedding vector.
        token_estimate: Rough token count of embedding_input_text.
        created_at: UTC ISO timestamp of creation.
    """

    chunk_id: str
    relative_path: str
    file_name: str
    language: str
    chunk_type: str
    symbol_name: str | None = None
    parent_symbol: str | None = None
    start_line: int | None = None
    end_line: int | None = None

    repo_id: str = ""
    source_url: str = ""

    content: str | None = None
    embedding_input_text: str

    embedding_model: str
    embedding_dimension: int
    vector: list[float]

    token_estimate: int | None = None
    created_at: str = Field(default_factory=utc_now_iso)

