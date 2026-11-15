"""Model for a chunk with its embedding vector."""

from pydantic import BaseModel


class EmbeddedChunk(BaseModel):
    """A code chunk with its embedding vector, ready to upsert into Qdrant.

    Attributes:
        chunk_type: Chunking strategy that produced the chunk.
        embedding_input_text: The exact text that was embedded (stored as the
            point's "document").
    """

    chunk_id: str
    relative_path: str
    file_name: str
    language: str
    chunk_type: str
    start_line: int | None = None
    end_line: int | None = None

    repo_id: str = ""
    embedding_input_text: str
    vector: list[float]
