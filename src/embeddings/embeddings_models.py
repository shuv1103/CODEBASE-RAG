from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        Timezone-aware ISO timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


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
    symbol_name: Optional[str] = None
    parent_symbol: Optional[str] = None
    # Some chunkers may not provide line numbers (e.g., binary/blob chunks).
    # Make these optional to accept chunks without line meta.
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    repo_id: str = ""
    source_url: str = ""

    # Useful for debugging
    content: Optional[str] = None
    embedding_input_text: str

    embedding_model: str
    embedding_dimension: int
    vector: List[float]

    token_estimate: Optional[int] = None
    created_at: str = Field(default_factory=utc_now_iso)


class EmbeddingStats(BaseModel):
    """Counters collected during an embedding run.

    Attributes:
        total_chunks_loaded: Chunks read for embedding.
        chunks_embedded_success: Chunks embedded and stored.
        chunks_failed: Chunks in batches that failed to embed.
        batches_processed: Number of batches attempted.
        model_name: Embedding model used.
        embedding_dimension: Length of the produced vectors.
        avg_batch_size: Mean number of chunks per batch.
        languages_distribution: Embedded chunk count per language.
        chunk_type_distribution: Embedded chunk count per chunk type.
    """

    total_chunks_loaded: int = 0
    chunks_embedded_success: int = 0
    chunks_failed: int = 0
    batches_processed: int = 0

    model_name: str = ""
    embedding_dimension: int = 0
    avg_batch_size: float = 0.0

    languages_distribution: Dict[str, int] = Field(default_factory=dict)
    chunk_type_distribution: Dict[str, int] = Field(default_factory=dict)


class EmbeddingResult(BaseModel):
    """Output of an embedding run.

    Attributes:
        embedded_chunks: Chunks with their vectors.
        stats: Counters for the run.
    """

    embedded_chunks: List[EmbeddedChunk] = Field(default_factory=list)
    stats: EmbeddingStats = Field(default_factory=EmbeddingStats)
