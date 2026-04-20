from datetime import datetime, timezone
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EmbeddedChunk(BaseModel):
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

    # Useful for debugging
    content: Optional[str] = None
    embedding_input_text: str

    embedding_model: str
    embedding_dimension: int
    vector: List[float]

    token_estimate: Optional[int] = None
    created_at: str = Field(default_factory=utc_now_iso)


class EmbeddingStats(BaseModel):
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
    embedded_chunks: List[EmbeddedChunk] = Field(default_factory=list)
    stats: EmbeddingStats = Field(default_factory=EmbeddingStats)
