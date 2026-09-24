from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """A single code chunk returned from a vector-store similarity search.

    Attributes:
        chunk_id: Unique id of the chunk in the vector store.
        relative_path: File path relative to the repository root.
        file_name: Base name of the source file.
        language: Detected programming language.
        chunk_type: Chunking strategy that produced the chunk (e.g. "ast").
        symbol_name: Function/class name, if the chunk maps to one.
        parent_symbol: Enclosing symbol name, if any.
        start_line: First source line of the chunk, if known.
        end_line: Last source line of the chunk, if known.
        content: The stored document text.
        embedding_input_text: The exact text that was embedded.
        similarity_score: Cosine similarity reported by Qdrant (higher is
            more similar).
    """

    chunk_id: str
    relative_path: str
    file_name: str
    language: str
    chunk_type: str
    symbol_name: Optional[str] = None
    parent_symbol: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content: Optional[str] = None
    embedding_input_text: Optional[str] = None
    similarity_score: float


class RetrievalResponse(BaseModel):
    """Ranked retrieval results for one query.

    Attributes:
        query: The (stripped) query that was searched.
        top_k: Number of results requested.
        results: Retrieved chunks, most relevant first.
    """

    query: str
    top_k: int
    results: List[RetrievedChunk] = Field(default_factory=list)
