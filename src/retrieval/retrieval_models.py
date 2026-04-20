from typing import List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
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
    query: str
    top_k: int
    results: List[RetrievedChunk] = Field(default_factory=list)
