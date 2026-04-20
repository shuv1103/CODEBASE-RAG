from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    chunk_id: str
    file_name: str
    absolute_path: str
    relative_path: str
    extension: str
    language: str
    size_bytes: int

    chunk_index: int
    total_chunks_in_file: int

    chunking_strategy: str  # "ast"
    symbol_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    extra: Dict[str, Any] = Field(default_factory=dict)


class CodeChunk(BaseModel):
    content: str
    metadata: ChunkMetadata


class ChunkingStats(BaseModel):
    total_documents_received: int = 0
    total_ast_eligible_documents: int = 0
    total_documents_chunked: int = 0
    total_chunks_created: int = 0

    skipped_unsupported_language: int = 0
    failed_files: int = 0

    language_distribution: Dict[str, int] = Field(default_factory=dict)


class ChunkingResult(BaseModel):
    chunks: List[CodeChunk] = Field(default_factory=list)
    stats: ChunkingStats = Field(default_factory=ChunkingStats)
