"""Models for code chunks and chunking statistics."""

from typing import Any

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Where a chunk came from and how it was made.

    Attributes:
        chunk_id: "<relative_path>::chunk::<index>", unique within a repo.
        chunking_strategy: Producer, e.g. "chonkie_ast".
        start_line: 1-based first line, if known.
        end_line: 1-based last line (inclusive), if known.
        extra: Strategy-specific data, e.g. {"token_count": 312}.
    """

    chunk_id: str
    file_name: str
    relative_path: str
    language: str

    chunking_strategy: str
    start_line: int | None = None
    end_line: int | None = None

    repo_id: str = ""

    extra: dict[str, Any] = Field(default_factory=dict)


class CodeChunk(BaseModel):
    """A chunk of source code plus its metadata."""

    content: str
    metadata: ChunkMetadata


class ChunkingStats(BaseModel):
    """Counters from one chunking run.

    Attributes:
        total_documents_chunked: Documents that produced chunks.
        failed_files: Documents whose chunking crashed, hung or errored.
    """

    total_documents_chunked: int = 0
    total_chunks_created: int = 0

    failed_files: int = 0


class ChunkingResult(BaseModel):
    """Output of ASTCodeChunker.chunk(): every chunk plus the run stats."""

    chunks: list[CodeChunk] = Field(default_factory=list)
    stats: ChunkingStats = Field(default_factory=ChunkingStats)
