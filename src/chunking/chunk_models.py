from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    """Metadata describing one code chunk and the file it came from.

    Attributes:
        chunk_id: Unique id, "<relative_path>::chunk::<index>".
        file_name: Base name of the source file.
        absolute_path: Absolute path of the source file on disk.
        relative_path: Path relative to the repository root.
        extension: Lowercased file extension, including the dot.
        language: Detected programming language.
        size_bytes: Source file size in bytes.
        chunk_index: Zero-based position of the chunk within its file.
        total_chunks_in_file: Number of chunks produced for the file.
        chunking_strategy: Strategy that produced the chunk
            (e.g. "chonkie_ast").
        symbol_name: Function/class name, if known.
        start_line: Zero-based line where the chunk starts, if known.
        end_line: Zero-based line where the chunk ends, if known.
        repo_id: Id of the repository the chunk belongs to.
        source_url: Normalized URL of the source repository.
        extra: Additional strategy-specific data (e.g. token_count).
    """

    chunk_id: str
    file_name: str
    absolute_path: str
    relative_path: str
    extension: str
    language: str
    size_bytes: int

    chunk_index: int
    total_chunks_in_file: int

    chunking_strategy: str  # "chonkie_ast"
    symbol_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    repo_id: str = ""
    source_url: str = ""

    extra: Dict[str, Any] = Field(default_factory=dict)


class CodeChunk(BaseModel):
    """A chunk of source code plus its metadata.

    Attributes:
        content: The chunk's source text.
        metadata: Where the chunk came from and how it was produced.
    """

    content: str
    metadata: ChunkMetadata


class ChunkingStats(BaseModel):
    """Counters collected during a chunking run.

    Attributes:
        total_documents_received: Documents passed in for chunking.
        total_ast_eligible_documents: Documents in an AST-supported language.
        total_documents_chunked: Documents that produced at least one chunk.
        total_chunks_created: Total chunks produced.
        skipped_unsupported_language: Documents skipped for their language.
        failed_files: Documents whose chunking crashed, hung, or errored.
        language_distribution: Document count per detected language.
    """

    total_documents_received: int = 0
    total_ast_eligible_documents: int = 0
    total_documents_chunked: int = 0
    total_chunks_created: int = 0

    skipped_unsupported_language: int = 0
    failed_files: int = 0

    language_distribution: Dict[str, int] = Field(default_factory=dict)


class ChunkingResult(BaseModel):
    """Output of ASTCodeChunker.chunk().

    Attributes:
        chunks: All chunks produced, across every document.
        stats: Counters for the run.
    """

    chunks: List[CodeChunk] = Field(default_factory=list)
    stats: ChunkingStats = Field(default_factory=ChunkingStats)
