"""Models for files loaded from a repo and the scan statistics."""

from pydantic import BaseModel, Field


class IngestionStats(BaseModel):
    """Counters collected while scanning and reading a repo."""

    total_files_scanned: int = 0
    files_supported: int = 0
    files_read_success: int = 0
    files_skipped_extension: int = 0
    files_skipped_size_or_error: int = 0


class FileDocument(BaseModel):
    """One source file loaded from a repo.

    Attributes:
        relative_path: Path relative to the repo root.
        extension: Lowercased extension, including the dot.
        language: Detected from the extension, e.g. "python".
        content: The file's full UTF-8 text.
    """

    file_name: str
    absolute_path: str
    relative_path: str
    extension: str
    language: str
    size_bytes: int
    content: str


class IngestionResult(BaseModel):
    """Output of RepoLoader.load(): the documents plus the scan stats."""

    documents: list[FileDocument] = Field(default_factory=list)
    stats: IngestionStats = Field(default_factory=IngestionStats)
