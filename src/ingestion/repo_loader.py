from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
from utils.config import SKIP_DIRECTORIES
from utils.file_utility import (
    is_supported_file,
    read_file_safe,
    get_relative_path,
    get_file_size,
)
from .language_detector import detect_language

# Pydantic Models
class IngestionStats(BaseModel):
    total_files_scanned: int = 0
    files_supported: int = 0
    files_read_success: int = 0
    files_skipped_extension: int = 0
    files_skipped_size_or_error: int = 0


class FileDocument(BaseModel):
    file_name: str
    absolute_path: str
    relative_path: str
    extension: str
    language: str
    size_bytes: int
    content: str


class IngestionResult(BaseModel):
    documents: List[FileDocument] = Field(default_factory=list)
    stats: IngestionStats = Field(default_factory=IngestionStats)


class RepoConfig(BaseModel):
    repo_path: str

    @model_validator(mode="after")
    def validate_repo_path(self) -> "RepoConfig":
        path = Path(self.repo_path).resolve()
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Repository path must be a directory: {path}")
        return self

# Repository Loader
class RepoLoader:
    """
    Repository ingestion engine.
    Scans code repositories and produces structured file documents.
    """

    def __init__(self, repo_path: str):
        config = RepoConfig(repo_path=repo_path)
        self.repo_root = Path(config.repo_path).resolve()
        self.stats = IngestionStats()

    def _is_in_skipped_directory(self, path: Path) -> bool:
        return any(part in SKIP_DIRECTORIES for part in path.relative_to(self.repo_root).parts)

    def _scan_repository(self) -> List[Path]:
        """Recursively discover supported files, skipping configured directories."""
        files: List[Path] = []

        for path in self.repo_root.rglob("*"):
            if path.is_dir():
                continue

            if self._is_in_skipped_directory(path):
                continue

            self.stats.total_files_scanned += 1

            if not is_supported_file(path):
                self.stats.files_skipped_extension += 1
                continue

            self.stats.files_supported += 1
            files.append(path)

        return files

    def _build_file_document(self, file_path: Path) -> Optional[FileDocument]:
        """Create a validated FileDocument from a file path."""
        content = read_file_safe(file_path)

        if content is None:
            self.stats.files_skipped_size_or_error += 1
            return None

        self.stats.files_read_success += 1

        return FileDocument(
            file_name=file_path.name,
            absolute_path=str(file_path),
            relative_path=get_relative_path(self.repo_root, file_path),
            extension=file_path.suffix.lower(),
            language=detect_language(file_path),
            size_bytes=get_file_size(file_path),
            content=content,
        )

    def load(self) -> IngestionResult:
        """
        Main ingestion pipeline.
        Returns a validated IngestionResult containing documents and stats.
        """
        files = self._scan_repository()

        documents: List[FileDocument] = []
        for file_path in files:
            doc = self._build_file_document(file_path)
            if doc:
                documents.append(doc)

        return IngestionResult(documents=documents, stats=self.stats)

# Main entry point
if __name__ == "__main__":

    repo_path = r"D:\CODEBASE-RAG-PROJECT\CODEBASE-RAG\public\Hospital-Management-System-React-and-SpringBoot-master"
    loader = RepoLoader(repo_path)
    result = loader.load()

    print("Stats:")
    print(result.stats)

    # print("\nSample Document:")
    # print(result.documents[0])
