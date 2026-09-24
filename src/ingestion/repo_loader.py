import logging
import os
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

logger = logging.getLogger(__name__)

# Pydantic Models
class IngestionStats(BaseModel):
    """Counters collected while scanning and reading a repository.

    Attributes:
        total_files_scanned: Files seen outside skipped directories.
        files_supported: Files with a supported extension.
        files_read_success: Supported files read successfully.
        files_skipped_extension: Files skipped for an unsupported extension.
        files_skipped_size_or_error: Supported files skipped because they
            were too large or unreadable.
    """

    total_files_scanned: int = 0
    files_supported: int = 0
    files_read_success: int = 0
    files_skipped_extension: int = 0
    files_skipped_size_or_error: int = 0


class FileDocument(BaseModel):
    """One source file loaded from a repository.

    Attributes:
        file_name: Base name of the file.
        absolute_path: Absolute path on disk.
        relative_path: Path relative to the repository root.
        extension: Lowercased file extension, including the dot.
        language: Language detected from the extension.
        size_bytes: File size in bytes.
        content: Full text content of the file.
    """

    file_name: str
    absolute_path: str
    relative_path: str
    extension: str
    language: str
    size_bytes: int
    content: str


class IngestionResult(BaseModel):
    """Output of RepoLoader.load().

    Attributes:
        documents: Successfully loaded file documents.
        stats: Scan/read counters for the run.
    """

    documents: List[FileDocument] = Field(default_factory=list)
    stats: IngestionStats = Field(default_factory=IngestionStats)


class RepoConfig(BaseModel):
    """Validated repository location.

    Attributes:
        repo_path: Path to an existing local directory.
    """

    repo_path: str

    @model_validator(mode="after")
    def validate_repo_path(self) -> "RepoConfig":
        """Ensure repo_path exists and is a directory.

        Returns:
            The validated config.

        Raises:
            ValueError: If the path does not exist or is not a directory.
        """
        path = Path(self.repo_path).resolve()
        if not path.exists():
            raise ValueError(f"Repository path does not exist: {path}")
        if not path.is_dir():
            raise ValueError(f"Repository path must be a directory: {path}")
        return self

# Repository Loader
class RepoLoader:
    """Repository ingestion engine.

    Scans code repositories and produces structured file documents.

    Args:
        repo_path: Path to a local repository directory.

    Attributes:
        repo_root: Resolved absolute path of the repository.
        stats: Counters updated while loading.

    Raises:
        pydantic.ValidationError: If repo_path does not exist or is not a
            directory.

    Example:
        >>> loader = RepoLoader("path/to/repo")
        >>> result = loader.load()
        >>> print(result.stats.files_read_success)
    """

    def __init__(self, repo_path: str):
        config = RepoConfig(repo_path=repo_path)
        self.repo_root = Path(config.repo_path).resolve()
        self.stats = IngestionStats()

    def _is_in_skipped_directory(self, path: Path) -> bool:
        """Check whether any path component is in SKIP_DIRECTORIES.

        Args:
            path: Path located under repo_root.

        Returns:
            True if the path lives in a skipped directory.
        """
        return any(part in SKIP_DIRECTORIES for part in path.relative_to(self.repo_root).parts)

    def _scan_repository(self) -> List[Path]:
        """Recursively discover supported files, skipping configured directories.

        Updates the scan counters in self.stats as a side effect.

        Returns:
            Paths of files with a supported extension.
        """
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
        """Create a validated FileDocument from a file path.

        Args:
            file_path: Path of a supported file under repo_root.

        Returns:
            The FileDocument, or None if the file was too large or
            unreadable.
        """
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
        """Scan the repository and read every supported file.

        Returns:
            A validated IngestionResult containing documents and stats.
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
    from dotenv import load_dotenv

    from utils.logging_config import configure_logging

    load_dotenv()
    configure_logging()

    repo_path = os.getenv("HOST_REPO_PATH")
    if not repo_path:
        raise ValueError("HOST_REPO_PATH environment variable is required")

    result = RepoLoader(repo_path).load()
    logger.info("Stats: %s", result.stats)
    if result.documents:
        logger.debug("Sample document: %s", result.documents[0])
