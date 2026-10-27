"""Loader for a cloned repo's supported source files."""

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from utils.config import SKIP_DIRECTORIES, settings
from utils.file_utility import (
    get_file_size,
    get_relative_path,
    is_supported_file,
    read_file_safe,
)

from .language_detector import detect_language

logger = logging.getLogger(__name__)


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

    documents: list[FileDocument] = Field(default_factory=list)
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


class RepoLoader:
    """Loader that reads a local repo's supported source files as documents.

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

    def __init__(self, repo_path: str) -> None:
        config = RepoConfig(repo_path=repo_path)
        self.repo_root = Path(config.repo_path).resolve()
        self.stats = IngestionStats()

    def scan_repository(self) -> list[Path]:
        """Recursively discover supported files, skipping configured dirs.

        Skip symlinks and paths resolving outside repo_root without counting
        them, and update the scan counters in self.stats.

        Returns:
            Paths of files with a supported extension.
        """
        files: list[Path] = []

        # followlinks=False never descends into a linked directory.
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            # Pruned in place, so skipped trees (node_modules, .git) are
            # never listed at all.
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRECTORIES
            )
            for name in sorted(filenames):
                path = Path(dirpath, name)
                # A repo can ship a link like leak.py -> /proc/self/environ
                # that would embed the worker's secrets.
                if path.is_symlink() or not path.resolve().is_relative_to(
                    self.repo_root
                ):
                    logger.debug("Skipping link or out-of-repo path: %s", path)
                    continue

                self.stats.total_files_scanned += 1

                if not is_supported_file(path):
                    self.stats.files_skipped_extension += 1
                    continue

                self.stats.files_supported += 1
                files.append(path)

        return files

    def build_file_document(self, file_path: Path) -> FileDocument | None:
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
        documents = [
            doc
            for file_path in self.scan_repository()
            if (doc := self.build_file_document(file_path)) is not None
        ]

        return IngestionResult(documents=documents, stats=self.stats)


if __name__ == "__main__":
    from utils.logging_config import configure_logging

    configure_logging()

    repo_path = settings.host_repo_path
    if not repo_path:
        raise ValueError("HOST_REPO_PATH environment variable is required")

    result = RepoLoader(repo_path).load()
    logger.info("Stats: %s", result.stats)
    if result.documents:
        logger.debug("Sample document: %s", result.documents[0])
