"""Loader for a cloned repo's supported source files."""

import logging
import os
from pathlib import Path

from models.ingestion_models import (
    FileDocument,
    IngestionResult,
    IngestionStats,
)
from utils.config import SKIP_DIRECTORIES, settings
from utils.file_utility import (
    detect_language,
    is_supported_file,
    read_file_safe,
)

logger = logging.getLogger(__name__)


class RepoLoader:
    """Reads a local repo's supported source files as documents.

    Args:
        repo_path: Path of an existing local directory.

    Raises:
        ValueError: If repo_path is not an existing directory.
    """

    def __init__(self, repo_path: str) -> None:
        self.repo_root = Path(repo_path).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(
                f"Repository path is not an existing directory: "
                f"{self.repo_root}"
            )
        self.stats = IngestionStats()

    def scan_repository(self) -> list[Path]:
        """Return the repo's supported files, updating the scan counters.

        Skips SKIP_DIRECTORIES, symlinks, and paths resolving outside the repo.

        Returns:
            Paths of supported files, in sorted walk order.
        """
        files: list[Path] = []

        # followlinks=False never descends into a linked directory.
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            # Pruned in place, so skipped trees are never listed.
            dirnames[:] = sorted(
                d for d in dirnames if d not in SKIP_DIRECTORIES
            )
            for name in sorted(filenames):
                path = Path(dirpath, name)
                # e.g. leak.py -> /proc/self/environ would embed secrets.
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
        """Read one file into a FileDocument.

        Args:
            file_path: A supported file under repo_root.

        Returns:
            The document, or None if the file is too large or unreadable.
        """
        content = read_file_safe(file_path)

        if content is None:
            self.stats.files_skipped_size_or_error += 1
            return None

        self.stats.files_read_success += 1

        return FileDocument(
            file_name=file_path.name,
            absolute_path=str(file_path),
            relative_path=str(file_path.relative_to(self.repo_root)),
            extension=file_path.suffix.lower(),
            language=detect_language(file_path),
            size_bytes=file_path.stat().st_size,
            content=content,
        )

    def load(self) -> IngestionResult:
        """Scan the repo and read every supported file.

        Returns:
            IngestionResult with the documents and stats.
        """
        documents = [
            doc
            for file_path in self.scan_repository()
            if (doc := self.build_file_document(file_path)) is not None
        ]

        return IngestionResult(documents=documents, stats=self.stats)


def main() -> None:
    """Load HOST_REPO_PATH and log the stats (local debugging).

    Raises:
        ValueError: If HOST_REPO_PATH is not set.
    """
    from utils.logging_config import configure_logging

    configure_logging()

    repo_path = settings.host_repo_path
    if not repo_path:
        raise ValueError("HOST_REPO_PATH environment variable is required")

    result = RepoLoader(repo_path).load()
    logger.info("Stats: %s", result.stats)
    if result.documents:
        logger.debug("Sample document: %s", result.documents[0])


if __name__ == "__main__":
    main()
