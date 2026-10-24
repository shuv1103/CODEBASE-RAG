"""Repository cloning, file loading and the end-to-end ingestion pipeline.

ingest_pipeline is not re-exported: it imports chunking, which imports this
package's repo_loader, so re-exporting it here would be a circular import.
Import it as ingestion.ingest_pipeline.
"""

from .git_cloner import GitCloneError, RepoTooLargeError, clone_repo
from .language_detector import detect_language
from .repo_loader import (
    FileDocument,
    IngestionResult,
    IngestionStats,
    RepoConfig,
    RepoLoader,
)

__all__ = [
    "FileDocument",
    "GitCloneError",
    "IngestionResult",
    "IngestionStats",
    "RepoConfig",
    "RepoLoader",
    "RepoTooLargeError",
    "clone_repo",
    "detect_language",
]
