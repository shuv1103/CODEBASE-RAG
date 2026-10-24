"""Ingestion pipeline: clone, chunk, embed and store one repo."""

import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chunking.chunker import ASTCodeChunker
from embeddings.embeddings_client import EmbeddingService, embed_chunks
from embeddings.qdrant_store import QdrantVectorStore
from utils.config import settings
from utils.iter_utils import batchify
from utils.rate_limiter import ExponentialBackoff

from .git_cloner import clone_repo
from .repo_loader import RepoLoader

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None] | None
# report(status, **fields): the stage-facing wrapper around ProgressCallback.
Reporter = Callable[..., None]


class IngestionError(Exception):
    """Raised when a repo cannot be ingested.

    For example it is empty or unsupported, or a pipeline step failed.
    """


class IngestionTimeoutError(IngestionError):
    """Raised when ingestion exceeds INGESTION_TIMEOUT_SECONDS."""


def clone_stage(source_url: str, report: Reporter) -> Path:
    """Shallow-clone the repo into a temp dir (pipeline stage 1).

    The caller checks the timeout once it holds the path, so a timeout
    can't leak the clone.

    Args:
        source_url: Normalized repository URL.
        report: Progress callback, called with the stage status.

    Returns:
        Path of the temp clone directory.

    Raises:
        GitCloneError: If the clone fails or times out.
        RepoTooLargeError: If the repo exceeds the size caps.
    """
    report("cloning")
    return clone_repo(source_url)


def chunk_stage(
    local_path: Path,
    repo_id: str,
    source_url: str,
    check_timeout: Callable[[], None],
    report: Reporter,
) -> list[dict[str, Any]]:
    """Load the repo's files and chunk them by AST (pipeline stage 2).

    Args:
        local_path: Directory holding the cloned repo.
        repo_id: Repository id to stamp on each chunk.
        source_url: Repository URL to stamp on each chunk.
        check_timeout: Raises IngestionTimeoutError once the budget is spent.
        report: Progress callback, called with the stage status.

    Returns:
        Chunks serialized as dicts (CodeChunk.model_dump()).

    Raises:
        IngestionError: If no AST-chunkable source files were found, or the
            repo produces more than MAX_REPO_CHUNKS chunks (caps embedding
            spend).
        IngestionTimeoutError: If the ingestion time budget is exceeded.
    """
    report("chunking")
    ingestion_result = RepoLoader(str(local_path)).load()

    def after_batch() -> None:
        """Enforce the timeout and refresh the record's heartbeat."""
        check_timeout()
        report("chunking")

    chunking_result = ASTCodeChunker().chunk(
        ingestion_result,
        repo_id=repo_id,
        source_url=source_url,
        on_batch=after_batch,
    )
    check_timeout()

    if not chunking_result.chunks:
        raise IngestionError(
            "No AST-chunkable source files found in this repository"
        )
    if len(chunking_result.chunks) > settings.max_repo_chunks:
        raise IngestionError(
            "Repository is too large to index: "
            f"{len(chunking_result.chunks)} code chunks "
            f"(limit {settings.max_repo_chunks})"
        )

    return [chunk.model_dump() for chunk in chunking_result.chunks]


def embed_and_store_stage(
    chunk_dicts: list[dict[str, Any]],
    repo_id: str,
    check_timeout: Callable[[], None],
    report: Reporter,
) -> int:
    """Embed chunks in batches and upsert them into Qdrant (stage 3).

    Delete the repo's existing points first, so a re-ingestion keeps no
    vectors of deleted files; chat is blocked until the repo is "ready"
    again, so the gap is never read. Report progress after every batch and
    retry rate-limit errors with exponential backoff.

    Args:
        chunk_dicts: Serialized chunks from the chunk stage.
        repo_id: Repository id, stored on every point for per-repo filtering.
        check_timeout: Raises IngestionTimeoutError once the budget is spent.
        report: Progress callback, called with status and chunk counts.

    Returns:
        Number of chunks embedded and stored.

    Raises:
        IngestionTimeoutError: If the ingestion time budget is exceeded.
    """
    report("embedding", chunks_total=len(chunk_dicts), chunks_embedded=0)

    model_name = settings.embedding_model
    embedder = EmbeddingService(model_name=model_name)
    backoff = ExponentialBackoff()
    vector_store = QdrantVectorStore(repo_id=repo_id)
    vector_store.delete_repo()

    embedded_total = 0
    for batch in batchify(chunk_dicts, settings.embedding_batch_size):
        check_timeout()

        embedded_chunks = embed_chunks(batch, embedder, backoff, model_name)
        vector_store.upsert(embedded_chunks)
        embedded_total += len(embedded_chunks)

        report(
            "embedding",
            chunks_total=len(chunk_dicts),
            chunks_embedded=embedded_total,
        )

    return embedded_total


def ingest_repo(
    source_url: str, repo_id: str, on_progress: ProgressCallback = None
) -> None:
    """Run clone -> chunk -> embed_and_store end to end for one repo.

    Run the stages in order and own what they share: timeout checks,
    progress reporting and cleanup. The temp clone is always deleted.
    The caller must pass a normalized source_url (normalize_github_url) and
    the repo_id computed from it; neither is re-derived here.

    Args:
        source_url: Normalized repository URL.
        repo_id: Id computed from source_url; stored on every Qdrant point
            for per-repo filtering.
        on_progress: Optional callback receiving {"status": ..., **fields}
            at each stage and after each embedding batch.

    Raises:
        IngestionTimeoutError: If the run exceeds INGESTION_TIMEOUT_SECONDS.
        IngestionError: If the repo has no AST-chunkable source files.
        GitCloneError: If cloning fails.
        RepoTooLargeError: If the repo exceeds the size caps.

    Example:
        >>> url = normalize_github_url("https://github.com/org/repo")
        >>> ingest_repo(url, compute_repo_id(url), on_progress=print)
    """
    timeout_seconds = settings.ingestion_timeout_seconds
    started_at = time.monotonic()
    local_path: Path | None = None

    def report(status: str, **fields: Any) -> None:
        if on_progress:
            on_progress({"status": status, **fields})

    def check_timeout() -> None:
        if time.monotonic() - started_at > timeout_seconds:
            raise IngestionTimeoutError(
                f"Ingestion exceeded {timeout_seconds}s"
            )

    try:
        local_path = clone_stage(source_url, report)
        check_timeout()
        chunk_dicts = chunk_stage(
            local_path, repo_id, source_url, check_timeout, report
        )
        embedded_total = embed_and_store_stage(
            chunk_dicts, repo_id, check_timeout, report
        )

        logger.info(
            "Ingested repo_id=%s: %d chunks embedded", repo_id, embedded_total
        )
    finally:
        if local_path:
            shutil.rmtree(local_path, ignore_errors=True)
