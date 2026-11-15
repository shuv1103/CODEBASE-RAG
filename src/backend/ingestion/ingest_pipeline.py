"""Ingestion pipeline: clone, chunk, embed and store one repo."""

import logging
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, Unpack

from chunking.chunker import ASTCodeChunker
from embeddings.embeddings_client import EmbeddingService, embed_chunks
from ingestion.git_cloner import clone_repo
from ingestion.repo_loader import RepoLoader
from models.chunk_models import CodeChunk
from storage.qdrant_store import QdrantVectorStore
from storage.status_store import RepoStatusValue, StatusFields
from utils.config import settings
from utils.iter_utils import batchify
from utils.rate_limiter import ExponentialBackoff

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[StatusFields], None]


class Reporter(Protocol):
    """Stage-facing progress callback: report(status, **fields)."""

    def __call__(
        self, status: RepoStatusValue, **fields: Unpack[StatusFields]
    ) -> None:
        """Report a stage status plus optional chunk counts."""


class IngestionError(Exception):
    """Raised when a repo cannot be ingested; the message is user-facing."""


class IngestionTimeoutError(IngestionError):
    """Raised when ingestion exceeds INGESTION_TIMEOUT_SECONDS."""


def chunk_stage(
    local_path: Path,
    repo_id: str,
    check_timeout: Callable[[], None],
    report: Reporter,
) -> list[CodeChunk]:
    """Stage 2: load the repo's supported files and chunk them by AST.

    Args:
        local_path: Directory holding the clone.
        repo_id: Stamped on each chunk.
        check_timeout: Raises IngestionTimeoutError once the budget is spent.
        report: Progress callback, called with "chunking" after each batch.

    Returns:
        The chunks, kept as models so no serialized copy of the repo is held.

    Raises:
        IngestionError: If there is nothing to chunk, or more than
            MAX_REPO_CHUNKS chunks (caps embedding spend).
        IngestionTimeoutError: If the time budget is spent.
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

    return chunking_result.chunks


def embed_and_store_stage(
    chunks: list[CodeChunk],
    repo_id: str,
    check_timeout: Callable[[], None],
    report: Reporter,
) -> int:
    """Stage 3: embed chunks in batches and upsert them into Qdrant.

    Points are tagged with a fresh run id; the repo's other points are deleted
    only after the last batch, so a failed run keeps the previous index.

    Args:
        chunks: Chunks from the chunk stage.
        repo_id: Stored on every point for per-repo filtering.
        check_timeout: Raises IngestionTimeoutError once the budget is spent.
        report: Progress callback, called with chunk counts after each batch.

    Returns:
        Number of chunks embedded and stored.

    Raises:
        IngestionTimeoutError: If the time budget is spent.
    """
    report("embedding", chunks_total=len(chunks), chunks_embedded=0)

    embedder = EmbeddingService()
    backoff = ExponentialBackoff()
    vector_store = QdrantVectorStore(repo_id=repo_id)
    run_id = uuid.uuid4().hex

    embedded_total = 0
    for batch in batchify(chunks, settings.embedding_batch_size):
        check_timeout()

        embedded_chunks = embed_chunks(batch, embedder, backoff)
        vector_store.upsert(embedded_chunks, run_id=run_id)
        embedded_total += len(embedded_chunks)

        report(
            "embedding",
            chunks_total=len(chunks),
            chunks_embedded=embedded_total,
        )

    vector_store.delete_stale(run_id)
    return embedded_total


def ingest_repo(
    source_url: str, repo_id: str, on_progress: ProgressCallback | None = None
) -> None:
    """Run clone -> chunk -> embed_and_store for one repo.

    The temp clone is always deleted.

    Args:
        source_url: Normalized repository URL (normalize_github_url).
        repo_id: compute_repo_id(source_url).
        on_progress: Receives StatusFields({"status", ...}) at each stage and
            after each batch.

    Raises:
        IngestionError: If the repo can't be indexed, including timeouts.
        GitCloneError: If cloning fails.
        RepoTooLargeError: If a size limit is exceeded.

    Example:
        >>> url = normalize_github_url("https://github.com/org/repo")
        >>> ingest_repo(url, compute_repo_id(url), on_progress=print)
    """
    timeout_seconds = settings.ingestion_timeout_seconds
    started_at = time.monotonic()
    local_path: Path | None = None

    def report(
        status: RepoStatusValue, **fields: Unpack[StatusFields]
    ) -> None:
        if on_progress:
            on_progress(StatusFields(status=status, **fields))

    def check_timeout() -> None:
        if time.monotonic() - started_at > timeout_seconds:
            raise IngestionTimeoutError(
                f"Ingestion exceeded {timeout_seconds}s"
            )

    try:
        report("cloning")
        local_path = clone_repo(source_url)
        check_timeout()
        chunks = chunk_stage(local_path, repo_id, check_timeout, report)
        embedded_total = embed_and_store_stage(
            chunks, repo_id, check_timeout, report
        )

        logger.info(
            "Ingested repo_id=%s: %d chunks embedded", repo_id, embedded_total
        )
    finally:
        if local_path:
            shutil.rmtree(local_path, ignore_errors=True)
