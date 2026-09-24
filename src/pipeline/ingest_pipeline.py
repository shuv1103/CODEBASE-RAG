import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from chunking.chunker import ASTCodeChunker
from codebase_rag_agent.rate_limiter import ExponentialBackoff
from embeddings.qdrant_store import QdrantVectorStore
from embeddings.embeddings_client import EmbeddingService
from embeddings.embeddings_formatter import build_embedded_chunk, build_embedding_input_text
from ingestion.git_cloner import clone_repo
from ingestion.repo_loader import RepoLoader

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[dict], None]]


class IngestionError(Exception):
    """Raised when a repo cannot be ingested (empty, unsupported, or a pipeline step failed)."""


class IngestionTimeoutError(IngestionError):
    """Raised when ingestion exceeds INGESTION_TIMEOUT_SECONDS."""


def _batchify(items: list, batch_size: int):
    """Yield consecutive slices of at most batch_size items.

    Args:
        items: Items to split.
        batch_size: Maximum items per batch.

    Yields:
        Lists of up to batch_size items, in order.
    """
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _clone_stage(source_url: str, check_timeout: Callable[[], None], report: Callable) -> Path:
    """Pipeline stage 1: shallow-clone the repo into a temp dir.

    Args:
        source_url: Normalized repository URL.
        check_timeout: Raises IngestionTimeoutError once the budget is spent.
        report: Progress callback, called with the stage status.

    Returns:
        Path of the temp clone directory.

    Raises:
        GitCloneError: If the clone fails or times out.
        RepoTooLargeError: If the repo exceeds the size caps.
        IngestionTimeoutError: If the ingestion time budget is exceeded.
    """
    report("cloning")
    local_path = clone_repo(source_url)
    check_timeout()
    return local_path


def _chunk_stage(
    local_path: Path, repo_id: str, source_url: str, check_timeout: Callable[[], None], report: Callable
) -> list[dict]:
    """Pipeline stage 2: load repo files and AST-chunk them.

    Args:
        local_path: Directory holding the cloned repo.
        repo_id: Repository id to stamp on each chunk.
        source_url: Repository URL to stamp on each chunk.
        check_timeout: Raises IngestionTimeoutError once the budget is spent.
        report: Progress callback, called with the stage status.

    Returns:
        Chunks serialized as dicts (CodeChunk.model_dump()).

    Raises:
        IngestionError: If no AST-chunkable source files were found.
        IngestionTimeoutError: If the ingestion time budget is exceeded.
    """
    report("chunking")
    ingestion_result = RepoLoader(str(local_path)).load()
    chunking_result = ASTCodeChunker().chunk(ingestion_result, repo_id=repo_id, source_url=source_url)
    check_timeout()

    if not chunking_result.chunks:
        raise IngestionError("No AST-chunkable source files found in this repository")

    return [chunk.model_dump() for chunk in chunking_result.chunks]


def _embed_and_store_stage(
    chunk_dicts: list[dict], repo_id: str, check_timeout: Callable[[], None], report: Callable
) -> int:
    """Pipeline stage 3: embed chunks in batches and upsert them into Qdrant.

    Writes to the repo's own collection and reports chunks_embedded progress
    after every batch. Rate-limit errors are retried with exponential
    backoff.

    Args:
        chunk_dicts: Serialized chunks from the chunk stage.
        repo_id: Repository id, stored on every point for per-repo filtering.
        check_timeout: Raises IngestionTimeoutError once the budget is spent.
        report: Progress callback, called with status and chunk counts.

    Returns:
        Number of chunks embedded and stored.

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
        IngestionTimeoutError: If the ingestion time budget is exceeded.
    """
    report("embedding", chunks_total=len(chunk_dicts), chunks_embedded=0)

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))

    embedder = EmbeddingService(api_key, model_name)
    backoff = ExponentialBackoff(max_retries=int(os.getenv("BACKOFF_MAX_RETRIES", "5")))
    vector_store = QdrantVectorStore(repo_id=repo_id)

    embedded_total = 0
    for batch in _batchify(chunk_dicts, batch_size):
        check_timeout()

        texts = [build_embedding_input_text(chunk) for chunk in batch]
        vectors = backoff.execute(lambda texts=texts: embedder.embed_documents(texts))

        embedded_chunks = [
            build_embedded_chunk(raw_chunk=chunk, vector=vector, model_name=model_name)
            for chunk, vector in zip(batch, vectors)
        ]
        vector_store.upsert(embedded_chunks)
        embedded_total += len(embedded_chunks)

        report("embedding", chunks_total=len(chunk_dicts), chunks_embedded=embedded_total)

    return embedded_total


def ingest_repo(source_url: str, repo_id: str, on_progress: ProgressCallback = None) -> None:
    """Run clone -> chunk -> embed_and_store end to end for one repo.

    Pipeline Pattern: each stage above is a discrete, independently readable function;
    this orchestrator only wires them together in sequence and owns the
    cross-cutting concerns every stage shares (timeout checks, progress
    reporting, and guaranteed cleanup).

    `source_url` must already be normalized (see utils.repo_id.normalize_github_url)
    and `repo_id` already computed from it — this function does not re-derive
    either, so the caller is the single source of truth for both.

    Always cleans up the temp clone directory, on success or failure.

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
    timeout_seconds = int(os.getenv("INGESTION_TIMEOUT_SECONDS", "900"))
    started_at = time.monotonic()
    local_path = None

    def _report(status: str, **fields) -> None:
        if on_progress:
            on_progress({"status": status, **fields})

    def _check_timeout() -> None:
        if time.monotonic() - started_at > timeout_seconds:
            raise IngestionTimeoutError(f"Ingestion exceeded {timeout_seconds}s")

    try:
        local_path = _clone_stage(source_url, _check_timeout, _report)
        chunk_dicts = _chunk_stage(local_path, repo_id, source_url, _check_timeout, _report)
        embedded_total = _embed_and_store_stage(chunk_dicts, repo_id, _check_timeout, _report)

        logger.info("Ingested repo_id=%s: %d chunks embedded", repo_id, embedded_total)
    finally:
        if local_path:
            shutil.rmtree(local_path, ignore_errors=True)
