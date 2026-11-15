"""AST code chunking with chonkie, run in a crash-isolated subprocess."""

import bisect
import functools
import logging
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from billiard.exceptions import WorkerLostError
from billiard.pool import ApplyResult, Pool

from chunking.chunk_config import (
    DEFAULT_CHUNK_BATCH_SIZE,
    DEFAULT_CODE_CHUNK_SIZE,
    DEFAULT_CODE_CHUNK_TOKENIZER,
)
from models.chunk_models import (
    ChunkingResult,
    ChunkingStats,
    ChunkMetadata,
    CodeChunk,
)
from models.ingestion_models import FileDocument, IngestionResult
from utils.iter_utils import batchify

if TYPE_CHECKING:
    from chonkie import CodeChunker

logger = logging.getLogger(__name__)

PER_FILE_TIMEOUT_SECONDS = 30
_POOL_POLL_INTERVAL_SECONDS = 0.05

# A chunk as pickled back from the worker: text, byte offsets, tokens.
RawChunk = dict[str, Any]


@dataclass
class ChunkRun:
    """State shared by every batch of one ASTCodeChunker.chunk() call.

    Attributes:
        repo_id: Stamped on each chunk.
        stats: Run counters, updated in place.
        pool: Single-worker pool; replaced when its worker crashes or hangs.
        chunks: Every chunk produced so far.
    """

    repo_id: str
    stats: ChunkingStats
    pool: Pool
    chunks: list[CodeChunk] = field(default_factory=list)


@functools.cache
def get_chunker(
    language: str, chunk_size: int, tokenizer: str
) -> "CodeChunker":
    """Return a cached chonkie CodeChunker, built on first use.

    Args:
        language: chonkie language name, e.g. "python".
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name, e.g. "character".

    Returns:
        The CodeChunker for these settings.
    """
    from chonkie import CodeChunker

    return CodeChunker(
        language=language, tokenizer=tokenizer, chunk_size=chunk_size
    )


def chunkify_batch_in_subprocess(
    contents: list[str],
    normalized_language: str,
    chunk_size: int,
    tokenizer: str,
) -> list[list[RawChunk]]:
    """Chunk same-language files with chonkie; runs in the worker process.

    Args:
        contents: Source text of each file.
        normalized_language: chonkie language shared by the batch.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name.

    Returns:
        Per file, a list of picklable {"text", "start_byte", "end_byte",
        "token_count"} dicts.
    """
    chunker = get_chunker(normalized_language, chunk_size, tokenizer)
    batches = chunker.chunk_batch(contents, show_progress=False)
    return [
        [
            {
                "text": chunk.text,
                "start_byte": chunk.start_index,
                "end_byte": chunk.end_index,
                "token_count": chunk.token_count,
            }
            for chunk in doc_chunks
        ]
        for doc_chunks in batches
    ]


def chunkify_in_subprocess(
    content: str,
    normalized_language: str,
    chunk_size: int,
    tokenizer: str,
) -> list[RawChunk]:
    """Chunk one file in the worker process (a batch of one).

    Args:
        content: Source text of the file.
        normalized_language: chonkie language of the file.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name.

    Returns:
        The file's {"text", "start_byte", "end_byte", "token_count"} dicts.
    """
    return chunkify_batch_in_subprocess(
        [content], normalized_language, chunk_size, tokenizer
    )[0]


def get_result_fast(
    pool: Pool, async_result: ApplyResult, hard_timeout: float
) -> Any:
    """Wait for a pool result, failing fast if the worker dies or hangs.

    Args:
        pool: Single-worker pool running the job.
        async_result: Pending result from pool.apply_async.
        hard_timeout: Maximum seconds to wait.

    Returns:
        The job's return value.

    Raises:
        WorkerLostError: If the worker process died.
        TimeoutError: If hard_timeout elapsed first.
    """
    start = time.monotonic()
    while True:
        if async_result.ready():
            return async_result.get()
        # billiard (pinned 4.3.0) has no public worker handle; a test fails
        # if _pool disappears in an upgrade.
        worker = pool._pool[0] if pool._pool else None
        if worker is not None and not worker.is_alive():
            raise WorkerLostError(
                f"chunking worker died (exitcode={worker.exitcode})"
            )
        if time.monotonic() - start > hard_timeout:
            raise TimeoutError(
                f"chunking a single file exceeded {hard_timeout}s"
            )
        async_result.wait(_POOL_POLL_INTERVAL_SECONDS)


class ASTCodeChunker:
    """Chunks source files by AST in a single-worker subprocess pool.

    A native tree-sitter crash kills only the worker: the file is skipped
    and counted in stats.failed_files.

    Args:
        chunk_size: Maximum chunk size, in tokenizer units (default 2300).
        tokenizer: chonkie tokenizer name (default "character").
        batch_size: Same-language files per subprocess call (default 20).

    Example:
        >>> result = ASTCodeChunker().chunk(RepoLoader("repo").load())
        >>> result.stats.total_chunks_created
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CODE_CHUNK_SIZE,
        tokenizer: str = DEFAULT_CODE_CHUNK_TOKENIZER,
        batch_size: int = DEFAULT_CHUNK_BATCH_SIZE,
    ) -> None:
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def new_pool(self) -> Pool:
        """Return a new single-worker billiard Pool."""
        return Pool(processes=1)

    def build_code_chunks(
        self,
        document: FileDocument,
        raw_chunks: list[RawChunk],
        repo_id: str,
    ) -> list[CodeChunk]:
        """Build CodeChunks from a file's raw chonkie chunks.

        Line numbers are 1-based and inclusive, derived from the byte offsets.

        Args:
            document: The file the chunks came from.
            raw_chunks: Its {"text", "start_byte", "end_byte", "token_count"}
                dicts.
            repo_id: Stamped on each chunk.

        Returns:
            One CodeChunk per raw chunk, in order.
        """
        content_bytes = document.content.encode("utf-8")
        # Newlines before an offset = its line - 1; found once, not per chunk.
        newlines = [m.start() for m in re.finditer(b"\n", content_bytes)]

        return [
            CodeChunk(
                content=raw["text"],
                metadata=ChunkMetadata(
                    chunk_id=f"{document.relative_path}::chunk::{idx}",
                    file_name=document.file_name,
                    relative_path=document.relative_path,
                    language=document.language,
                    chunking_strategy="chonkie_ast",
                    start_line=bisect.bisect_left(newlines, raw["start_byte"])
                    + 1,
                    # Last byte, so a trailing newline isn't a new line.
                    end_line=bisect.bisect_left(
                        newlines, max(raw["end_byte"] - 1, raw["start_byte"])
                    )
                    + 1,
                    repo_id=repo_id,
                    extra={"token_count": raw["token_count"]},
                ),
            )
            for idx, raw in enumerate(raw_chunks)
        ]

    def replace_pool(self, run: ChunkRun) -> None:
        """Terminate the run's pool and start a fresh one.

        Args:
            run: The current run; its pool's worker crashed or hung.
        """
        run.pool.terminate()
        run.pool = self.new_pool()

    def chunk_single_document(
        self, document: FileDocument, normalized_language: str, run: ChunkRun
    ) -> list[CodeChunk]:
        """Chunk one file in the run pool's worker.

        Args:
            document: The file to chunk.
            normalized_language: chonkie language of the file.
            run: The current run.

        Returns:
            The file's chunks.

        Raises:
            WorkerLostError: If the worker process died.
            TimeoutError: If chunking exceeded PER_FILE_TIMEOUT_SECONDS.
        """
        async_result = run.pool.apply_async(
            chunkify_in_subprocess,
            args=(
                document.content,
                normalized_language,
                self.chunk_size,
                self.tokenizer,
            ),
        )
        raw_chunks = get_result_fast(
            run.pool, async_result, PER_FILE_TIMEOUT_SECONDS
        )
        return self.build_code_chunks(document, raw_chunks, run.repo_id)

    def record_file_result(
        self,
        document: FileDocument,
        file_chunks: list[CodeChunk],
        run: ChunkRun,
    ) -> None:
        """Add a file's chunks to the run totals, if it produced any.

        Args:
            document: The chunked file.
            file_chunks: Its chunks.
            run: The current run.
        """
        if not file_chunks:
            return
        run.stats.total_documents_chunked += 1
        run.stats.total_chunks_created += len(file_chunks)
        run.chunks.extend(file_chunks)
        logger.debug(
            "Chunked %s: %d chunks (total so far: %d)",
            document.relative_path,
            len(file_chunks),
            run.stats.total_chunks_created,
        )

    def chunk_one_with_recovery(
        self, document: FileDocument, normalized_language: str, run: ChunkRun
    ) -> None:
        """Chunk one file, counting a crash, hang or error instead of raising.

        Args:
            document: The file to chunk.
            normalized_language: chonkie language of the file.
            run: The current run; its pool is replaced after a crash or hang.
        """
        try:
            file_chunks = self.chunk_single_document(
                document, normalized_language, run
            )
        except (WorkerLostError, TimeoutError) as exc:
            run.stats.failed_files += 1
            logger.warning(
                "Chunking crashed/hung on %s: %s", document.relative_path, exc
            )
            self.replace_pool(run)
            return
        # chonkie can raise anything; one bad file must not fail the repo.
        except Exception as exc:  # noqa: BLE001
            run.stats.failed_files += 1
            logger.warning(
                "Failed chunking %s: %s", document.relative_path, exc
            )
            return
        self.record_file_result(document, file_chunks, run)

    def retry_batch_per_file(
        self,
        batch: list[FileDocument],
        normalized_language: str,
        run: ChunkRun,
        on_batch: Callable[[], None] | None,
    ) -> None:
        """Chunk a failed batch file by file, isolating the bad files.

        Args:
            batch: Same-language files whose batch call failed.
            normalized_language: chonkie language of the batch.
            run: The current run.
            on_batch: Called after each file (progress, timeout checks).
        """
        for document in batch:
            self.chunk_one_with_recovery(document, normalized_language, run)
            if on_batch is not None:
                on_batch()

    def chunk_batch_with_fallback(
        self,
        batch: list[FileDocument],
        normalized_language: str,
        run: ChunkRun,
        on_batch: Callable[[], None] | None = None,
    ) -> None:
        """Chunk a same-language batch in one call; on failure, retry per file.

        Args:
            batch: Same-language files to chunk.
            normalized_language: chonkie language of the batch.
            run: The current run.
            on_batch: Called after each per-file retry.
        """
        try:
            async_result = run.pool.apply_async(
                chunkify_batch_in_subprocess,
                args=(
                    [d.content for d in batch],
                    normalized_language,
                    self.chunk_size,
                    self.tokenizer,
                ),
            )
            batched_raw = get_result_fast(
                run.pool, async_result, PER_FILE_TIMEOUT_SECONDS * len(batch)
            )
            # Build all first, so a failure can't record files twice.
            batch_chunks = [
                self.build_code_chunks(document, raw_chunks, run.repo_id)
                for document, raw_chunks in zip(
                    batch, batched_raw, strict=True
                )
            ]
        except (WorkerLostError, TimeoutError) as exc:
            log_batch_failure(batch, normalized_language, exc)
            # A dead or stuck worker must be replaced before the retries.
            self.replace_pool(run)
            self.retry_batch_per_file(
                batch, normalized_language, run, on_batch
            )
            return
        # The worker is still usable; per-file retries find the bad file.
        except Exception as exc:  # noqa: BLE001
            log_batch_failure(batch, normalized_language, exc)
            self.retry_batch_per_file(
                batch, normalized_language, run, on_batch
            )
            return

        for document, file_chunks in zip(batch, batch_chunks, strict=True):
            self.record_file_result(document, file_chunks, run)

    def chunk(
        self,
        ingestion_result: IngestionResult,
        repo_id: str = "",
        on_batch: Callable[[], None] | None = None,
    ) -> ChunkingResult:
        """Chunk every document, batched by language.

        Crashing files are counted in stats.failed_files.

        Args:
            ingestion_result: Documents loaded by RepoLoader.
            repo_id: Stamped on each chunk.
            on_batch: Called after every batch; may raise to stop (timeout).

        Returns:
            ChunkingResult with every chunk and the run stats.
        """
        stats = ChunkingStats()

        documents_by_language: defaultdict[str, list[FileDocument]] = (
            defaultdict(list)
        )
        # RepoLoader only loads EXTENSION_LANGUAGE_MAP languages, whose names
        # are chonkie's too.
        for document in ingestion_result.documents:
            documents_by_language[document.language].append(document)

        run = ChunkRun(
            repo_id=repo_id,
            stats=stats,
            pool=self.new_pool(),
        )
        try:
            for (
                normalized_language,
                documents,
            ) in documents_by_language.items():
                for batch in batchify(documents, self.batch_size):
                    self.chunk_batch_with_fallback(
                        batch, normalized_language, run, on_batch
                    )
                    if on_batch is not None:
                        on_batch()
        finally:
            run.pool.terminate()

        return ChunkingResult(chunks=run.chunks, stats=stats)


def log_batch_failure(
    batch: list[FileDocument], normalized_language: str, exc: Exception
) -> None:
    """Warn that a batch failed and will be retried file by file.

    Args:
        batch: Files in the failed batch.
        normalized_language: chonkie language of the batch.
        exc: The error the batch raised.
    """
    logger.warning(
        "Batch of %d %s file(s) failed (%s) — retrying individually "
        "to isolate the failing file(s).",
        len(batch),
        normalized_language,
        exc,
    )
