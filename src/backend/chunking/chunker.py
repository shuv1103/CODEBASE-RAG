"""AST-based code chunker that runs chonkie in a subprocess pool."""

import functools
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from billiard.exceptions import WorkerLostError
from billiard.pool import ApplyResult, Pool

from ingestion.repo_loader import FileDocument, IngestionResult
from utils.iter_utils import batchify

from .chunk_config import (
    AST_LANGUAGE_ALIASES,
    AST_SUPPORTED_LANGUAGES,
    DEFAULT_CHUNK_BATCH_SIZE,
    DEFAULT_CODE_CHUNK_SIZE,
    DEFAULT_CODE_CHUNK_TOKENIZER,
)
from .chunk_models import (
    ChunkingResult,
    ChunkingStats,
    ChunkMetadata,
    CodeChunk,
)

if TYPE_CHECKING:
    from chonkie import CodeChunker

logger = logging.getLogger(__name__)

PER_FILE_TIMEOUT_SECONDS = 30
_POOL_POLL_INTERVAL_SECONDS = 0.05

# One chunk as pickled back from the worker subprocess:
# {"text", "start_byte", "end_byte", "token_count"}.
RawChunk = dict[str, Any]


@dataclass
class ChunkRun:
    """State shared by every batch of one ASTCodeChunker.chunk() call.

    Attributes:
        repo_id: Repository id to stamp on each chunk.
        source_url: Repository URL to stamp on each chunk.
        stats: Run stats, updated in place.
        pool: Single-worker pool; replaced when its worker crashes or hangs.
        chunks: Every chunk produced so far.
    """

    repo_id: str
    source_url: str
    stats: ChunkingStats
    pool: Pool
    chunks: list[CodeChunk] = field(default_factory=list)


@functools.cache
def get_chunker(
    language: str, chunk_size: int, tokenizer: str
) -> "CodeChunker":
    """Return a cached chonkie.CodeChunker, creating it on first use.

    Args:
        language: Normalized language name.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name (e.g. "character").

    Returns:
        The CodeChunker cached for (language, chunk_size, tokenizer).
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
    """Run chonkie's CodeChunker.chunk_batch() in a subprocess.

    Return plain dicts, which pickle back to the parent process.

    Args:
        contents: Source text of each file in the batch.
        normalized_language: Normalized language shared by the batch.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name.

    Returns:
        One list per input file of {"text", "start_byte", "end_byte",
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
    """Chunk a single file, expressed as a batch of one.

    Module-level so it can be pickled and sent to the subprocess.

    Args:
        content: Source text of the file.
        normalized_language: Normalized language of the file.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name.

    Returns:
        List of {"text", "start_byte", "end_byte", "token_count"} dicts.
    """
    return chunkify_batch_in_subprocess(
        [content], normalized_language, chunk_size, tokenizer
    )[0]


def get_result_fast(
    pool: Pool, async_result: ApplyResult, hard_timeout: float
) -> Any:
    """Wait for a pool result, failing fast if the worker dies or hangs.

    Poll rather than block on get(), so a native crash in the worker is
    caught at once instead of after the full timeout.

    Args:
        pool: Single-worker pool running the job.
        async_result: Pending result from pool.apply_async.
        hard_timeout: Maximum seconds to wait.

    Returns:
        The job's return value.

    Raises:
        WorkerLostError: If the pool's worker process has died.
        TimeoutError: If hard_timeout elapses first.
    """
    start = time.monotonic()
    while True:
        if async_result.ready():
            return async_result.get()
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
    """AST-based chunker for the languages in AST_SUPPORTED_LANGUAGES.

    Files are chunked in same-language batches inside a single-worker
    subprocess pool, so a native tree-sitter crash kills only the worker,
    not the caller.

    Args:
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name.
        batch_size: Same-language files sent per subprocess call.

    Attributes:
        chunk_size: Configured maximum chunk size.
        tokenizer: Configured tokenizer name.
        batch_size: Configured batch size.

    Example:
        >>> chunker = ASTCodeChunker()
        >>> result = chunker.chunk(RepoLoader("path/to/repo").load())
        >>> print(result.stats.total_chunks_created)
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

    def normalize_language(self, document: FileDocument) -> str | None:
        """Map a document's language to its chonkie language name.

        Args:
            document: The loaded file.

        Returns:
            The normalized language, or None if it has no AST alias.
        """
        language = (document.language or "").strip().lower()
        return AST_LANGUAGE_ALIASES.get(language)

    def update_language_distribution(
        self, stats: ChunkingStats, language: str
    ) -> None:
        """Increment the per-language document count.

        Args:
            stats: Stats object to update in place.
            language: Detected language of a document.
        """
        stats.language_distribution[language] = (
            stats.language_distribution.get(language, 0) + 1
        )

    def new_pool(self) -> Pool:
        """Create a single-worker billiard.Pool for chunking one repo.

        Returns:
            A new Pool with one worker process.
        """
        return Pool(processes=1)

    def build_code_chunks(
        self,
        document: FileDocument,
        raw_chunks: list[RawChunk],
        repo_id: str,
        source_url: str,
    ) -> list[CodeChunk]:
        """Build CodeChunk objects from raw chunk data.

        Derive 1-based, inclusive line numbers from chonkie's byte offsets
        into the UTF-8 content.

        Args:
            document: The file the chunks came from.
            raw_chunks: Dicts produced by chunkify_batch_in_subprocess.
            repo_id: Repository id to stamp on each chunk.
            source_url: Repository URL to stamp on each chunk.

        Returns:
            One CodeChunk per raw chunk, in order.
        """
        content_bytes = document.content.encode("utf-8")
        total = len(raw_chunks)

        return [
            CodeChunk(
                content=raw["text"],
                metadata=ChunkMetadata(
                    chunk_id=f"{document.relative_path}::chunk::{idx}",
                    file_name=document.file_name,
                    absolute_path=document.absolute_path,
                    relative_path=document.relative_path,
                    extension=document.extension,
                    language=document.language,
                    size_bytes=document.size_bytes,
                    chunk_index=idx,
                    total_chunks_in_file=total,
                    chunking_strategy="chonkie_ast",
                    symbol_name=None,
                    start_line=content_bytes[: raw["start_byte"]].count(b"\n")
                    + 1,
                    # The chunk's last byte, so a trailing newline doesn't
                    # count as the start of another line.
                    end_line=content_bytes[
                        : max(raw["end_byte"] - 1, raw["start_byte"])
                    ].count(b"\n")
                    + 1,
                    repo_id=repo_id,
                    source_url=source_url,
                    extra={"token_count": raw["token_count"]},
                ),
            )
            for idx, raw in enumerate(raw_chunks)
        ]

    def replace_pool(self, run: ChunkRun) -> None:
        """Swap the run's pool for a fresh one after its worker crashed or hung.

        Args:
            run: The current chunking run.
        """
        run.pool.terminate()
        run.pool = self.new_pool()

    def chunk_single_document(
        self, document: FileDocument, normalized_language: str, run: ChunkRun
    ) -> list[CodeChunk]:
        """Chunk one document in the run pool's worker subprocess.

        Args:
            document: The file to chunk.
            normalized_language: Normalized language of the file.
            run: The current chunking run.

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
        return self.build_code_chunks(
            document, raw_chunks, run.repo_id, run.source_url
        )

    def record_file_result(
        self,
        document: FileDocument,
        file_chunks: list[CodeChunk],
        run: ChunkRun,
    ) -> None:
        """Add a file's chunks to the run's totals, if it produced any.

        Args:
            document: The file that was chunked.
            file_chunks: Chunks produced for the file.
            run: The current chunking run.
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
        """Chunk one file, recovering from crashes, hangs, and errors.

        Count failures in run.stats.failed_files instead of raising them.

        Args:
            document: The file to chunk.
            normalized_language: Normalized language of the file.
            run: The current chunking run; its pool is replaced if the
                worker crashed or hung.
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
        # chonkie and tree-sitter can raise anything on a malformed file;
        # one bad file must not fail the repo.
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
        """Chunk each file of a failed batch on its own, isolating bad files.

        Args:
            batch: Same-language files whose batch call failed.
            normalized_language: Normalized language shared by the batch.
            run: The current chunking run.
            on_batch: Called after each file, so a slow fallback still
                reports progress and honours the timeout.
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
        """Chunk a same-language batch in one subprocess call.

        If the batch fails, retry each file on its own to isolate the bad
        ones, replacing the pool first if the worker crashed or hung.

        Args:
            batch: Same-language files to chunk.
            normalized_language: Normalized language shared by the batch.
            run: The current chunking run.
            on_batch: Called after each per-file retry, so a slow fallback
                still reports progress and honours the timeout.
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
            # Build all chunks before recording any, so a mid-batch failure
            # can't record files twice when they are retried one by one.
            batch_chunks = [
                self.build_code_chunks(
                    document, raw_chunks, run.repo_id, run.source_url
                )
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
        # Any other failure leaves the worker usable; the per-file retries
        # find (and count) the file that caused it.
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
        source_url: str = "",
        on_batch: Callable[[], None] | None = None,
    ) -> ChunkingResult:
        """Chunk every supported document in an ingestion result by its AST.

        Skip and count documents in unsupported languages. Files that crash
        or hang the parser are counted in stats.failed_files.

        Args:
            ingestion_result: Documents loaded by RepoLoader.
            repo_id: Repository id to stamp on each chunk.
            source_url: Repository URL to stamp on each chunk.
            on_batch: Called after every batch, e.g. to report progress or
                stop on a timeout by raising.

        Returns:
            ChunkingResult with all chunks and run stats.
        """
        stats = ChunkingStats(
            total_documents_received=len(ingestion_result.documents)
        )

        documents_by_language: defaultdict[str, list[FileDocument]] = (
            defaultdict(list)
        )
        for document in ingestion_result.documents:
            self.update_language_distribution(stats, document.language)

            normalized_language = self.normalize_language(document)

            if (
                not normalized_language
                or normalized_language not in AST_SUPPORTED_LANGUAGES
            ):
                stats.skipped_unsupported_language += 1
                continue

            stats.total_ast_eligible_documents += 1
            documents_by_language[normalized_language].append(document)

        run = ChunkRun(
            repo_id=repo_id,
            source_url=source_url,
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
    """Warn that a batch failed and is about to be retried file by file.

    Args:
        batch: The files in the failed batch.
        normalized_language: Normalized language shared by the batch.
        exc: The error the batch raised.
    """
    logger.warning(
        "Batch of %d %s file(s) failed (%s) — retrying individually "
        "to isolate the failing file(s).",
        len(batch),
        normalized_language,
        exc,
    )
