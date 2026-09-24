import logging
import time
from typing import List, Optional

from billiard.exceptions import WorkerLostError
from billiard.pool import Pool

from ingestion.repo_loader import FileDocument, IngestionResult

from .chunk_config import (
    AST_SUPPORTED_LANGUAGES,
    AST_LANGUAGE_ALIASES,
    DEFAULT_CODE_CHUNK_SIZE,
    DEFAULT_CODE_CHUNK_TOKENIZER,
    DEFAULT_CHUNK_BATCH_SIZE,
)
from .chunk_models import CodeChunk, ChunkMetadata, ChunkingResult, ChunkingStats

logger = logging.getLogger(__name__)

PER_FILE_TIMEOUT_SECONDS = 30
_POOL_POLL_INTERVAL_SECONDS = 0.05
_CHUNKER_CACHE: dict = {}


def _get_chunker(language: str, chunk_size: int, tokenizer: str):
    """Return a cached chonkie.CodeChunker, creating it on first use.

    Args:
        language: Normalized language name.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name (e.g. "character").

    Returns:
        The CodeChunker cached for (language, chunk_size, tokenizer).
    """
    from chonkie import CodeChunker

    key = (language, chunk_size, tokenizer)
    chunker = _CHUNKER_CACHE.get(key)
    if chunker is None:
        chunker = CodeChunker(language=language, tokenizer=tokenizer, chunk_size=chunk_size)
        _CHUNKER_CACHE[key] = chunker
    return chunker


def _chunkify_batch_in_subprocess(
    contents: list,
    normalized_language: str,
    chunk_size: int,
    tokenizer: str,
) -> list:
    """Run chonkie's CodeChunker.chunk_batch() in a subprocess.

    Returns plain dicts so the result can be pickled back to the parent.

    Args:
        contents: Source text of each file in the batch.
        normalized_language: Normalized language shared by the batch.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name.

    Returns:
        One list per input file of {"text", "start_byte", "end_byte",
        "token_count"} dicts.
    """
    chunker = _get_chunker(normalized_language, chunk_size, tokenizer)
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


def _chunkify_in_subprocess(
    content: str,
    normalized_language: str,
    chunk_size: int,
    tokenizer: str,
) -> list:
    """Chunk a single file, expressed as a batch of one.

    Kept as a module-level function so it can be pickled and shipped to a
    subprocess.

    Args:
        content: Source text of the file.
        normalized_language: Normalized language of the file.
        chunk_size: Maximum chunk size, in tokenizer units.
        tokenizer: chonkie tokenizer name.

    Returns:
        List of {"text", "start_byte", "end_byte", "token_count"} dicts.
    """
    return _chunkify_batch_in_subprocess([content], normalized_language, chunk_size, tokenizer)[0]


def _get_result_fast(pool: Pool, async_result, hard_timeout: float):
    """Wait for a pool result, failing fast if the worker dies or hangs.

    Polls instead of blocking on get(), so a native crash in the worker is
    detected immediately rather than waiting for the full timeout.

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
            raise WorkerLostError(f"chunking worker died (exitcode={worker.exitcode})")
        if time.monotonic() - start > hard_timeout:
            raise TimeoutError(f"chunking a single file exceeded {hard_timeout}s")
        async_result.wait(_POOL_POLL_INTERVAL_SECONDS)


class ASTCodeChunker:
    """AST-based chunker for chonkie/tree-sitter-language-pack languages.

    See chunk_config.AST_SUPPORTED_LANGUAGES / AST_LANGUAGE_ALIASES for the
    authoritative set:
    - python
    - java
    - csharp
    - typescript
    - javascript
    - rust
    - go
    - c
    - cpp
    - html
    - css
    - markdown

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
    ):
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def _normalize_language(self, document: FileDocument) -> Optional[str]:
        """Map a document's language to its chonkie language name.

        Args:
            document: The loaded file.

        Returns:
            The normalized language, or None if it has no AST alias.
        """
        language = (document.language or "").strip().lower()
        return AST_LANGUAGE_ALIASES.get(language)

    def _update_language_distribution(self, stats: ChunkingStats, language: str) -> None:
        """Increment the per-language document count.

        Args:
            stats: Stats object to update in place.
            language: Detected language of a document.
        """
        stats.language_distribution[language] = stats.language_distribution.get(language, 0) + 1

    def _new_pool(self) -> Pool:
        """Create a single-worker billiard.Pool for chunking one repo.

        Returns:
            A new Pool with one worker process.
        """
        return Pool(processes=1)

    def _build_code_chunks(
        self,
        document: FileDocument,
        raw_chunks: list,
        repo_id: str,
        source_url: str,
    ) -> List[CodeChunk]:
        """Build CodeChunk objects from raw chunk data.

        Line numbers are derived from byte offsets into the UTF-8 content.

        Args:
            document: The file the chunks came from.
            raw_chunks: Dicts produced by _chunkify_batch_in_subprocess.
            repo_id: Repository id to stamp on each chunk.
            source_url: Repository URL to stamp on each chunk.

        Returns:
            One CodeChunk per raw chunk, in order.
        """
        content_bytes = document.content.encode("utf-8")
        total = len(raw_chunks)
        result: List[CodeChunk] = []

        for idx, raw in enumerate(raw_chunks):
            start_byte = raw["start_byte"]
            end_byte = raw["end_byte"]

            metadata = ChunkMetadata(
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
                start_line=content_bytes[:start_byte].count(b"\n"),
                end_line=content_bytes[:end_byte].count(b"\n"),
                repo_id=repo_id,
                source_url=source_url,
                extra={"token_count": raw["token_count"]},
            )

            result.append(CodeChunk(content=raw["text"], metadata=metadata))

        return result

    def _chunk_single_document(
        self,
        document: FileDocument,
        normalized_language: str,
        repo_id: str,
        source_url: str,
        pool: Pool,
    ) -> List[CodeChunk]:
        """Chunk one document in the pool's worker subprocess.

        Args:
            document: The file to chunk.
            normalized_language: Normalized language of the file.
            repo_id: Repository id to stamp on each chunk.
            source_url: Repository URL to stamp on each chunk.
            pool: Single-worker pool to run the job in.

        Returns:
            The file's chunks.

        Raises:
            WorkerLostError: If the worker process died.
            TimeoutError: If chunking exceeded PER_FILE_TIMEOUT_SECONDS.
        """
        async_result = pool.apply_async(
            _chunkify_in_subprocess,
            args=(document.content, normalized_language, self.chunk_size, self.tokenizer),
        )
        raw_chunks = _get_result_fast(pool, async_result, PER_FILE_TIMEOUT_SECONDS)
        return self._build_code_chunks(document, raw_chunks, repo_id, source_url)

    def _record_file_result(
        self,
        document: FileDocument,
        file_chunks: List[CodeChunk],
        stats: ChunkingStats,
        all_chunks: List[CodeChunk],
    ) -> None:
        """Add a file's chunks to the running totals, if it produced any.

        Args:
            document: The file that was chunked.
            file_chunks: Chunks produced for the file.
            stats: Stats object to update in place.
            all_chunks: Accumulator list, extended in place.
        """
        if file_chunks:
            stats.total_documents_chunked += 1
            stats.total_chunks_created += len(file_chunks)
            all_chunks.extend(file_chunks)
            logger.debug(
                "Chunked %s: %d chunks (total so far: %d)",
                document.relative_path,
                len(file_chunks),
                stats.total_chunks_created,
            )

    def _chunk_one_with_recovery(
        self,
        document: FileDocument,
        normalized_language: str,
        repo_id: str,
        source_url: str,
        pool: Pool,
        stats: ChunkingStats,
        all_chunks: List[CodeChunk],
    ) -> Pool:
        """Chunk one file, recovering from crashes, hangs, and errors.

        The original per-file crash handling, extracted so it can be reused
        both as the batch-crash fallback and (implicitly, via a batch of one)
        the only path when batch_size == 1. Failures are counted in
        stats.failed_files and never raised.

        Args:
            document: The file to chunk.
            normalized_language: Normalized language of the file.
            repo_id: Repository id to stamp on each chunk.
            source_url: Repository URL to stamp on each chunk.
            pool: Single-worker pool to run the job in.
            stats: Stats object to update in place.
            all_chunks: Accumulator list, extended in place.

        Returns:
            The pool to keep using — a fresh one if the worker crashed or
            hung.
        """
        try:
            file_chunks = self._chunk_single_document(document, normalized_language, repo_id, source_url, pool)
            self._record_file_result(document, file_chunks, stats, all_chunks)

        except (WorkerLostError, TimeoutError) as exc:
            stats.failed_files += 1
            logger.warning("Chunking crashed/hung on %s: %s", document.relative_path, exc)
            pool.terminate()
            pool = self._new_pool()

        except Exception as exc:
            stats.failed_files += 1
            logger.warning("Failed chunking %s: %s", document.relative_path, exc)

        return pool

    def _chunk_batch_with_fallback(
        self,
        batch: List[FileDocument],
        normalized_language: str,
        repo_id: str,
        source_url: str,
        pool: Pool,
        stats: ChunkingStats,
        all_chunks: List[CodeChunk],
    ) -> Pool:
        """Chunk a same-language batch in one subprocess call.

        If the batch crashes or hangs, the pool is replaced and every file
        in the batch is retried individually to isolate the failing file(s).

        Args:
            batch: Same-language files to chunk.
            normalized_language: Normalized language shared by the batch.
            repo_id: Repository id to stamp on each chunk.
            source_url: Repository URL to stamp on each chunk.
            pool: Single-worker pool to run the job in.
            stats: Stats object to update in place.
            all_chunks: Accumulator list, extended in place.

        Returns:
            The pool to keep using — a fresh one if the worker crashed or
            hung.
        """
        try:
            async_result = pool.apply_async(
                _chunkify_batch_in_subprocess,
                args=([d.content for d in batch], normalized_language, self.chunk_size, self.tokenizer),
            )
            batched_raw = _get_result_fast(pool, async_result, PER_FILE_TIMEOUT_SECONDS * len(batch))

            for document, raw_chunks in zip(batch, batched_raw):
                file_chunks = self._build_code_chunks(document, raw_chunks, repo_id, source_url)
                self._record_file_result(document, file_chunks, stats, all_chunks)

            return pool

        except (WorkerLostError, TimeoutError) as exc:
            logger.warning(
                "Batch of %d %s file(s) crashed/hung (%s) — retrying individually to isolate the failing file(s).",
                len(batch),
                normalized_language,
                exc,
            )
            pool.terminate()
            pool = self._new_pool()
            for document in batch:
                pool = self._chunk_one_with_recovery(
                    document, normalized_language, repo_id, source_url, pool, stats, all_chunks
                )
            return pool

    def chunk(
        self,
        ingestion_result: IngestionResult,
        repo_id: str = "",
        source_url: str = "",
    ) -> ChunkingResult:
        """AST-chunk every supported document in an ingestion result.

        Documents in unsupported languages are counted and skipped; files
        that crash or hang the parser are counted in stats.failed_files.

        Args:
            ingestion_result: Documents loaded by RepoLoader.
            repo_id: Repository id to stamp on each chunk.
            source_url: Repository URL to stamp on each chunk.

        Returns:
            ChunkingResult with all chunks and run stats.
        """
        stats = ChunkingStats(total_documents_received=len(ingestion_result.documents))
        all_chunks: List[CodeChunk] = []

        documents_by_language: dict = {}
        for document in ingestion_result.documents:
            self._update_language_distribution(stats, document.language)

            normalized_language = self._normalize_language(document)

            if not normalized_language or normalized_language not in AST_SUPPORTED_LANGUAGES:
                stats.skipped_unsupported_language += 1
                continue

            stats.total_ast_eligible_documents += 1
            documents_by_language.setdefault(normalized_language, []).append(document)

        pool = self._new_pool()
        try:
            for normalized_language, documents in documents_by_language.items():
                for batch_start in range(0, len(documents), self.batch_size):
                    batch = documents[batch_start : batch_start + self.batch_size]
                    pool = self._chunk_batch_with_fallback(
                        batch, normalized_language, repo_id, source_url, pool, stats, all_chunks
                    )
        finally:
            pool.terminate()

        return ChunkingResult(chunks=all_chunks, stats=stats)
