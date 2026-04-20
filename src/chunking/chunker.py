from typing import List, Optional

from astchunk import ASTChunkBuilder

from ingestion.repo_loader import FileDocument, IngestionResult

from utils.chunk_config import (
    AST_SUPPORTED_LANGUAGES,
    AST_LANGUAGE_ALIASES,
    DEFAULT_AST_MAX_CHUNK_SIZE,
    DEFAULT_AST_CHUNK_OVERLAP,
    DEFAULT_AST_CHUNK_EXPANSION,
    DEFAULT_METADATA_TEMPLATE,
)
from .chunk_models import CodeChunk, ChunkMetadata, ChunkingResult, ChunkingStats


class ASTCodeChunker:
    """
    AST-only chunker for astchunk-supported languages:
    - python
    - java
    - csharp
    - typescript
    """

    def __init__(
        self,
        max_chunk_size: int = DEFAULT_AST_MAX_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_AST_CHUNK_OVERLAP,
        chunk_expansion: bool = DEFAULT_AST_CHUNK_EXPANSION,
        metadata_template: str = DEFAULT_METADATA_TEMPLATE,
    ):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_expansion = chunk_expansion
        self.metadata_template = metadata_template

    def _normalize_language(self, document: FileDocument) -> Optional[str]:
        language = (document.language or "").strip().lower()
        return AST_LANGUAGE_ALIASES.get(language)

    def _update_language_distribution(self, stats: ChunkingStats, language: str) -> None:
        stats.language_distribution[language] = stats.language_distribution.get(language, 0) + 1

    def _chunk_single_document(self, document: FileDocument, normalized_language: str) -> List[CodeChunk]:
        builder = ASTChunkBuilder(
            max_chunk_size=self.max_chunk_size,
            language=normalized_language,
            metadata_template=self.metadata_template,
        )

        repo_level_metadata = {
            "filepath": document.relative_path,
            "filename": document.file_name,
            "language": document.language,
        }

        raw_chunks = builder.chunkify(
            document.content,
            repo_level_metadata=repo_level_metadata,
            chunk_overlap=self.chunk_overlap,
            chunk_expansion=self.chunk_expansion,
        )

        total = len(raw_chunks)
        result: List[CodeChunk] = []

        for idx, raw in enumerate(raw_chunks):
            raw_metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
            raw_content = raw.get("content", "") if isinstance(raw, dict) else ""

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
                chunking_strategy="ast",
                symbol_name=raw_metadata.get("symbol_name"),
                start_line=raw_metadata.get("start_line"),
                end_line=raw_metadata.get("end_line"),
                extra={
                    "astchunk_metadata": raw_metadata,
                },
            )

            result.append(
                CodeChunk(
                    content=raw_content,
                    metadata=metadata,
                )
            )

        return result

    def chunk(self, ingestion_result: IngestionResult) -> ChunkingResult:
        stats = ChunkingStats(total_documents_received=len(ingestion_result.documents))
        all_chunks: List[CodeChunk] = []

        for document in ingestion_result.documents:
            self._update_language_distribution(stats, document.language)

            normalized_language = self._normalize_language(document)

            if not normalized_language or normalized_language not in AST_SUPPORTED_LANGUAGES:
                stats.skipped_unsupported_language += 1
                continue

            stats.total_ast_eligible_documents += 1

            try:
                file_chunks = self._chunk_single_document(document, normalized_language)

                if file_chunks:
                    stats.total_documents_chunked += 1
                    stats.total_chunks_created += len(file_chunks)
                    all_chunks.extend(file_chunks)

            except Exception as exc:
                stats.failed_files += 1
                print(f"[WARN] Failed chunking {document.relative_path}: {exc}")

        return ChunkingResult(chunks=all_chunks, stats=stats)
