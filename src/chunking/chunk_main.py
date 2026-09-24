import json
import logging
import os
from pathlib import Path

from ingestion.repo_loader import RepoLoader
from utils.logging_config import configure_logging
from .chunk_models import ChunkingResult
from .chunker import ASTCodeChunker
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def save_chunks_jsonl_format(result: ChunkingResult, output_path: str) -> None:
    """Write each chunk as one JSON line of {"content", "metadata"}.

    Parent directories are created if missing.

    Args:
        result: Chunking output to persist.
        output_path: Destination .jsonl file path.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for chunk in result.chunks:
            row = {
                "content": chunk.content,
                "metadata": chunk.metadata.model_dump(),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_chunk_json_summary(result: ChunkingResult, output_path: str) -> None:
    """Write a JSON manifest with the chunking stats and total chunk count.

    Parent directories are created if missing.

    Args:
        result: Chunking output to summarize.
        output_path: Destination .json file path.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stats": result.stats.model_dump(),
        "total_chunks": len(result.chunks),
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point: load HOST_REPO_PATH, chunk it, and save the outputs.

    Writes the chunks to CHUNKS_JSON_PATH (the same file embeddings_main
    reads) and chunk_manifest.json next to it, and logs ingestion/chunking
    stats plus a sample chunk.

    Raises:
        ValueError: If HOST_REPO_PATH is not set.
    """
    configure_logging()
    repo_path = os.getenv("HOST_REPO_PATH")
    if not repo_path:
        raise ValueError("HOST_REPO_PATH environment variable is required")
    chunks_path = os.getenv("CHUNKS_JSON_PATH", "data/chunks/code_chunks.jsonl")
    manifest_path = str(Path(chunks_path).with_name("chunk_manifest.json"))

    # Step 1 - Ingest repository
    loader = RepoLoader(repo_path)
    ingestion_result = loader.load()

    logger.info("Ingestion stats: %s", ingestion_result.stats)

    # Step 2 - AST-based chunking only
    chunker = ASTCodeChunker()
    chunking_result = chunker.chunk(ingestion_result)

    logger.info("Chunking stats: %s", chunking_result.stats)

    # Step 3 - Persist chunk outputs
    save_chunks_jsonl_format(chunking_result, chunks_path)
    save_chunk_json_summary(chunking_result, manifest_path)

    logger.info("Total chunks created: %d (written to %s)", len(chunking_result.chunks), chunks_path)

    if chunking_result.chunks:
        logger.info("Sample chunk metadata: %s", chunking_result.chunks[0].metadata.model_dump())
        logger.debug("Sample chunk content (first 2000 chars):\n%s", chunking_result.chunks[0].content[:2000])


if __name__ == "__main__":
    main()
