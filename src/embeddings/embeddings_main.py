import json
import logging
import os
from collections import Counter
from json import JSONDecodeError

from dotenv import load_dotenv

from codebase_rag_agent.rate_limiter import ExponentialBackoff
from utils.logging_config import configure_logging

from .qdrant_store import QdrantVectorStore
from .embeddings_client import EmbeddingService
from .embeddings_formatter import build_embedded_chunk, build_embedding_input_text


load_dotenv()

logger = logging.getLogger(__name__)


def load_chunks(file_path: str) -> list[dict]:
    """Load serialized chunks from a JSON or JSONL file.

    The file is first parsed as a single JSON document (a list, or one
    object); if that fails, it is parsed as JSONL, skipping blank lines.

    Args:
        file_path: Path of the chunks file.

    Returns:
        List of serialized chunk dicts.

    Raises:
        ValueError: If a JSONL line is not valid JSON.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except JSONDecodeError:
        chunks = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL in {file_path}: {e}") from e
        return chunks


def batchify(items: list[dict], batch_size: int):
    """Yield consecutive slices of at most batch_size items.

    Args:
        items: Items to split.
        batch_size: Maximum items per batch.

    Yields:
        Lists of up to batch_size items, in order.

    Example:
        >>> list(batchify([1, 2, 3], 2))
        [[1, 2], [3]]
    """
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def main():
    """CLI entry point: embed the chunks file and upsert it into Qdrant.

    Reads CHUNKS_JSON_PATH, embeds in EMBEDDING_BATCH_SIZE batches with
    exponential backoff, stores each batch, and logs a summary. A batch
    that still fails after retries is counted and skipped.

    Raises:
        ValueError: If OPENAI_API_KEY or CHUNKS_JSON_PATH is not set.
    """
    configure_logging()
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
    chunks_json_path = os.getenv("CHUNKS_JSON_PATH")
    output_dimensionality = os.getenv("EMBEDDING_OUTPUT_DIMENSION")

    if not api_key:
        raise ValueError("OPENAI_API_KEY missing in .env")

    if not chunks_json_path:
        raise ValueError("CHUNKS_JSON_PATH missing in .env")


    if output_dimensionality and output_dimensionality.strip():
        output_dimensionality = int(output_dimensionality)
    else:
        output_dimensionality = None


    # load chunks from the JSONL file produced by the chunking step
    chunks = load_chunks(chunks_json_path)

    if not chunks:
        logger.warning("No chunks found to embed in %s", chunks_json_path)
        return

    embedder = EmbeddingService(api_key, model_name, output_dimensionality)
    backoff = ExponentialBackoff(max_retries=int(os.getenv("BACKOFF_MAX_RETRIES", "5")))

    vector_store = QdrantVectorStore()

    total_success = 0
    total_failed = 0
    total_batches = 0
    language_counter = Counter()
    chunk_type_counter = Counter()

    for batch_number, batch in enumerate(batchify(chunks, batch_size), start=1):
        total_batches += 1
        texts = [build_embedding_input_text(chunk) for chunk in batch]

        try:
            vectors = backoff.execute(lambda texts=texts: embedder.embed_documents(texts))
        except Exception as e:
            logger.warning("Batch %d failed: %s", batch_number, e)
            total_failed += len(batch)
            continue

        embedded_chunks = []
        for chunk, vector in zip(batch, vectors):
            embedded_chunk = build_embedded_chunk(
                raw_chunk=chunk,
                vector=vector,
                model_name=model_name,
            )
            embedded_chunks.append(embedded_chunk)

            language_counter[embedded_chunk.language] += 1
            chunk_type_counter[embedded_chunk.chunk_type] += 1

        vector_store.upsert(embedded_chunks)
        total_success += len(embedded_chunks)

        logger.info("Batch %d stored: %d chunks", batch_number, len(embedded_chunks))

    logger.info(
        "Embedding complete: loaded=%d embedded=%d failed=%d batches=%d model=%s "
        "languages=%s chunk_types=%s stored_vectors=%d",
        len(chunks),
        total_success,
        total_failed,
        total_batches,
        model_name,
        dict(language_counter),
        dict(chunk_type_counter),
        vector_store.count(),
    )


if __name__ == "__main__":
    main()
