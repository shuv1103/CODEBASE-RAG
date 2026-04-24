import json
import os
from collections import Counter
from json import JSONDecodeError

from dotenv import load_dotenv
import time

from .chroma_store import ChromaVectorStore
from .embeddings_client import EmbeddingService
from .embeddings_formatter import build_embedded_chunk, build_embedding_input_text


load_dotenv()


def load_chunks(file_path: str) -> list[dict]:
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
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
    chunks_json_path = os.getenv("CHUNKS_JSON_PATH")
    output_dimensionality = os.getenv("EMBEDDING_OUTPUT_DIMENSION")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing in .env")

    if not chunks_json_path:
        raise ValueError("CHUNKS_JSON_PATH missing in .env")


    if output_dimensionality and output_dimensionality.strip():
        output_dimensionality = int(output_dimensionality)
    else:
        output_dimensionality = None


    # load the chunks from the JSONL file created by the chunking step
    chunks = load_chunks(chunks_json_path)

    if not chunks:
        print("No chunks found to embed.")
        return

    embedder = EmbeddingService(api_key, model_name)

    vector_store = ChromaVectorStore()

    total_success = 0
    total_failed = 0
    total_batches = 0
    language_counter = Counter()
    chunk_type_counter = Counter()

    for batch_number, batch in enumerate(batchify(chunks, batch_size), start=1):
        total_batches += 1
        time.sleep(60)
        texts = [build_embedding_input_text(chunk) for chunk in batch]
        
        try:
            vectors = embedder.embed_documents(texts)
        except Exception as e:
            print(f"Batch {batch_number} failed: {e}")
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

        print(f"Batch {batch_number} stored: {len(embedded_chunks)} chunks")

    print("\n=== EMBEDDING COMPLETE ===")
    print(f"Total loaded: {len(chunks)}")
    print(f"Embedded success: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"Batches processed: {total_batches}")
    print(f"Model: {model_name}")
    print(f"Languages: {dict(language_counter)}")
    print(f"Chunk types: {dict(chunk_type_counter)}")
    print(f"Stored vectors: {vector_store.count()}")


if __name__ == "__main__":
    main()
