import os

from ingestion.repo_loader import RepoLoader
from .chunker import ASTCodeChunker
from .chunk_persist import save_chunk_json_summary, save_chunks_jsonl_format


def main():
    repo_path = os.getenv("REPO_PATH")
    if not repo_path:
        raise ValueError("REPO_PATH environment variable is required")

    # Step 1 - Ingest repository
    loader = RepoLoader(repo_path)
    ingestion_result = loader.load()

    print("=== Ingestion Stats ===")
    print(ingestion_result.stats)

    # Step 2 - AST-based chunking only
    chunker = ASTCodeChunker()
    chunking_result = chunker.chunk(ingestion_result)

    print("\n=== Chunking Stats ===")
    print(chunking_result.stats)

    # Step 3 - Persist chunk outputs
    save_chunks_jsonl_format(chunking_result, "data/chunks/code_chunks.jsonl")
    save_chunk_json_summary(chunking_result, "data/chunks/chunk_manifest.json")

    print(f"\nTotal chunks created: {len(chunking_result.chunks)}")

    if chunking_result.chunks:
        print("\n=== Sample Chunk Metadata ===")
        print(chunking_result.chunks[0].metadata.model_dump())

        print("\n=== Sample Chunk Content (first 2000 chars) ===")
        print(chunking_result.chunks[0].content[:2000])


if __name__ == "__main__":
    main()
