import json
from pathlib import Path
from .chunk_models import ChunkingResult

def save_chunks_jsonl_format(result: ChunkingResult, output_path: str) -> None:
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "stats": result.stats.model_dump(),
        "total_chunks": len(result.chunks),
    }

    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
