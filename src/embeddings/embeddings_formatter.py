from typing import Any, Dict, List
from .embeddings_models import EmbeddedChunk


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_embedding_input_text(chunk: Dict[str, Any]) -> str:
    content = chunk.get("content", "")
    metadata = chunk.get("metadata", {}) or {}

    relative_path = metadata.get("relative_path", "")
    file_name = metadata.get("file_name", "")
    language = metadata.get("language", "unknown")
    symbol_name = metadata.get("symbol_name")
    start_line = metadata.get("start_line")
    end_line = metadata.get("end_line")

    parts: List[str] = [
        f"File: {file_name}",
        f"Path: {relative_path}",
        f"Language: {language}",
    ]

    if symbol_name:
        parts.append(f"Symbol: {symbol_name}")

    if start_line is not None and end_line is not None:
        parts.append(f"Lines: {start_line}-{end_line}")

    parts.append("Code:")
    parts.append(content)

    return "\n".join(parts).strip()


def build_embedded_chunk(
    raw_chunk: Dict[str, Any],
    vector: List[float],
    model_name: str,
) -> EmbeddedChunk:
    metadata = raw_chunk.get("metadata", {}) or {}
    extra = metadata.get("extra", {}) or {}

    embedding_input_text = build_embedding_input_text(raw_chunk)

    return EmbeddedChunk(
        chunk_id=metadata.get("chunk_id", ""),
        relative_path=metadata.get("relative_path", ""),
        file_name=metadata.get("file_name", ""),
        language=metadata.get("language", "unknown"),
        chunk_type=metadata.get("chunking_strategy", "ast"),
        symbol_name=metadata.get("symbol_name"),
        parent_symbol=extra.get("parent_symbol"),
        start_line=metadata.get("start_line"),
        end_line=metadata.get("end_line"),
        content=raw_chunk.get("content"),
        embedding_input_text=embedding_input_text,
        embedding_model=model_name,
        embedding_dimension=len(vector),
        vector=vector,
        token_estimate=estimate_token_count(embedding_input_text),
    )
