from typing import Any, Dict, List
from .embeddings_models import EmbeddedChunk


def estimate_token_count(text: str) -> int:
    """Roughly estimate a text's token count as len(text) // 4.

    Args:
        text: Text to measure.

    Returns:
        0 for empty text, otherwise at least 1.

    Example:
        >>> estimate_token_count("def f(): pass")
        3
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_embedding_input_text(chunk: Dict[str, Any]) -> str:
    """Build the text that gets embedded for a chunk.

    Prefixes the code with File/Path/Language and, when known, Symbol and
    Lines headers so the embedding captures where the code lives.

    Args:
        chunk: Serialized CodeChunk dict with "content" and "metadata".

    Returns:
        The newline-joined header lines followed by "Code:" and the content.

    Example:
        >>> build_embedding_input_text({
        ...     "content": "def f(): pass",
        ...     "metadata": {"file_name": "a.py", "relative_path": "pkg/a.py",
        ...                  "language": "python", "start_line": 0, "end_line": 0},
        ... })
        'File: a.py\\nPath: pkg/a.py\\nLanguage: python\\nLines: 0-0\\nCode:\\ndef f(): pass'
    """
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
    """Combine a serialized chunk and its vector into an EmbeddedChunk.

    Args:
        raw_chunk: Serialized CodeChunk dict with "content" and "metadata".
        vector: Embedding vector for the chunk.
        model_name: Embedding model that produced the vector.

    Returns:
        The EmbeddedChunk, with embedding_input_text and token_estimate
        derived from the chunk.
    """
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
        repo_id=metadata.get("repo_id", ""),
        source_url=metadata.get("source_url", ""),
        content=raw_chunk.get("content"),
        embedding_input_text=embedding_input_text,
        embedding_model=model_name,
        embedding_dimension=len(vector),
        vector=vector,
        token_estimate=estimate_token_count(embedding_input_text),
    )
