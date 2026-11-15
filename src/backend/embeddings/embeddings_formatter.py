"""Embedding input text and EmbeddedChunk records built from chunks."""

from models.chunk_models import CodeChunk
from models.embeddings_models import EmbeddedChunk


def estimate_token_count(text: str) -> int:
    """Roughly estimate a text's token count as len(text) // 4.

    Args:
        text: Text to measure.

    Returns:
        0 for empty text, otherwise at least 1.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def build_embedding_input_text(chunk: CodeChunk) -> str:
    """Build the text embedded for a chunk: location headers, then the code.

    Args:
        chunk: The code chunk.

    Returns:
        "File:", "Path:" and "Language:" lines (plus "Lines:" when known),
        then "Code:" and the content.
    """
    metadata = chunk.metadata
    parts = [
        f"File: {metadata.file_name}",
        f"Path: {metadata.relative_path}",
        f"Language: {metadata.language}",
    ]
    if metadata.start_line is not None and metadata.end_line is not None:
        parts.append(f"Lines: {metadata.start_line}-{metadata.end_line}")
    parts.extend(("Code:", chunk.content))
    return "\n".join(parts).strip()


def build_embedded_chunk(
    chunk: CodeChunk, vector: list[float]
) -> EmbeddedChunk:
    """Combine a chunk and its vector into an EmbeddedChunk.

    Args:
        chunk: The code chunk.
        vector: The chunk's embedding.

    Returns:
        The EmbeddedChunk, with its embedding input text.
    """
    metadata = chunk.metadata
    return EmbeddedChunk(
        chunk_id=metadata.chunk_id,
        relative_path=metadata.relative_path,
        file_name=metadata.file_name,
        language=metadata.language,
        chunk_type=metadata.chunking_strategy,
        start_line=metadata.start_line,
        end_line=metadata.end_line,
        repo_id=metadata.repo_id,
        embedding_input_text=build_embedding_input_text(chunk),
        vector=vector,
    )
