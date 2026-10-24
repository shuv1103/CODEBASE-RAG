"""OpenAI embedding client and batch embedding of chunks."""

import functools
from typing import TYPE_CHECKING, Any

from langchain_openai import OpenAIEmbeddings

from utils.config import settings

from .embeddings_formatter import (
    build_embedded_chunk,
    build_embedding_input_text,
)
from .embeddings_models import EmbeddedChunk

if TYPE_CHECKING:
    from utils.rate_limiter import ExponentialBackoff


class EmbeddingService:
    """OpenAI embedding client for code chunks and user queries.

    Args:
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY.
        model_name: OpenAI embedding model. Defaults to EMBEDDING_MODEL.
        dimensions: Optional output dimensionality; None uses the model
            default.

    Example:
        >>> service = EmbeddingService()
        >>> vector = service.embed_query("Where is the login handler?")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=model_name or settings.embedding_model,
            api_key=api_key or settings.openai_api_key.get_secret_value(),
            dimensions=dimensions,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple code or document chunks.

        Drop empty and whitespace-only texts first, so the result can be
        shorter than the input.

        Args:
            texts: List of chunk texts.

        Returns:
            List of embedding vectors, one per non-empty text.
        """
        if not texts:
            return []

        cleaned_texts = [
            text.strip() for text in texts if text and text.strip()
        ]

        if not cleaned_texts:
            return []

        return self._embeddings.embed_documents(cleaned_texts)

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a user query.

        Args:
            query: User question / retrieval query.

        Returns:
            Embedding vector for the query.

        Raises:
            ValueError: If the query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")

        return self._embeddings.embed_query(query.strip())


def embed_chunks(
    chunks: list[dict[str, Any]],
    embedder: EmbeddingService,
    backoff: "ExponentialBackoff",
    model_name: str,
) -> list[EmbeddedChunk]:
    """Embed one batch of serialized chunks, retrying rate limits with backoff.

    Args:
        chunks: Serialized chunk dicts from the chunk stage.
        embedder: Embedding client.
        backoff: Retry handler wrapping the OpenAI call.
        model_name: Embedding model name stored on each result.

    Returns:
        One EmbeddedChunk per returned vector.
    """
    texts = [build_embedding_input_text(chunk) for chunk in chunks]
    vectors = backoff.execute(
        functools.partial(embedder.embed_documents, texts)
    )
    return [
        build_embedded_chunk(
            raw_chunk=chunk, vector=vector, model_name=model_name
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
