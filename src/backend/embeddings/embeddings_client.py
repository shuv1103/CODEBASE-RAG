"""OpenAI embedding client and batch embedding of chunks."""

import functools

from langchain_openai import OpenAIEmbeddings

from embeddings.embeddings_formatter import (
    build_embedded_chunk,
    build_embedding_input_text,
)
from models.chunk_models import CodeChunk
from models.embeddings_models import EmbeddedChunk
from utils.config import settings
from utils.rate_limiter import ExponentialBackoff


class EmbeddingService:
    """OpenAI embeddings for code chunks and user queries.

    Args:
        api_key: Defaults to OPENAI_API_KEY.
        model_name: Defaults to EMBEDDING_MODEL.
        dimensions: Output size; None keeps the model default (3072).
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
            # ExponentialBackoff owns retries (one budget slot per attempt).
            max_retries=0,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in one OpenAI call, one vector per text.

        Args:
            texts: Chunk texts to embed.

        Returns:
            One vector per text, in order.

        Raises:
            ValueError: If any text is blank (vectors would no longer line
                up with their chunks).
        """
        cleaned_texts = [text.strip() for text in texts]
        if not all(cleaned_texts):
            raise ValueError("Cannot embed a blank text")
        if not cleaned_texts:
            return []
        return self._embeddings.embed_documents(cleaned_texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed one user query (RetrievalService validates it first).

        Args:
            query: The stripped, non-blank question.

        Returns:
            The query's embedding vector.
        """
        return self._embeddings.embed_query(query)


def embed_chunks(
    chunks: list[CodeChunk],
    embedder: EmbeddingService,
    backoff: ExponentialBackoff,
) -> list[EmbeddedChunk]:
    """Embed one batch of chunks, retrying rate limits.

    Args:
        chunks: The batch's code chunks.
        embedder: Embedding client.
        backoff: Retry handler wrapping the OpenAI call.

    Returns:
        One EmbeddedChunk per chunk, in order.
    """
    texts = [build_embedding_input_text(chunk) for chunk in chunks]
    vectors = backoff.execute(
        functools.partial(embedder.embed_documents, texts)
    )
    return [
        build_embedded_chunk(chunk, vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
