import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class EmbeddingService:
    """OpenAI embedding client for code chunks and user queries.

    This service provides:
    - document embeddings for code chunks
    - query embeddings for semantic retrieval

    Args:
        api_key: OpenAI API key.
        model_name: OpenAI embedding model (e.g. "text-embedding-3-large").
        dimensions: Optional output dimensionality; None uses the model
            default.

    Raises:
        ValueError: If api_key is empty.

    Example:
        >>> service = EmbeddingService(os.getenv("OPENAI_API_KEY"), "text-embedding-3-large")
        >>> vector = service.embed_query("Where is the login handler?")
    """

    def __init__(self, api_key, model_name, dimensions=None) -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._dimensions = dimensions

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env")

        self._embeddings = OpenAIEmbeddings(
            model=self._model_name,
            api_key=self._api_key,
            dimensions=self._dimensions,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple code/document chunks.

        Empty and whitespace-only texts are dropped before embedding, so the
        result can be shorter than the input.

        Args:
            texts: List of chunk texts.

        Returns:
            List of embedding vectors, one per non-empty text.
        """
        if not texts:
            return []

        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            return []

        return self._embeddings.embed_documents(cleaned_texts)

    def embed_query(self, query: str) -> List[float]:
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
