"""User query embedding for retrieval."""

from embeddings.embeddings_client import EmbeddingService


class QueryEmbedder:
    """Embedder for natural language queries.

    Configured from OPENAI_API_KEY and EMBEDDING_MODEL.
    """

    def __init__(self) -> None:
        self._embedding_service = EmbeddingService()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query into a vector.

        Args:
            query: Natural language user query.

        Returns:
            Embedding vector for the query.

        Raises:
            ValueError: If the query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        return self._embedding_service.embed_query(query.strip())
