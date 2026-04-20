from embeddings.embedding_client import EmbeddingService


class QueryEmbedder:
    """
    Converts natural language queries into embeddings
    using the shared EmbeddingService.
    """

    def __init__(self) -> None:
        self._embedding_service = EmbeddingService()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query into a vector.

        Args:
            query: Natural language user query.

        Returns:
            Embedding vector for the query.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        return self._embedding_service.embed_query(query.strip())
