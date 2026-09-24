from embeddings.embeddings_client import EmbeddingService
import os
from dotenv import load_dotenv

load_dotenv()

class QueryEmbedder:
    """Converts natural language queries into embeddings.

    Uses the shared EmbeddingService, configured from the OPENAI_API_KEY and
    EMBEDDING_MODEL environment variables.
    """

    def __init__(self) -> None:
        self._embedding_service = EmbeddingService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
        )

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
