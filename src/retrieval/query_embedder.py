from embeddings.embeddings_client import EmbeddingService
import os
from dotenv import load_dotenv

load_dotenv()

class QueryEmbedder:
    """
    Converts natural language queries into embeddings
    using the shared EmbeddingService.
    """

    def __init__(self) -> None:
        self._embedding_service = EmbeddingService(api_key=os.getenv("GOOGLE_API_KEY"), model_name=os.getenv("EMBEDDING_MODEL"))

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
