import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


class EmbeddingService:
    """
    This service provides:
    - document embeddings for code chunks
    - query embeddings for semantic retrieval
    """

    def __init__(self) -> None:
        self._api_key = os.getenv("GOOGLE_API_KEY")
        self._model_name = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY is missing in .env")

        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._model_name,
            google_api_key=self._api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple code/document chunks.

        Args:
            texts: List of chunk texts.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        cleaned_texts = [text.strip() for text in texts if text and text.strip()]

        if not cleaned_texts:
            return []

        return self._embeddings.embed_documents(cleaned_texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a user query.

        Args:
            query: User question / retrieval query.

        Returns:
            Embedding vector for the query.
        """
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty")

        return self._embeddings.embed_query(query.strip())
