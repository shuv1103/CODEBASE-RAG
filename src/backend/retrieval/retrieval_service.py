"""Retrieval service: embed a query, search, and format context."""

import logging

from embeddings.embeddings_client import EmbeddingService
from models.retrieval_models import RetrievalResponse, RetrievedChunk
from retrieval.qdrant_retriever import QdrantRetriever
from utils.config import settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """Embeds a query and searches one repo's chunks.

    Args:
        repo_id: Repo to search; required unless all_repos is True.
        all_repos: Search every repo (adk web, evaluation only).

    Raises:
        ValueError: Unless exactly one of repo_id / all_repos is given.
    """

    def __init__(
        self, repo_id: str | None = None, *, all_repos: bool = False
    ) -> None:
        # Validate before building the embedder: fail fast, no client setup.
        self._retriever = QdrantRetriever(repo_id=repo_id, all_repos=all_repos)
        self._embedder = EmbeddingService()

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> RetrievalResponse:
        """Return the chunks most relevant to a question.

        Args:
            query: Natural-language question.
            top_k: Number of chunks; defaults to TOP_K.

        Returns:
            RetrievalResponse with the ranked chunks.

        Raises:
            ValueError: If the query is blank.
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("Query cannot be empty")

        final_top_k = top_k if top_k is not None else settings.top_k

        response = self._retriever.search(
            query=query,
            query_embedding=self._embedder.embed_query(query),
            top_k=final_top_k,
        )
        logger.debug(
            "Retrieved %d chunks",
            len(response.results),
            extra={
                "top_chunks": [
                    f"{chunk.relative_path}:{chunk.start_line} ({chunk.similarity_score})"
                    for chunk in response.results
                ]
            },
        )
        return response

    def retrieve_as_context(self, query: str, top_k: int | None = None) -> str:
        """Return the most relevant chunks formatted as LLM context.

        Args:
            query: Natural-language question.
            top_k: Number of chunks; defaults to TOP_K.

        Returns:
            "[Chunk N]" blocks joined by blank lines, or a fixed "No relevant
            code chunks found" message.

        Raises:
            ValueError: If the query is blank.
        """
        response = self.retrieve(query=query, top_k=top_k)

        if not response.results:
            return "No relevant code chunks found in the codebase."

        return "\n\n".join(
            self.format_block(index, chunk)
            for index, chunk in enumerate(response.results, start=1)
        )

    @classmethod
    def format_block(cls, index: int, chunk: RetrievedChunk) -> str:
        """Format one chunk as a context block.

        Args:
            index: 1-based rank of the chunk.
            chunk: The retrieved chunk.

        Returns:
            "[Chunk N]" then Path, File, Language, Chunk Type, optional Lines,
            Similarity Score, and the code.
        """
        return (
            f"[Chunk {index}]\n"
            f"Path: {chunk.relative_path}\n"
            f"File: {chunk.file_name}\n"
            f"Language: {chunk.language}\n"
            f"Chunk Type: {chunk.chunk_type}\n"
            f"{cls.format_location(chunk.start_line, chunk.end_line)}"
            f"Similarity Score: {chunk.similarity_score}\n\n"
            f"Code:\n{chunk.content or ''}"
        )

    @staticmethod
    def format_location(start_line: int | None, end_line: int | None) -> str:
        r"""Format a block's line-range line.

        Args:
            start_line: First line, if known.
            end_line: Last line, if known.

        Returns:
            "Lines: a-b\n", or "" if either is unknown.
        """
        if start_line is None or end_line is None:
            return ""
        return f"Lines: {start_line}-{end_line}\n"
