import os
from dotenv import load_dotenv
from .qdrant_retriever import QdrantRetriever
from .query_embedder import QueryEmbedder
from .retrieval_models import RetrievalResponse

load_dotenv()


class RetrievalService:
    """High-level retrieval orchestration service.

    Responsibilities:
    1. Convert user query into embedding
    2. Query the Qdrant vector store, scoped to one repo
    3. Return structured retrieval response
    4. Optionally format retrieval results into grounded LLM context

    Args:
        repo_id: Repository to search. None searches every repo in the
            collection (local dev tooling and evaluation).

    Example:
        >>> service = RetrievalService(repo_id="c3460b930023")
        >>> context = service.retrieve_as_context("How is auth handled?")
    """

    def __init__(self, repo_id: str | None = None) -> None:
        self._query_embedder = QueryEmbedder()
        self._retriever = QdrantRetriever(repo_id=repo_id)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResponse:
        """Retrieve the top-k relevant code chunks for a natural language query.

        Args:
            query: Natural language user query.
            top_k: Number of chunks to return. Defaults to the TOP_K env
                var (5 if unset).

        Returns:
            RetrievalResponse with the ranked chunks.

        Raises:
            ValueError: If the query is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        default_top_k = int(os.getenv("TOP_K", "5"))
        final_top_k = top_k if top_k is not None else default_top_k

        query_embedding = self._query_embedder.embed_query(query.strip())

        return self._retriever.search(
            query=query.strip(),
            query_embedding=query_embedding,
            top_k=final_top_k,
        )

    def retrieve_as_context(self, query: str, top_k: int | None = None) -> str:
        """Retrieve relevant chunks and format them as grounded LLM context.

        The output is suitable for an LLM prompt or tool response.

        Args:
            query: Natural language user query.
            top_k: Number of chunks to include. Defaults to the TOP_K env var.

        Returns:
            One "[Chunk N]" text block per chunk (path, language, symbol,
            lines, score, code), or a fixed message when nothing matched.

        Raises:
            ValueError: If the query is empty or whitespace-only.
        """
        response = self.retrieve(query=query, top_k=top_k)

        if not response.results:
            return "No relevant code chunks found in the codebase."

        blocks: list[str] = []

        for index, chunk in enumerate(response.results, start=1):
            location = self._format_location(chunk.start_line, chunk.end_line)
            symbol = self._format_symbol(chunk.symbol_name, chunk.parent_symbol)

            block = (
                f"[Chunk {index}]\n"
                f"Path: {chunk.relative_path}\n"
                f"File: {chunk.file_name}\n"
                f"Language: {chunk.language}\n"
                f"Chunk Type: {chunk.chunk_type}\n"
                f"{symbol}"
                f"{location}"
                f"Similarity Score: {chunk.similarity_score}\n\n"
                f"Code:\n{chunk.content or ''}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    @staticmethod
    def _format_symbol(symbol_name: str | None, parent_symbol: str | None) -> str:
        """Format the "Symbol:" line of a context block.

        Args:
            symbol_name: Chunk's symbol name, if any.
            parent_symbol: Enclosing symbol name, if any.

        Returns:
            "Symbol: parent.name\\n", "Symbol: name\\n", or "" when unknown.
        """
        if symbol_name and parent_symbol:
            return f"Symbol: {parent_symbol}.{symbol_name}\n"
        if symbol_name:
            return f"Symbol: {symbol_name}\n"
        return ""

    @staticmethod
    def _format_location(start_line: int | None, end_line: int | None) -> str:
        """Format the line-range line of a context block.

        Args:
            start_line: First line of the chunk, if known.
            end_line: Last line of the chunk, if known.

        Returns:
            "Lines: a-b\\n", "Start Line: a\\n", or "" when unknown.
        """
        if start_line is not None and end_line is not None:
            return f"Lines: {start_line}-{end_line}\n"
        if start_line is not None:
            return f"Start Line: {start_line}\n"
        return ""
