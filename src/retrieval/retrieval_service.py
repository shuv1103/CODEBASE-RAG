import os
from dotenv import load_dotenv

from .chroma_retriever import ChromaRetriever
from .query_embedder import QueryEmbedder
from .retriever_schema import RetrievalResponse

load_dotenv()


class RetrievalService:
    """
    High-level retrieval orchestration service.

    Responsibilities:
    1. Convert user query into embedding
    2. Query Chroma vector store
    3. Return structured retrieval response
    4. Optionally format retrieval results into grounded LLM context
    """

    def __init__(self) -> None:
        vector_store_path = "C:\\Users\\SHUVRADIPS\\Documents\\RAG-Codebase-ADK\\data\\vector_store"
        collection_name = "code_chunks"

        if not vector_store_path:
            raise ValueError("VECTOR_STORE_PATH is missing in .env")
        if not collection_name:
            raise ValueError("VECTOR_STORE_COLLECTION_NAME is missing in .env")

        self._query_embedder = QueryEmbedder()
        self._retriever = ChromaRetriever(
            persist_dir=vector_store_path,
            collection_name=collection_name,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResponse:
        """
        Retrieve top-k relevant code chunks for a natural language query.
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
        """
        Retrieve relevant code chunks and format them into a grounded textual context
        suitable for an LLM prompt/tool response.
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
        if symbol_name and parent_symbol:
            return f"Symbol: {parent_symbol}.{symbol_name}\n"
        if symbol_name:
            return f"Symbol: {symbol_name}\n"
        return ""

    @staticmethod
    def _format_location(start_line: int | None, end_line: int | None) -> str:
        if start_line is not None and end_line is not None:
            return f"Lines: {start_line}-{end_line}\n"
        if start_line is not None:
            return f"Start Line: {start_line}\n"
        return ""
