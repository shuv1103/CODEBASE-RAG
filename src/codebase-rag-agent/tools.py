from typing import Optional

from retrieval.retriever_service import RetrievalService


class CodebaseRetrievalTool:
    """
    Tool wrapper around the retrieval service.

    This is the grounded retrieval layer used by the ADK LLM agent.
    It fetches relevant code chunks from ChromaDB and formats them as context.
    """

    def __init__(self) -> None:
        self._retrieval_service = RetrievalService()

    def retrieve_code_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve relevant code context for a user query.

        Args:
            query: Natural language question about the codebase.
            top_k: Optional number of chunks to retrieve.

        Returns:
            Formatted code context string.
        """
        if not query or not query.strip():
            return "Query is empty. No retrieval performed."

        return self._retrieval_service.retrieve_as_context(
            query=query.strip(),
            top_k=top_k,
        )


# Single Tool Instance
codebase_retrieval_tool = CodebaseRetrievalTool()
