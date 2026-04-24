from typing import Optional

from retrieval.retrieval_service import RetrievalService


class CodebaseRetrievalTool:
    """
    Tool wrapper around the retrieval service.

    This is the grounded retrieval layer used by the ADK LLM agent.
    It fetches relevant code chunks from ChromaDB and formats them as context.
    """

    def __init__(self) -> None:
        self._retrieval_service = RetrievalService()

    def retrieve_code_context(self, query: str) -> str:
        """
        Retrieve relevant code context for a user query.

        Args:
            query: Natural language question about the codebase.

        Returns:
            Formatted code context string.
        """
        
        top_k = 5
        if not query or not query.strip():
            return "Query is empty. No retrieval performed."

        return str(self._retrieval_service.retrieve_as_context(
            query=query.strip(),
            top_k=top_k,
        ))


# Single Tool Instance
codebase_retrieval_tool = CodebaseRetrievalTool()

if __name__ == "__main__":
    print(codebase_retrieval_tool.retrieve_code_context("What is the purpose of the codebase?"))
