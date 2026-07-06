import os

from dotenv import load_dotenv

from retrieval.retrieval_service import RetrievalService
from .rate_limiter import ExponentialBackoff

load_dotenv()

_BACKOFF_MAX_RETRIES = int(os.getenv("BACKOFF_MAX_RETRIES", 5))


class CodebaseRetrievalTool:
    """
    Tool wrapper around the retrieval service.

    This is the grounded retrieval layer used by the ADK LLM agent.
    It fetches relevant code chunks from ChromaDB and formats them as context.
    API calls are protected by exponential backoff to handle transient
    429 RESOURCE_EXHAUSTED errors from the Gemini API.
    """

    def __init__(self) -> None:
        self._retrieval_service = RetrievalService()
        self._backoff = ExponentialBackoff(max_retries=_BACKOFF_MAX_RETRIES)

    def retrieve_code_context(self, query: str) -> str:
        """
        Retrieve relevant code context for a user query.

        Args:
            query: Natural language question about the codebase.

        Returns:
            Formatted code context string.
        """
        if not query or not query.strip():
            return "Query is empty. No retrieval performed."

        return str(
            self._backoff.execute(
                lambda: self._retrieval_service.retrieve_as_context(
                    query=query.strip(),
                    top_k=5,
                )
            )
        )


# Single Tool Instance
codebase_retrieval_tool = CodebaseRetrievalTool()

if __name__ == "__main__":
    print(codebase_retrieval_tool.retrieve_code_context("What is the purpose of the codebase?"))
