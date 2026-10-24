"""The agent's codebase retrieval tool."""

import asyncio
import functools
import logging

from retrieval.retrieval_service import RetrievalService
from utils.rate_limiter import ExponentialBackoff

logger = logging.getLogger(__name__)


class CodebaseRetrievalTool:
    """Agent tool that fetches relevant code chunks as answer context.

    Retrieval retries OpenAI rate-limit (429) errors with exponential
    backoff.

    Args:
        retrieval_service: Service scoped to one repo's collection. Defaults
            to a RetrievalService on the default collection.
    """

    def __init__(
        self, retrieval_service: RetrievalService | None = None
    ) -> None:
        self._retrieval_service = retrieval_service or RetrievalService()
        self._backoff = ExponentialBackoff()

    async def retrieve_code_context(self, query: str) -> str:
        """Retrieve relevant code context for a user query.

        The retrieval (OpenAI query embedding + Qdrant search, with blocking
        backoff sleeps) runs in a worker thread: ADK awaits this tool on the
        API's event loop, and a synchronous call there would stall every
        other request until it finished.

        Args:
            query: Natural language question about the codebase.

        Returns:
            Formatted code context string.
        """
        if not query or not query.strip():
            return "Query is empty. No retrieval performed."

        return await asyncio.to_thread(self.retrieve, query.strip())

    def retrieve(self, query: str) -> str:
        """Run the blocking retrieval, retrying on rate-limit errors.

        Args:
            query: Stripped, non-empty question.

        Returns:
            Formatted code context string.
        """
        return str(
            self._backoff.execute(
                functools.partial(
                    self._retrieval_service.retrieve_as_context, query=query
                )
            )
        )


# Built on first access, not at import: it creates OpenAI and Qdrant
# clients, which processes that only need the class (like the API) must not.
_codebase_retrieval_tool: CodebaseRetrievalTool | None = None


def get_codebase_retrieval_tool() -> CodebaseRetrievalTool:
    """Return the default tool singleton, building it on first call.

    Returns:
        The module-level CodebaseRetrievalTool.
    """
    global _codebase_retrieval_tool
    if _codebase_retrieval_tool is None:
        _codebase_retrieval_tool = CodebaseRetrievalTool()
    return _codebase_retrieval_tool


def __getattr__(name: str) -> CodebaseRetrievalTool:
    """Lazily resolve `codebase_retrieval_tool` (PEP 562 module __getattr__).

    Args:
        name: Attribute being looked up on the module.

    Returns:
        The default CodebaseRetrievalTool singleton.

    Raises:
        AttributeError: For any other attribute name.
    """
    if name == "codebase_retrieval_tool":
        return get_codebase_retrieval_tool()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from utils.logging_config import configure_logging

    configure_logging()
    logger.info(
        "Retrieved context:\n%s",
        asyncio.run(
            get_codebase_retrieval_tool().retrieve_code_context(
                "What is the purpose of the codebase?"
            )
        ),
    )
