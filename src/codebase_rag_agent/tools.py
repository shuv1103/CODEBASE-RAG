import logging
import os

from dotenv import load_dotenv

from retrieval.retrieval_service import RetrievalService
from .rate_limiter import ExponentialBackoff

load_dotenv()

logger = logging.getLogger(__name__)

_BACKOFF_MAX_RETRIES = int(os.getenv("BACKOFF_MAX_RETRIES", 5))


class CodebaseRetrievalTool:
    """Tool wrapper around the retrieval service.

    This is the grounded retrieval layer used by the ADK LLM agent.
    It fetches relevant code chunks from Qdrant and formats them as context.
    API calls are protected by exponential backoff to handle transient
    429 RESOURCE_EXHAUSTED errors from the Gemini API.

    Args:
        retrieval_service: Service scoped to one repo's collection. Defaults
            to a RetrievalService on the default collection.
    """

    def __init__(self, retrieval_service: RetrievalService | None = None) -> None:
        self._retrieval_service = retrieval_service or RetrievalService()
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
                )
            )
        )


# Single Tool Instance — built lazily (on first access) instead of at import
# time. CodebaseRetrievalTool() eagerly builds a RetrievalService (embedding
# + vector-store clients), so building it as an import-time side effect breaks any
# process that merely imports this module without needing this particular
# singleton — e.g. the API process, which only needs `CodebaseRetrievalTool`
# the class (to build per-repo_id instances via AgentRegistry) and never
# touches this default one.
_codebase_retrieval_tool: CodebaseRetrievalTool | None = None


def _get_codebase_retrieval_tool() -> CodebaseRetrievalTool:
    """Return the default tool singleton, building it on first call.

    Returns:
        The module-level CodebaseRetrievalTool.
    """
    global _codebase_retrieval_tool
    if _codebase_retrieval_tool is None:
        _codebase_retrieval_tool = CodebaseRetrievalTool()
    return _codebase_retrieval_tool


def __getattr__(name: str):
    """Lazily resolve `codebase_retrieval_tool` (PEP 562 module __getattr__).

    Args:
        name: Attribute being looked up on the module.

    Returns:
        The default CodebaseRetrievalTool singleton.

    Raises:
        AttributeError: For any other attribute name.
    """
    if name == "codebase_retrieval_tool":
        return _get_codebase_retrieval_tool()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from utils.logging_config import configure_logging

    configure_logging()
    logger.info(
        "Retrieved context:\n%s",
        _get_codebase_retrieval_tool().retrieve_code_context("What is the purpose of the codebase?"),
    )
