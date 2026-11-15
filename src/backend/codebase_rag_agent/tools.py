"""The agent's codebase retrieval tool."""

import asyncio
import functools
import logging

from retrieval.retrieval_service import RetrievalService
from utils.rate_limiter import ExponentialBackoff

logger = logging.getLogger(__name__)


class CodebaseRetrievalTool:
    """Agent tool that returns relevant code chunks as answer context.

    Args:
        retrieval_service: Repo-scoped search. Defaults to all repos, for adk
            web debugging only.
    """

    def __init__(
        self, retrieval_service: RetrievalService | None = None
    ) -> None:
        self._retrieval_service = retrieval_service or RetrievalService(
            all_repos=True
        )
        self._backoff = ExponentialBackoff()

    # ADK sends this docstring to the LLM as the tool's description.
    async def retrieve_code_context(self, query: str) -> str:
        """Retrieve the code chunks most relevant to a codebase question.

        Args:
            query: Natural-language question about the code.

        Returns:
            "[Chunk N]" blocks with path, language, lines, score and code.
        """
        if not query or not query.strip():
            return "Query is empty. No retrieval performed."

        return await asyncio.to_thread(self.retrieve, query.strip())

    def retrieve(self, query: str) -> str:
        """Run the blocking retrieval, retrying OpenAI rate-limit errors.

        Args:
            query: Stripped, non-empty question.

        Returns:
            The formatted code context.
        """
        return self._backoff.execute(
            functools.partial(
                self._retrieval_service.retrieve_as_context, query=query
            )
        )


# Lazy: building creates OpenAI and Qdrant clients.
@functools.cache
def get_codebase_retrieval_tool() -> CodebaseRetrievalTool:
    """Return the default (all-repos) retrieval tool, built once."""
    return CodebaseRetrievalTool()


def main() -> None:
    """Log one sample all-repos retrieval (local debugging)."""
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


if __name__ == "__main__":
    main()
