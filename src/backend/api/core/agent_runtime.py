"""Agent runtime: per-repo ADK runners and the per-process LLM run cap."""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from google.adk.runners import InMemoryRunner

from api.core.exceptions import LLMBusyError
from codebase_rag_agent.agent import build_agent
from codebase_rag_agent.tools import CodebaseRetrievalTool
from retrieval.retrieval_service import RetrievalService
from utils.config import settings

logger = logging.getLogger(__name__)

# Listing sessions copies each one, so sweep at most this often per repo.
_SESSION_SWEEP_INTERVAL_SECONDS = 60


def build_runner(repo_id: str) -> InMemoryRunner:
    """Build an ADK runner whose retrieval searches only one repo.

    Args:
        repo_id: Repo the agent's retrieval tool is scoped to.

    Returns:
        A new InMemoryRunner with its own in-memory session store.
    """
    retrieval_service = RetrievalService(repo_id=repo_id)
    tool = CodebaseRetrievalTool(retrieval_service=retrieval_service)
    return InMemoryRunner(agent=build_agent(tool))


class AgentRegistry:
    """LRU cache of one ADK runner, and its chat sessions, per repo.

    Bounds process memory: holds at most max_repos runners and deletes
    sessions idle past session_ttl_seconds.

    Args:
        max_repos: Most runners kept. Defaults to AGENT_REGISTRY_MAX_REPOS.
        session_ttl_seconds: Idle time before a session is deleted. Defaults
            to CHAT_SESSION_TTL_SECONDS.
        build_runner: Factory called with a repo_id on a cache miss.
    """

    def __init__(
        self,
        *,
        max_repos: int | None = None,
        session_ttl_seconds: int | None = None,
        build_runner: Callable[[str], InMemoryRunner] = build_runner,
    ) -> None:
        self._lock = threading.Lock()
        self._runners: OrderedDict[str, InMemoryRunner] = OrderedDict()
        self._last_swept: dict[str, float] = {}
        self._max_repos = (
            max_repos
            if max_repos is not None
            else settings.agent_registry_max_repos
        )
        self._session_ttl_seconds = (
            session_ttl_seconds
            if session_ttl_seconds is not None
            else settings.chat_session_ttl_seconds
        )
        self._build_runner = build_runner

    def get_runner(self, repo_id: str) -> InMemoryRunner:
        """Return the repo's runner, building it on first use (thread-safe).

        Marks the repo most recently used and evicts the least recently used
        runner past max_repos.

        Args:
            repo_id: Repo to get the runner for.

        Returns:
            The cached or newly built InMemoryRunner.
        """
        with self._lock:
            runner = self._runners.get(repo_id)
            if runner is not None:
                self._runners.move_to_end(repo_id)
                return runner

        # Built outside the lock: client setup would stall other repos.
        # ponytail: two first chats for one repo may both build; one wins.
        built = self._build_runner(repo_id)
        with self._lock:
            runner = self._runners.setdefault(repo_id, built)
            self._runners.move_to_end(repo_id)

            while len(self._runners) > self._max_repos:
                evicted_repo_id, _ = self._runners.popitem(last=False)
                self._last_swept.pop(evicted_repo_id, None)
                logger.info(
                    "Evicted idle chat agent for repo %s", evicted_repo_id
                )
            return runner

    async def expire_idle_sessions(
        self, repo_id: str, runner: InMemoryRunner, user_id: str
    ) -> None:
        """Delete the repo's sessions idle past the TTL, at most once a minute.

        Args:
            repo_id: Repo the runner belongs to.
            runner: The repo's runner, whose session service is swept.
            user_id: ADK user id the repo's sessions are stored under.
        """
        now = time.time()
        # get_runner edits _last_swept from a worker thread.
        with self._lock:
            if (
                now - self._last_swept.get(repo_id, 0.0)
                < _SESSION_SWEEP_INTERVAL_SECONDS
            ):
                return
            self._last_swept[repo_id] = now

        listed = await runner.session_service.list_sessions(
            app_name=runner.app_name, user_id=user_id
        )
        for session in listed.sessions:
            if now - session.last_update_time > self._session_ttl_seconds:
                await runner.session_service.delete_session(
                    app_name=runner.app_name,
                    user_id=user_id,
                    session_id=session.id,
                )


class LLMConcurrencyLimiter:
    """Per-process cap on agent runs in flight; the overflow waits briefly.

    Args:
        max_concurrent: Most runs in flight. Defaults to
            MAX_CONCURRENT_LLM_CALLS.
        queue_timeout_seconds: Longest wait for a slot. Defaults to
            LLM_QUEUE_TIMEOUT_SECONDS.

    Example:
        >>> async with LLMConcurrencyLimiter().slot():
        ...     await run_agent()
    """

    def __init__(
        self,
        *,
        max_concurrent: int | None = None,
        queue_timeout_seconds: float | None = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(
            max_concurrent
            if max_concurrent is not None
            else settings.max_concurrent_llm_calls
        )
        self._queue_timeout_seconds = (
            queue_timeout_seconds
            if queue_timeout_seconds is not None
            else settings.llm_queue_timeout_seconds
        )

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one slot for the duration of the `async with` block.

        Raises:
            LLMBusyError: If no slot frees up within queue_timeout_seconds
                (retry_after_seconds=5).
        """
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._queue_timeout_seconds
            )
        except TimeoutError as exc:
            raise LLMBusyError(retry_after_seconds=5) from exc
        try:
            yield
        finally:
            self._semaphore.release()
