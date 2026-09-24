import asyncio
import logging
import threading

from google.adk.agents.invocation_context import new_invocation_context_id
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types

from caching.semantic_cache_repository import LangCacheRepository
from caching.semantic_cache_service import SemanticCacheService
from codebase_rag_agent.agent import build_agent
from codebase_rag_agent.tools import CodebaseRetrievalTool
from codebase_rag_agent.tracing import log_cache_event
from retrieval.retrieval_service import RetrievalService
from utils.repo_id import compute_repo_id, normalize_github_url
from worker.tasks import ingest_repo_task

from .exceptions import (
    AgentResponseError,
    InvalidRepoUrlError,
    RepoNotFoundError,
    RepoNotReadyError,
)
from .redis_status_store import RedisStatusStore
from .status_store import RepoStatus, StatusStore

logger = logging.getLogger(__name__)

_status_store: StatusStore = RedisStatusStore()


class AgentRegistry:
    """Lazily builds and caches one ADK agent/runner per repo_id.

    Each runner is scoped to that repo's chunks (Qdrant repo_id filter) via a
    dedicated RetrievalService. Built once per repo_id and reused so ADK
    session state (conversation history) persists across chat turns for
    that repo.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runners: dict[str, InMemoryRunner] = {}

    def get_runner(self, repo_id: str) -> InMemoryRunner:
        """Return the repo's runner, building it on first use (thread-safe).

        Args:
            repo_id: Repo whose chunks the agent's retrieval is scoped to.

        Returns:
            The cached InMemoryRunner for the repo.
        """
        with self._lock:
            runner = self._runners.get(repo_id)
            if runner is None:
                retrieval_service = RetrievalService(repo_id=repo_id)
                tool = CodebaseRetrievalTool(retrieval_service=retrieval_service)
                runner = InMemoryRunner(agent=build_agent(tool))
                self._runners[repo_id] = runner
            return runner


_agent_registry = AgentRegistry()
_semantic_cache_service = SemanticCacheService(LangCacheRepository())


class RagWorkflowFacade:
    """Single entry point for the ingest -> status -> chat workflow.

    Facade Pattern: the API routes only talk to this instead of wiring
    StatusStore, AgentRegistry, ingest_pipeline, and the repo_id helpers
    together themselves.

    Args:
        status_store: Store for repo ingestion status.
        agent_registry: Per-repo agent/runner cache.
        semantic_cache_service: Two-tier answer cache.
    """

    def __init__(
        self,
        status_store: StatusStore,
        agent_registry: AgentRegistry,
        semantic_cache_service: SemanticCacheService,
    ) -> None:
        self._status_store = status_store
        self._agent_registry = agent_registry
        self._semantic_cache_service = semantic_cache_service

    def start_ingestion(self, url: str) -> tuple[str, str, bool, str]:
        """Validate the URL, apply the dedup check, and create a status record.

        The dedup check is a single atomic create_if_absent() call rather
        than a separate get()-then-create(), so two concurrent POSTs for the
        same brand-new repo can't both observe "doesn't exist yet" and both
        enqueue an ingestion (see StatusStore.create_if_absent). A previously
        failed repo is reset to "pending" (and its cache invalidated) so it
        can be retried.

        Args:
            url: User-supplied GitHub repository URL.

        Returns:
            (repo_id, status, is_new, normalized_url). is_new=False means an
            ingestion for this repo is already in-flight or ready — the
            caller should not enqueue a new ingestion task.

        Raises:
            InvalidRepoUrlError: If the URL is not a github.com repo URL.
        """
        try:
            normalized_url = normalize_github_url(url)
        except ValueError as exc:
            raise InvalidRepoUrlError(str(exc)) from exc

        repo_id = compute_repo_id(normalized_url)
        record, created = self._status_store.create_if_absent(repo_id, normalized_url)

        if created:
            return repo_id, "pending", True, normalized_url

        if record.status == "failed":
            # A retry re-runs the full clone->chunk->embed->store pipeline
            # (§10.4), so any answers already cached against the old/partial
            # data must not survive into the re-ingested repo.
            self._semantic_cache_service.invalidate(repo_id)
            self._status_store.update(repo_id, status="pending", error=None)
            return repo_id, "pending", True, normalized_url

        return repo_id, record.status, False, normalized_url

    def enqueue_ingestion(self, repo_id: str, source_url: str) -> None:
        """Hand the repo off to a Celery worker and return immediately.

        The actual clone/chunk/embed/store pipeline runs in
        worker.tasks.ingest_repo_task, in a separate process, reporting
        progress through RedisStatusStore.

        Args:
            repo_id: Repo to ingest.
            source_url: Normalized repository URL to clone.
        """
        ingest_repo_task.delay(repo_id, source_url)

    def get_status(self, repo_id: str) -> RepoStatus:
        """Return a repo's status record.

        Args:
            repo_id: Repo to look up.

        Returns:
            The status record.

        Raises:
            RepoNotFoundError: If the repo_id is unknown.
        """
        record = self._status_store.get(repo_id)
        if record is None:
            raise RepoNotFoundError(repo_id)
        return record

    async def chat(self, repo_id: str, message: str, session_id: str | None) -> tuple[str, str, bool]:
        """Answer a question about a ready repo, consulting the cache first.

        On a cache hit the cached Q&A is appended to the ADK session so
        later turns still see it; on a miss the agent runs and its answer
        is cached.

        Args:
            repo_id: Repo to chat about.
            message: The user's question.
            session_id: Existing ADK session to continue, or None to create
                one (an unknown id also creates a new session).

        Returns:
            (session_id, answer, cached).

        Raises:
            RepoNotFoundError: If the repo_id is unknown.
            RepoNotReadyError: If the repo is not "ready".
            AgentResponseError: If the agent produced no final answer.
        """
        record = self.get_status(repo_id)
        if record.status != "ready":
            raise RepoNotReadyError(repo_id, record.status)

        runner = self._agent_registry.get_runner(repo_id)
        user_id = f"repo-{repo_id}"

        session = None
        if session_id:
            session = await runner.session_service.get_session(
                app_name=runner.app_name, user_id=user_id, session_id=session_id
            )

        if session is None:
            session = await runner.session_service.create_session(app_name=runner.app_name, user_id=user_id)

        # SemanticCacheService.lookup/store are plain sync calls (Redis
        # round trips, plus an OpenAI embedding call on an exact-cache miss)
        # — run them off the event loop so they don't block every other
        # concurrent request the way the retrieval tool call already does
        # (CLAUDE.md §12.1).
        cache_result = await asyncio.to_thread(self._semantic_cache_service.lookup, repo_id, message)
        if cache_result.answer is not None:
            await asyncio.to_thread(
                log_cache_event, repo_id, cache_result.outcome, message, cache_result.nearest_similarity
            )
            await self._append_cached_turn_to_session(runner, session, message, cache_result.answer)
            return session.id, cache_result.answer, True
        await asyncio.to_thread(
            log_cache_event, repo_id, cache_result.outcome, message, cache_result.nearest_similarity
        )

        answer = ""
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            if event.is_final_response() and event.content and event.content.parts:
                answer = "".join(part.text for part in event.content.parts if getattr(part, "text", None))

        if not answer:
            raise AgentResponseError("Agent did not produce a response")

        await asyncio.to_thread(self._semantic_cache_service.store, repo_id, message, answer)

        return session.id, answer, False

    async def _append_cached_turn_to_session(
        self, runner: InMemoryRunner, session: Session, message: str, answer: str
    ) -> None:
        """Record a cache-served turn in the ADK session history.

        Appends a user event and an agent-authored model event under one
        fresh invocation id, mirroring what a real turn leaves behind.

        Args:
            runner: The repo's runner (provides the session service and
                agent name).
            session: Session to append to.
            message: The user's question.
            answer: The cached answer.
        """
        # A cache hit skips runner.run_async, so nothing else writes this
        # turn into session history — later turns would have no memory of it.
        invocation_id = new_invocation_context_id()
        user_event = Event(
            invocation_id=invocation_id,
            author="user",
            content=types.Content(role="user", parts=[types.Part(text=message)]),
        )
        answer_event = Event(
            invocation_id=invocation_id,
            author=runner.agent.name,
            content=types.Content(role="model", parts=[types.Part(text=answer)]),
        )
        await runner.session_service.append_event(session=session, event=user_event)
        await runner.session_service.append_event(session=session, event=answer_event)


_rag_workflow = RagWorkflowFacade(_status_store, _agent_registry, _semantic_cache_service)


def get_rag_workflow() -> RagWorkflowFacade:
    """FastAPI dependency returning the process-wide workflow facade.

    Returns:
        The singleton RagWorkflowFacade.
    """
    return _rag_workflow
