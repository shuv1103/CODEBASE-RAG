"""Workflow facade, per-repo agent registry and the API's singletons."""

import asyncio
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable

from google.adk.agents.invocation_context import new_invocation_context_id
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types

from caching.query_normalizer import normalize_query
from caching.semantic_cache_repository import LangCacheRepository
from caching.semantic_cache_service import SemanticCacheService
from codebase_rag_agent.agent import build_agent
from codebase_rag_agent.tools import CodebaseRetrievalTool
from codebase_rag_agent.tracing import log_cache_event
from retrieval.retrieval_service import RetrievalService
from storage.redis_status_store import RedisStatusStore
from storage.status_store import RepoStatus, StatusStore
from utils.config import settings
from utils.repo_id import compute_repo_id, normalize_github_url
from worker.tasks import ingest_repo_task

from .concurrency import LLMConcurrencyLimiter
from .exceptions import (
    AgentResponseError,
    InvalidRepoUrlError,
    RepoNotFoundError,
    RepoNotReadyError,
)

logger = logging.getLogger(__name__)

_ABANDONED_ERROR = (
    "Ingestion was interrupted (worker restarted or crashed). "
    "Submit the repository again to retry."
)
_ENQUEUE_FAILED_ERROR = (
    "Ingestion could not be queued. Submit the repository again to retry."
)

_status_store: StatusStore = RedisStatusStore()

# Listing a runner's sessions copies each one, so idle sessions are swept at
# most this often per repo rather than on every chat turn.
_SESSION_SWEEP_INTERVAL_SECONDS = 60


def is_abandoned(record: RepoStatus) -> bool:
    """Check whether an unfinished ingestion is past its staleness threshold.

    Args:
        record: The repo's status record.

    Returns:
        True if no live or queued task can still own the record.
    """
    pending_threshold = settings.ingestion_pending_stale_after_seconds
    return record.is_abandoned(
        settings.ingestion_stale_after_seconds,
        pending_stale_after_seconds=pending_threshold,
    )


def build_runner(repo_id: str) -> InMemoryRunner:
    """Build an ADK runner whose retrieval is scoped to one repo's chunks.

    Args:
        repo_id: Repo whose chunks the agent's retrieval is scoped to.

    Returns:
        A new InMemoryRunner.
    """
    retrieval_service = RetrievalService(repo_id=repo_id)
    tool = CodebaseRetrievalTool(retrieval_service=retrieval_service)
    return InMemoryRunner(agent=build_agent(tool))


class AgentRegistry:
    """Bounded cache of one ADK runner per repo, built on first use.

    Each runner's retrieval is scoped to its repo, and reusing the runner
    keeps that repo's chat history across turns. Runners and sessions live
    in process memory, so only the max_repos most recently used runners are
    kept and sessions idle past session_ttl_seconds are deleted. A deleted
    session's id simply starts a new session.

    Args:
        max_repos: Most runners kept at once.
        session_ttl_seconds: Idle time after which a session is deleted.
        build_runner: Factory that builds a runner for a repo_id.
    """

    def __init__(
        self,
        max_repos: int = settings.agent_registry_max_repos,
        session_ttl_seconds: int = settings.chat_session_ttl_seconds,
        build_runner: Callable[[str], InMemoryRunner] = build_runner,
    ) -> None:
        self._lock = threading.Lock()
        self._runners: OrderedDict[str, InMemoryRunner] = OrderedDict()
        self._last_swept: dict[str, float] = {}
        self._max_repos = max_repos
        self._session_ttl_seconds = session_ttl_seconds
        self._build_runner = build_runner

    def get_runner(self, repo_id: str) -> InMemoryRunner:
        """Return the repo's runner, building it on first use (thread-safe).

        Mark the repo as most recently used and evict the least recently
        used runner once more than max_repos are held.

        Args:
            repo_id: Repo whose chunks the agent's retrieval is scoped to.

        Returns:
            The cached InMemoryRunner for the repo.
        """
        with self._lock:
            runner = self._runners.get(repo_id)
            if runner is not None:
                self._runners.move_to_end(repo_id)
                return runner

        # Built outside the lock: building creates network clients, and
        # holding the lock would stall every other repo's lookup.
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
        """Delete the repo's sessions that have been idle past the TTL.

        Skip the sweep if this repo was swept within the last
        _SESSION_SWEEP_INTERVAL_SECONDS.

        Args:
            repo_id: Repo the runner belongs to.
            runner: The repo's runner.
            user_id: ADK user id the repo's sessions are stored under.
        """
        now = time.time()
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


_agent_registry = AgentRegistry()
_semantic_cache_service = SemanticCacheService(LangCacheRepository())


class RagWorkflowFacade:
    """Single entry point the API routes use for ingest, status and chat.

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
        self._llm_concurrency = LLMConcurrencyLimiter()

    def start_ingestion(
        self,
        url: str,
        admit_new_work: Callable[[], None] | None = None,
        force: bool = False,
    ) -> tuple[str, str, bool, str, str | None]:
        """Validate the URL and create or reuse the repo's status record.

        Dedup is one atomic create_if_absent() call, so two concurrent
        requests for the same new repo can't both enqueue it. A failed or
        abandoned repo is reset to "pending", with its cached answers
        dropped, so it can be retried; the reset is a compare-and-set on the
        attempt id, so of two concurrent retries only one starts.

        Args:
            url: User-supplied GitHub repository URL.
            admit_new_work: Called only when this request would start an
                ingestion; raises to refuse it (e.g. the global cap).
            force: Re-ingest a "ready" repo too (an operator action, e.g.
                after a change to stored chunk data).

        Returns:
            (repo_id, status, is_new, normalized_url, attempt_id).
            is_new=False means the repo is already in flight or ready, so
            the caller must not enqueue another ingestion. attempt_id is the
            attempt to pass to enqueue_ingestion when is_new is True.

        Raises:
            InvalidRepoUrlError: If the URL is not a github.com repo URL.
        """
        try:
            normalized_url = normalize_github_url(url)
        except ValueError as exc:
            raise InvalidRepoUrlError(str(exc)) from exc

        repo_id = compute_repo_id(normalized_url)

        def needs_work(record: RepoStatus | None) -> bool:
            return (
                record is None
                or record.status == "failed"
                or (force and record.status == "ready")
                or is_abandoned(record)
            )

        if admit_new_work is not None and needs_work(
            self._status_store.get(repo_id)
        ):
            admit_new_work()

        record, created = self._status_store.create_if_absent(
            repo_id, normalized_url
        )

        if created:
            return repo_id, "pending", True, normalized_url, record.attempt_id

        if needs_work(record):
            new_attempt_id = uuid.uuid4().hex
            # The new attempt_id turns any still-queued task for the old
            # attempt into a no-op.
            if self._status_store.update_if_attempt(
                repo_id,
                record.attempt_id,
                status="pending",
                error=None,
                attempt_id=new_attempt_id,
            ):
                # A retry re-ingests from scratch, so answers cached from
                # the old data must go.
                self._semantic_cache_service.invalidate(repo_id)
                return repo_id, "pending", True, normalized_url, new_attempt_id
            # A concurrent request reset the record first; report its state.
            record = self._status_store.get(repo_id) or record

        return repo_id, record.status, False, normalized_url, record.attempt_id

    def enqueue_ingestion(
        self, repo_id: str, source_url: str, attempt_id: str | None
    ) -> None:
        """Queue the repo for ingestion by a Celery worker.

        The task carries the attempt_id, so a superseded or duplicate
        delivery skips itself.

        Args:
            repo_id: Repo to ingest.
            source_url: Normalized repository URL to clone.
            attempt_id: Attempt returned by start_ingestion.

        Raises:
            Exception: Whatever the broker raised, after the record is failed.
        """
        try:
            ingest_repo_task.delay(repo_id, source_url, attempt_id)
        except Exception:
            # Left "pending", the record would block re-submits for a day.
            self._status_store.update_if_attempt(
                repo_id,
                attempt_id,
                status="failed",
                error=_ENQUEUE_FAILED_ERROR,
            )
            raise

    def get_status(self, repo_id: str) -> RepoStatus:
        """Return a repo's status record, failing an abandoned ingestion.

        A record still in flight long after any task could be running (its
        worker was killed, or the task was lost on a restart) is marked
        failed, so the frontend stops polling and a re-submit retries it.

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
        if is_abandoned(record):
            logger.warning(
                "Ingestion of repo_id=%s abandoned at status=%s; "
                "marking failed",
                repo_id,
                record.status,
            )
            # Conditional, so a retry that started meanwhile is not failed.
            # A fresh attempt_id cuts off the old run if it is in fact still
            # alive: its next status write no longer matches.
            failed_fields = {
                "status": "failed",
                "error": _ABANDONED_ERROR,
                "attempt_id": uuid.uuid4().hex,
            }
            if self._status_store.update_if_attempt(
                repo_id, record.attempt_id, **failed_fields
            ):
                record = record.model_copy(update=failed_fields)
            else:
                record = self._status_store.get(repo_id) or record
        return record

    async def chat(
        self, repo_id: str, message: str, session_id: str | None
    ) -> tuple[str, str, str | None]:
        """Answer a question about a ready repo, consulting the cache first.

        On a hit, append the cached turn to the session so later turns see
        it; on a miss, run the agent and cache its answer. Every turn uses
        the cache; a follow-up matches only other follow-ups.

        Args:
            repo_id: Repo to chat about.
            message: The user's question.
            session_id: Existing ADK session to continue, or None to create
                one (an unknown id also creates a new session).

        Returns:
            (session_id, answer, cache_hit): cache_hit is "exact" or
            "semantic" when the cache answered, None when the agent did.

        Raises:
            RepoNotFoundError: If the repo_id is unknown.
            RepoNotReadyError: If the repo is not "ready".
            AgentResponseError: If the agent produced no final answer.
            LLMBusyError: If no LLM slot freed up in time.
        """
        # Sync Redis call: on the event loop it would stall every request.
        record = await asyncio.to_thread(self.get_status, repo_id)
        if record.status != "ready":
            raise RepoNotReadyError(repo_id, record.status)

        # Building a runner creates OpenAI and Qdrant clients; keep that off
        # the event loop.
        runner = await asyncio.to_thread(
            self._agent_registry.get_runner, repo_id
        )
        user_id = f"repo-{repo_id}"
        await self._agent_registry.expire_idle_sessions(
            repo_id, runner, user_id
        )

        session = None
        if session_id:
            session = await runner.session_service.get_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )

        if session is None:
            session = await runner.session_service.create_session(
                app_name=runner.app_name, user_id=user_id
            )

        # The cache is shared by all users, and a follow-up's answer can
        # depend on the conversation; the cache service keys a follow-up by
        # this previous question too, apart from first-turn entries. Earlier
        # asks of this same question are skipped, so a repeat looks up the
        # key its first ask was stored under.
        # ponytail: one turn of context; key on more turns if answers drift.
        same_question = normalize_query(message).casefold()
        previous_question = next(
            (
                event.content.parts[0].text
                for event in reversed(session.events)
                if event.author == "user"
                and event.content
                and event.content.parts
                and event.content.parts[0].text
                and normalize_query(event.content.parts[0].text).casefold()
                != same_question
            ),
            None,
        )
        turn = "first" if previous_question is None else "followup"

        # Cache calls are sync network I/O; keep them off the event loop.
        cache_result = await asyncio.to_thread(
            self._semantic_cache_service.lookup,
            repo_id,
            message,
            previous_question,
        )
        await asyncio.to_thread(
            log_cache_event,
            repo_id,
            cache_result.outcome,
            message,
            cache_result.nearest_similarity,
        )
        if cache_result.answer is not None:
            await self.append_cached_turn_to_session(
                runner, session, message, cache_result.answer
            )
            logger.info(
                "Chat turn served from cache",
                extra={
                    "repo_id": repo_id,
                    "turn": turn,
                    "outcome": cache_result.outcome,
                },
            )
            return (
                session.id,
                cache_result.answer,
                cache_result.outcome.removesuffix("_hit"),
            )

        # Only cache misses reach the LLM, so only they take a concurrency slot.
        answer = ""
        started = time.perf_counter()
        async with self._llm_concurrency.slot():
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=message)]
                ),
            ):
                if (
                    event.is_final_response()
                    and event.content
                    and event.content.parts
                ):
                    answer = "".join(
                        part.text
                        for part in event.content.parts
                        if getattr(part, "text", None)
                    )

        # Includes waiting for a concurrency slot, as the user experiences it.
        agent_ms = round((time.perf_counter() - started) * 1000)
        if not answer:
            raise AgentResponseError("Agent did not produce a response")

        await asyncio.to_thread(
            self._semantic_cache_service.store,
            repo_id,
            message,
            answer,
            previous_question,
        )
        logger.info(
            "Chat turn answered by agent",
            extra={
                "repo_id": repo_id,
                "turn": turn,
                "outcome": cache_result.outcome,
                "agent_ms": agent_ms,
            },
        )

        return session.id, answer, None

    async def append_cached_turn_to_session(
        self,
        runner: InMemoryRunner,
        session: Session,
        message: str,
        answer: str,
    ) -> None:
        """Record a cache-served turn in the ADK session history.

        Append a user event and an agent-authored model event under one
        fresh invocation id, mirroring what a real turn leaves behind.

        Args:
            runner: The repo's runner (provides the session service and
                agent name).
            session: Session to append to.
            message: The user's question.
            answer: The cached answer.
        """
        # A cache hit skips run_async, which normally records the turn.
        invocation_id = new_invocation_context_id()
        user_event = Event(
            invocation_id=invocation_id,
            author="user",
            content=types.Content(
                role="user", parts=[types.Part(text=message)]
            ),
        )
        answer_event = Event(
            invocation_id=invocation_id,
            author=runner.agent.name,
            content=types.Content(
                role="model", parts=[types.Part(text=answer)]
            ),
        )
        await runner.session_service.append_event(
            session=session, event=user_event
        )
        await runner.session_service.append_event(
            session=session, event=answer_event
        )


_rag_workflow = RagWorkflowFacade(
    _status_store, _agent_registry, _semantic_cache_service
)


def get_rag_workflow() -> RagWorkflowFacade:
    """Return the process-wide workflow facade (a FastAPI dependency).

    Returns:
        The singleton RagWorkflowFacade.
    """
    return _rag_workflow
