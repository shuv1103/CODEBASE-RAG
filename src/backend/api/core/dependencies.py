"""Workflow facade behind the API routes, and the API's singletons."""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from typing import NamedTuple

from google.adk.agents.invocation_context import new_invocation_context_id
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.adk.sessions import Session
from google.genai import types

from api.core.agent_runtime import AgentRegistry, LLMConcurrencyLimiter
from api.core.exceptions import (
    AgentResponseError,
    InvalidRepoUrlError,
    RepoNotFoundError,
    RepoNotReadyError,
)
from caching.semantic_cache_repository import (
    CacheStrategy,
    LangCacheRepository,
)
from caching.semantic_cache_service import (
    CacheLookupResult,
    SemanticCacheService,
    normalize_query,
)
from codebase_rag_agent.tracing import log_cache_event
from storage.redis_status_store import RedisStatusStore
from storage.status_store import RepoStatus, RepoStatusValue, StatusStore
from utils.config import settings
from utils.repo_id import compute_repo_id, normalize_github_url
from worker.tasks import ingest_repo_task

logger = logging.getLogger(__name__)

_ABANDONED_ERROR = (
    "Ingestion was interrupted (worker restarted or crashed). "
    "Submit the repository again to retry."
)
_ENQUEUE_FAILED_ERROR = (
    "Ingestion could not be queued. Submit the repository again to retry."
)

_status_store: StatusStore = RedisStatusStore()


class IngestionStart(NamedTuple):
    """Result of RagWorkflowFacade.start_ingestion.

    Attributes:
        repo_id: 12-hex id derived from the normalized URL.
        status: The repo's status after the call.
        is_new: True if the caller must enqueue an ingestion.
        normalized_url: Canonical https://github.com/<org>/<repo> URL.
        attempt_id: Attempt id to pass to enqueue_ingestion.
    """

    repo_id: str
    status: RepoStatusValue
    is_new: bool
    normalized_url: str
    attempt_id: str | None


class ChatTurn(NamedTuple):
    """Result of RagWorkflowFacade.chat.

    Attributes:
        session_id: ADK session to send on the next turn.
        answer: The answer text.
        cache_hit: "exact" or "semantic" when the cache answered, else None.
    """

    session_id: str
    answer: str
    cache_hit: CacheStrategy | None


def is_abandoned(record: RepoStatus) -> bool:
    """Check whether no live or queued task can still own an in-flight record.

    Args:
        record: The repo's status record.

    Returns:
        True if the record is in flight and past its staleness threshold.
    """
    return record.is_abandoned(
        settings.ingestion_stale_after_seconds,
        pending_stale_after_seconds=(
            settings.ingestion_pending_stale_after_seconds
        ),
    )


def event_text(event: Event) -> str:
    """Return the text of an ADK event.

    Args:
        event: A session or run event.

    Returns:
        Its text parts joined, or "" when it has no text.
    """
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text or "" for part in event.content.parts)


def previous_user_question(session: Session, message: str) -> str | None:
    """Return the latest earlier user question that differs from message.

    Repeats of the same question are skipped, so a repeated follow-up finds
    the cache key its first ask was stored under.

    Args:
        session: The chat's ADK session, before message is added.
        message: The question being asked now.

    Returns:
        The previous question, or None on a session's first turn.
    """
    # ponytail: one turn of context; key on more turns if answers drift.
    same_question = normalize_query(message)
    for event in reversed(session.events):
        if event.author != "user":
            continue
        text = event_text(event)
        if text and normalize_query(text) != same_question:
            return text
    return None


def session_answers(session: Session) -> set[str]:
    """Return every non-empty answer this session already received.

    Args:
        session: The chat's ADK session.

    Returns:
        The texts of all non-user events.
    """
    return {
        text
        for event in session.events
        if event.author != "user" and (text := event_text(event))
    }


def latest_answer(session: Session) -> str:
    """Return the last answer this session received.

    Args:
        session: The chat's ADK session.

    Returns:
        The text of the latest non-user event, or "" if there is none.
    """
    for event in reversed(session.events):
        if event.author != "user" and (text := event_text(event)):
            return text
    return ""


_agent_registry = AgentRegistry()
_semantic_cache_service = SemanticCacheService(LangCacheRepository())


class RagWorkflowFacade:
    """Single entry point the API routes use for ingest, status and chat.

    Args:
        status_store: Store of repo ingestion status records.
        agent_registry: Per-repo cache of ADK runners.
        semantic_cache_service: Answer cache consulted before the agent.
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
    ) -> IngestionStart:
        """Create the repo's status record, or reset a failed one for retry.

        Dedup and the retry reset are atomic, so concurrent requests for one
        repo start at most one ingestion. A reset also drops the repo's cached
        answers.

        Args:
            url: User-supplied GitHub repository URL.
            admit_new_work: Called only when this request would start work;
                raises to refuse it (the global ingestion cap).
            force: Also re-ingest a "ready" repo (operator action).

        Returns:
            IngestionStart; enqueue an ingestion only when is_new is True.

        Raises:
            InvalidRepoUrlError: If url is not a github.com repository URL.
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
            return IngestionStart(
                repo_id, "pending", True, normalized_url, record.attempt_id
            )

        if needs_work(record):
            # A new attempt_id turns any still-queued old task into a no-op.
            new_attempt_id = uuid.uuid4().hex
            if self._status_store.update_if_attempt(
                repo_id,
                record.attempt_id,
                status="pending",
                error=None,
                attempt_id=new_attempt_id,
            ):
                # Answers cached from the old index must not survive.
                self._semantic_cache_service.invalidate(repo_id)
                return IngestionStart(
                    repo_id, "pending", True, normalized_url, new_attempt_id
                )
            # A concurrent request reset the record first; report its state.
            record = self._status_store.get(repo_id) or record

        return IngestionStart(
            repo_id, record.status, False, normalized_url, record.attempt_id
        )

    def enqueue_ingestion(
        self, repo_id: str, source_url: str, attempt_id: str | None
    ) -> None:
        """Queue the repo's ingestion task for a Celery worker.

        Args:
            repo_id: Repo to ingest.
            source_url: Normalized repository URL to clone.
            attempt_id: Attempt id from start_ingestion; a stale task skips
                itself.

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
        """Return a repo's status record, failing an abandoned ingestion first.

        Args:
            repo_id: Repo to look up.

        Returns:
            The current RepoStatus.

        Raises:
            RepoNotFoundError: If repo_id has no record.
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
            # Conditional, so a retry that started meanwhile survives; the
            # new attempt_id stops the old run if it is somehow still alive.
            new_attempt_id = uuid.uuid4().hex
            if self._status_store.update_if_attempt(
                repo_id,
                record.attempt_id,
                status="failed",
                error=_ABANDONED_ERROR,
                attempt_id=new_attempt_id,
            ):
                record = record.model_copy(
                    update={
                        "status": "failed",
                        "error": _ABANDONED_ERROR,
                        "attempt_id": new_attempt_id,
                    }
                )
            else:
                record = self._status_store.get(repo_id) or record
        return record

    async def chat(
        self, repo_id: str, message: str, session_id: str | None
    ) -> ChatTurn:
        """Answer a question about a ready repo, from the cache when possible.

        A cache hit is appended to the session; a miss runs the agent and
        caches its answer.

        Args:
            repo_id: Repo to chat about; must be "ready".
            message: The user's question.
            session_id: Session to continue; None or an unknown id starts one.

        Returns:
            ChatTurn with the session id, the answer and the cache tier used.

        Raises:
            RepoNotFoundError: If repo_id has no record.
            RepoNotReadyError: If the repo has never been "ready".
            AgentResponseError: If the agent produced no final answer.
            LLMBusyError: If no LLM slot freed up in time.
        """
        # Redis and client setup are blocking; keep them off the event loop.
        record = await asyncio.to_thread(self.get_status, repo_id)
        # A re-ingest keeps the old index until it succeeds (delete_stale), so
        # a repo that was ready once stays chattable while one runs or fails.
        if record.status != "ready" and record.ready_at is None:
            raise RepoNotReadyError(repo_id, record.status)

        runner = await asyncio.to_thread(
            self._agent_registry.get_runner, repo_id
        )
        user_id = f"repo-{repo_id}"
        await self._agent_registry.expire_idle_sessions(
            repo_id, runner, user_id
        )
        session = await self.get_or_create_session(runner, user_id, session_id)

        previous_question = previous_user_question(session, message)
        previous_answer = latest_answer(session)
        turn = "first" if previous_question is None else "followup"

        cache_result = await self.lookup_cached_answer(
            repo_id, message, session, previous_question, previous_answer
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
            return ChatTurn(
                session.id,
                cache_result.answer,
                cache_result.outcome.removesuffix("_hit"),
            )

        started = time.perf_counter()
        answer = await self.run_agent(runner, user_id, session.id, message)
        # Includes waiting for a concurrency slot, as the user experiences it.
        agent_ms = round((time.perf_counter() - started) * 1000)

        await asyncio.to_thread(
            self._semantic_cache_service.store,
            repo_id,
            message,
            answer,
            previous_question,
            previous_answer=previous_answer,
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

        return ChatTurn(session.id, answer, None)

    @staticmethod
    async def get_or_create_session(
        runner: InMemoryRunner, user_id: str, session_id: str | None
    ) -> Session:
        """Return the session to continue, or a new one.

        Args:
            runner: The repo's runner.
            user_id: ADK user id the repo's sessions are stored under.
            session_id: Session to continue; None or an unknown id starts one.

        Returns:
            The existing or newly created Session.
        """
        session = None
        if session_id:
            session = await runner.session_service.get_session(
                app_name=runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )
        if session is None:
            if session_id:
                # Evicted runner or expired session: this turn loses context.
                logger.warning(
                    "Chat session %r not found; starting a new one",
                    session_id,
                    extra={"user_id": user_id},
                )
            session = await runner.session_service.create_session(
                app_name=runner.app_name, user_id=user_id
            )
        return session

    async def lookup_cached_answer(
        self,
        repo_id: str,
        message: str,
        session: Session,
        previous_question: str | None,
        previous_answer: str,
    ) -> CacheLookupResult:
        """Look the turn up in the answer cache and trace the outcome.

        A follow-up that misses is retried as a first turn; that answer is used
        only if this session already received it (a rephrased question).

        Args:
            repo_id: Repo the chat is about.
            message: The user's question.
            session: The chat's session, before message is added.
            previous_question: The previous question, or None on a first turn.
            previous_answer: The answer the session last received.

        Returns:
            CacheLookupResult; its answer is None on a miss.
        """
        cache_result = await asyncio.to_thread(
            self._semantic_cache_service.lookup,
            repo_id,
            message,
            previous_question,
            previous_answer=previous_answer,
        )
        if cache_result.answer is None and previous_question is not None:
            rephrase = await asyncio.to_thread(
                self._semantic_cache_service.lookup, repo_id, message
            )
            if rephrase.answer is not None and rephrase.answer in (
                session_answers(session)
            ):
                cache_result = rephrase
        await asyncio.to_thread(
            log_cache_event,
            repo_id,
            cache_result.outcome,
            message,
            cache_result.nearest_similarity,
        )
        return cache_result

    async def run_agent(
        self,
        runner: InMemoryRunner,
        user_id: str,
        session_id: str,
        message: str,
    ) -> str:
        """Run one agent turn inside an LLM concurrency slot.

        Args:
            runner: The repo's runner.
            user_id: ADK user id the session is stored under.
            session_id: Session the turn is appended to.
            message: The user's question.

        Returns:
            The agent's final answer text.

        Raises:
            AgentResponseError: If the agent produced no final answer.
            LLMBusyError: If no LLM slot freed up in time.
        """
        answer = ""
        async with self._llm_concurrency.slot():
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=message)]
                ),
            ):
                if event.is_final_response() and (text := event_text(event)):
                    answer = text
        if not answer:
            raise AgentResponseError("Agent did not produce a response")
        return answer

    async def append_cached_turn_to_session(
        self,
        runner: InMemoryRunner,
        session: Session,
        message: str,
        answer: str,
    ) -> None:
        """Record a cache-served turn in the session, as an agent turn would.

        Args:
            runner: The repo's runner (session service and agent name).
            session: Session to append the user and answer events to.
            message: The user's question.
            answer: The cached answer.
        """
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
    """Return the process-wide workflow facade (a FastAPI dependency)."""
    return _rag_workflow
