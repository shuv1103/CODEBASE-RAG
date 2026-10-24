"""Shared OpenAI request budget and exponential backoff on 429s."""

import asyncio
import functools
import logging
import math
import random
import time
from collections.abc import Callable
from typing import TypeVar

import redis

from .config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OpenAIRequestBudget:
    """Requests-per-minute budget for OpenAI, shared by every process via Redis.

    The API and every Celery worker draw on one OpenAI account limit, so a
    per-process limit can't hold the total once the app scales out. Each
    call takes a slot from a Redis counter for the current second, capped
    at ceil(requests_per_minute / 60), which smooths out bursts.

    Fails open: a disabled budget never waits, and while Redis is
    unreachable the budget is skipped for _REDIS_RETRY_SECONDS.

    Args:
        requests_per_minute: Shared OpenAI request budget; <= 0 disables it.
        redis_url: Redis holding the shared counters.
        max_wait_seconds: Longest acquire() waits for a slot before letting
            the call through anyway (backoff then handles any 429).

    Attributes:
        enabled: Whether the budget is enforced.
        per_second: Slots handed out per one-second window.
    """

    _KEY_PREFIX = "codebase-rag:openai-budget:"
    _REDIS_RETRY_SECONDS = 30.0
    # Each slot is one sub-millisecond INCR, so two connections serve many
    # concurrent callers; a caller waits this long for one before failing
    # open.
    _MAX_CONNECTIONS = 2
    _POOL_WAIT_SECONDS = 2

    def __init__(
        self,
        requests_per_minute: int,
        redis_url: str,
        max_wait_seconds: float = 30.0,
    ) -> None:
        self.enabled = requests_per_minute > 0
        self.per_second = (
            max(1, math.ceil(requests_per_minute / 60)) if self.enabled else 0
        )
        self._max_wait_seconds = max_wait_seconds
        # Capped like every other pool, so the budget can't push a process
        # past its share of Redis Cloud's 30-client limit. Building the pool
        # doesn't connect, so a budget never touches Redis until used.
        pool = redis.BlockingConnectionPool.from_url(
            redis_url,
            max_connections=self._MAX_CONNECTIONS,
            timeout=self._POOL_WAIT_SECONDS,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        self._client = redis.Redis(connection_pool=pool)
        self._redis_down_until = 0.0

    def take_slot(self) -> bool:
        """Count one request against the current one-second window.

        Returns:
            True if the window still had room.

        Raises:
            redis.RedisError: If Redis is unreachable.
        """
        key = f"{self._KEY_PREFIX}{int(time.time())}"
        pipe = self._client.pipeline()
        pipe.incr(key)
        pipe.expire(key, 2)
        count, _ = pipe.execute()
        return int(count) <= self.per_second

    def skips(self) -> bool:
        """Check whether acquire() returns at once (budget off/Redis down)."""
        return not self.enabled or time.monotonic() < self._redis_down_until

    def poll(self) -> float | None:
        """Try once to take a slot.

        Returns:
            None if the caller may proceed (slot taken, or Redis unreachable);
            otherwise the seconds to sleep before trying again.
        """
        try:
            if self.take_slot():
                return None
        except redis.RedisError as exc:
            logger.warning(
                "OpenAI budget: Redis unreachable (%s); skipping it for %.0fs",
                exc,
                self._REDIS_RETRY_SECONDS,
            )
            self._redis_down_until = (
                time.monotonic() + self._REDIS_RETRY_SECONDS
            )
            return None
        # Window full: sleep into the next one, with jitter so waiting
        # processes don't all retry on the same millisecond.
        return (1.0 - (time.time() % 1.0)) + random.uniform(0, 0.25)

    def past_deadline(self, deadline: float, sleep_for: float) -> bool:
        """Check whether sleeping sleep_for overruns deadline; log if so."""
        if time.monotonic() + sleep_for > deadline:
            logger.warning(
                "OpenAI budget: no slot within %.0fs; calling anyway",
                self._max_wait_seconds,
            )
            return True
        return False

    def acquire(self) -> bool:
        """Block until a slot is free, or max_wait_seconds pass.

        Returns:
            True if a slot was taken (or the budget is disabled/unreachable);
            False if the wait timed out — the caller proceeds anyway.
        """
        if self.skips():
            return True
        deadline = time.monotonic() + self._max_wait_seconds
        while (sleep_for := self.poll()) is not None:
            if self.past_deadline(deadline, sleep_for):
                return False
            time.sleep(sleep_for)
        return True

    async def acquire_async(self) -> bool:
        """Wait for a slot like acquire(), without blocking the event loop.

        Only the Redis round trip runs in a thread, so a full window doesn't
        tie up the default executor's threads.

        Returns:
            Same as acquire().
        """
        if self.skips():
            return True
        deadline = time.monotonic() + self._max_wait_seconds
        while (sleep_for := await asyncio.to_thread(self.poll)) is not None:
            if self.past_deadline(deadline, sleep_for):
                return False
            await asyncio.sleep(sleep_for)
        return True


@functools.cache
def get_openai_budget() -> OpenAIRequestBudget:
    """Return the process-wide OpenAI budget, built from settings on first use.

    OPENAI_RPM_LIMIT sets the shared budget (0/unset = disabled);
    OPENAI_BUDGET_MAX_WAIT_SECONDS caps how long a caller waits for a slot.

    Returns:
        The shared OpenAIRequestBudget.
    """
    return OpenAIRequestBudget(
        requests_per_minute=settings.openai_rpm_limit,
        redis_url=settings.redis_url,
        max_wait_seconds=settings.openai_budget_max_wait_seconds,
    )


class ExponentialBackoff:
    """Retry handler with exponential backoff for OpenAI rate-limit errors.

    Each attempt first takes a slot from the shared OpenAI budget. A 429 is
    retried up to max_retries attempts in total, each wait being the larger
    of OpenAI's Retry-After hint and 2 ** attempt plus jitter, capped at
    max_wait_seconds. An exhausted quota (insufficient_quota) is a billing
    problem retries can't fix, so it is re-raised at once like any other
    error.

    Args:
        max_retries: Maximum total attempts. Defaults to BACKOFF_MAX_RETRIES.
        max_wait_seconds: Longest single wait between attempts.
        budget: Shared OpenAI request budget. Defaults to the process-wide one.

    Attributes:
        max_retries: Maximum total attempts.

    Example:
        >>> backoff = ExponentialBackoff(max_retries=3)
        >>> backoff.execute(lambda: 42)
        42
    """

    def __init__(
        self,
        max_retries: int = settings.backoff_max_retries,
        max_wait_seconds: float = settings.backoff_max_wait_seconds,
        budget: OpenAIRequestBudget | None = None,
    ) -> None:
        self.max_retries = max_retries
        self._max_wait_seconds = max_wait_seconds
        self._budget = budget

    def execute(self, api_func: Callable[[], T]) -> T:
        """Call api_func, retrying with backoff on rate-limit errors.

        Args:
            api_func: Zero-argument callable to invoke.

        Returns:
            The value returned by api_func.

        Raises:
            Exception: Any non-retryable error immediately, or the last
                rate-limit error once attempts are exhausted.
        """
        budget = self._budget or get_openai_budget()
        for attempt in range(self.max_retries):
            budget.acquire()
            try:
                return api_func()
            except Exception as e:
                if not self.is_rate_limit_error(e):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.wait_seconds(e, attempt)
                logger.warning(
                    "OpenAI rate limit (attempt %d/%d); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    wait_time,
                )
                time.sleep(wait_time)

    def wait_seconds(self, e: Exception, attempt: int) -> float:
        """Return the seconds to wait before the next attempt.

        Args:
            e: The rate-limit error.
            attempt: Zero-based index of the attempt that failed.

        Returns:
            max(Retry-After, exponential + jitter), capped at max_wait_seconds.
        """
        exponential = (2 ** attempt) + random.uniform(0, 1)
        retry_after = self.retry_after_seconds(e) or 0.0
        return min(max(exponential, retry_after), self._max_wait_seconds)

    @staticmethod
    def retry_after_seconds(e: Exception) -> float | None:
        """Read OpenAI's retry hint from the error's response headers.

        Args:
            e: The raised exception.

        Returns:
            Seconds to wait, or None if the error carries no usable hint.
        """
        headers = getattr(getattr(e, "response", None), "headers", None)
        if not headers:
            return None
        try:
            if headers.get("retry-after-ms"):
                return float(headers["retry-after-ms"]) / 1000
            if headers.get("retry-after"):
                return float(headers["retry-after"])
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def is_rate_limit_error(e: Exception) -> bool:
        """Detect a retryable rate-limit error.

        Check the HTTP status first (openai and litellm RateLimitError set
        status_code=429), then the message, for wrappers that keep only the
        text. A 429 for insufficient_quota is not retryable.

        Args:
            e: The raised exception.

        Returns:
            True for a retryable 429 / RESOURCE_EXHAUSTED error.
        """
        msg = str(e).upper()
        if (
            getattr(e, "code", None) == "insufficient_quota"
            or "INSUFFICIENT_QUOTA" in msg
        ):
            return False
        if getattr(e, "status_code", None) == 429:
            return True
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg
