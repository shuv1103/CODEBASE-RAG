"""Shared OpenAI request budget and exponential backoff on 429s."""

import asyncio
import functools
import logging
import math
import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

import redis

from utils.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# "429" only as a whole number, so ids, sizes and line numbers don't match.
RATE_LIMIT_MESSAGE = re.compile(
    r"\b429\b|RATE[ _]LIMIT|TOO MANY REQUESTS|RESOURCE_EXHAUSTED"
)


class OpenAIRequestBudget:
    """OpenAI requests-per-minute budget shared by every process via Redis.

    Each call takes a slot from a per-second Redis counter capped at
    ceil(rpm / 60). Fails open: disabled, or skipped while Redis is down.

    Args:
        requests_per_minute: Shared budget; <= 0 disables it.
        redis_url: Redis holding the counters.
        max_wait_seconds: Longest acquire() waits before calling anyway.

    Attributes:
        enabled: Whether the budget is enforced.
        per_second: Slots handed out per one-second window.
    """

    _KEY_PREFIX = "codebase-rag:openai-budget:"
    _REDIS_RETRY_SECONDS = 30.0
    # One INCR per slot, so two connections serve many callers.
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
        # Bounded (Redis Cloud's 30-client limit); connects only when used.
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
        """Check whether acquire() returns at once (off, or Redis down)."""
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
        # Window full: sleep into the next one, jittered.
        return (1.0 - (time.time() % 1.0)) + random.uniform(0, 0.25)

    def past_deadline(self, deadline: float, sleep_for: float) -> bool:
        """Check whether sleeping would overrun the deadline, logging if so.

        Args:
            deadline: time.monotonic() value to stop waiting at.
            sleep_for: Seconds the caller is about to sleep.

        Returns:
            True if the caller should stop waiting and call anyway.
        """
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
            True if a slot was taken (or the budget is off); False if the wait
            timed out, in which case the caller proceeds anyway.
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
    """Return the process-wide OpenAI budget, built once from settings.

    Returns:
        The budget for OPENAI_RPM_LIMIT and OPENAI_BUDGET_MAX_WAIT_SECONDS.
    """
    return OpenAIRequestBudget(
        requests_per_minute=settings.openai_rpm_limit,
        redis_url=settings.redis_url,
        max_wait_seconds=settings.openai_budget_max_wait_seconds,
    )


class ExponentialBackoff:
    """Retries OpenAI rate-limit (429) errors with exponential backoff.

    Each attempt first takes a shared budget slot. An exhausted quota
    (insufficient_quota) can't be fixed by waiting, so it is not retried.

    Args:
        max_retries: Total attempts. Defaults to BACKOFF_MAX_RETRIES.
        max_wait_seconds: Longest single wait. Defaults to
            BACKOFF_MAX_WAIT_SECONDS.
        budget: Defaults to the process-wide budget.
    """

    def __init__(
        self,
        *,
        max_retries: int | None = None,
        max_wait_seconds: float | None = None,
        budget: OpenAIRequestBudget | None = None,
    ) -> None:
        self.max_retries = (
            max_retries
            if max_retries is not None
            else settings.backoff_max_retries
        )
        self._max_wait_seconds = (
            max_wait_seconds
            if max_wait_seconds is not None
            else settings.backoff_max_wait_seconds
        )
        self._budget = budget

    def execute(self, api_func: Callable[[], T]) -> T:
        """Call api_func, retrying rate-limit errors with backoff.

        Args:
            api_func: Zero-argument callable making the OpenAI request.

        Returns:
            Whatever api_func returns.

        Raises:
            Exception: A non-retryable error at once, or the last rate-limit
                error once attempts run out.
            ValueError: If max_retries < 1.
        """
        budget = self._budget or get_openai_budget()
        for attempt in range(self.max_retries):
            budget.acquire()
            try:
                return api_func()
            except Exception as error:
                if not self.is_rate_limit_error(error):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                wait_time = self.wait_seconds(error, attempt)
                logger.warning(
                    "OpenAI rate limit (attempt %d/%d); retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
        # Reached only when max_retries < 1, so api_func never ran.
        raise ValueError(f"max_retries must be >= 1, got {self.max_retries}")

    def wait_seconds(self, error: Exception, attempt: int) -> float:
        """Return the seconds to wait before the next attempt.

        Args:
            error: The rate-limit error.
            attempt: Zero-based index of the attempt that failed.

        Returns:
            max(Retry-After, 2**attempt + jitter), capped at max_wait_seconds.
        """
        exponential = (2**attempt) + random.uniform(0, 1)
        retry_after = self.retry_after_seconds(error) or 0.0
        return min(max(exponential, retry_after), self._max_wait_seconds)

    @staticmethod
    def retry_after_seconds(error: Exception) -> float | None:
        """Read OpenAI's retry hint from the error's response headers.

        Args:
            error: The raised exception.

        Returns:
            Seconds from retry-after-ms or retry-after, or None without a hint.
        """
        headers = getattr(getattr(error, "response", None), "headers", None)
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
    def is_rate_limit_error(error: Exception) -> bool:
        """Detect a retryable rate-limit error.

        Checks status_code == 429 first, then the message text, for wrappers
        that keep only the text.

        Args:
            error: The raised exception.

        Returns:
            True for a retryable 429 / RESOURCE_EXHAUSTED; False for
            insufficient_quota and everything else.
        """
        message = str(error).upper()
        if (
            getattr(error, "code", None) == "insufficient_quota"
            or "INSUFFICIENT_QUOTA" in message
        ):
            return False
        if getattr(error, "status_code", None) == 429:
            return True
        return RATE_LIMIT_MESSAGE.search(message) is not None
