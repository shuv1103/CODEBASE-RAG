"""Per-process cap on agent (LLM) runs in flight."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from utils.config import settings

from .exceptions import LLMBusyError


class LLMConcurrencyLimiter:
    """Cap on agent (LLM) runs this process has in flight at once.

    A burst inside one rate-limit window would still hit OpenAI all at once,
    so the overflow queues here. A request that waits longer than
    queue_timeout_seconds gets LLMBusyError (HTTP 503) instead of piling up.
    The cap is per process; the cross-process cap is OPENAI_RPM_LIMIT.

    Args:
        max_concurrent: Most agent runs in flight at once.
        queue_timeout_seconds: Longest a request waits for a free slot.
    """

    def __init__(
        self,
        max_concurrent: int = settings.max_concurrent_llm_calls,
        queue_timeout_seconds: float = settings.llm_queue_timeout_seconds,
    ) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_timeout_seconds = queue_timeout_seconds

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one slot for the duration of the block.

        Raises:
            LLMBusyError: If no slot frees up within queue_timeout_seconds.
        """
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._queue_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise LLMBusyError(retry_after_seconds=5) from exc
        try:
            yield
        finally:
            self._semaphore.release()
