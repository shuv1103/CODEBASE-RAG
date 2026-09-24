import time
import random
from typing import Callable, TypeVar

T = TypeVar("T")


class ExponentialBackoff:
    """Exponential backoff retry handler for API calls.

    On a rate-limit error (429 / RESOURCE_EXHAUSTED) the call is attempted up
    to max_retries times in total. Each wait doubles: (2 ** attempt) + random
    jitter [0, 1). Non-rate-limit exceptions are re-raised immediately
    without retrying.

    Args:
        max_retries: Maximum total attempts (default: 5).

    Attributes:
        max_retries: Maximum total attempts.

    Example:
        >>> backoff = ExponentialBackoff(max_retries=3)
        >>> backoff.execute(lambda: 42)
        42
    """

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def execute(self, api_func: Callable[[], T]) -> T:
        """Call api_func, retrying with backoff on rate-limit errors.

        Args:
            api_func: Zero-argument callable to invoke.

        Returns:
            The value returned by api_func.

        Raises:
            Exception: Any non-rate-limit error immediately, or the last
                rate-limit error once attempts are exhausted.
        """
        for attempt in range(self.max_retries):
            try:
                return api_func()
            except Exception as e:
                if not self._is_rate_limit_error(e):
                    raise
                if attempt == self.max_retries - 1:
                    raise
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)

    @staticmethod
    def _is_rate_limit_error(e: Exception) -> bool:
        """Detect a rate-limit error from its message.

        Args:
            e: The raised exception.

        Returns:
            True if the message mentions "429" or "RESOURCE_EXHAUSTED".
        """
        msg = str(e).upper()
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg
