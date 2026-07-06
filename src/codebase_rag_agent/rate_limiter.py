import time
import random
from typing import Callable, TypeVar

T = TypeVar("T")


class ExponentialBackoff:
    """
    Exponential backoff retry handler for API calls.

    On a rate-limit error (429 / RESOURCE_EXHAUSTED) the call is retried up to
    max_retries times. Each wait doubles: (2 ** attempt) + random jitter [0, 1).
    Non-rate-limit exceptions are re-raised immediately without retrying.
    """

    def __init__(self, max_retries: int = 5) -> None:
        self.max_retries = max_retries

    def execute(self, api_func: Callable[[], T]) -> T:
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
        msg = str(e).upper()
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg
