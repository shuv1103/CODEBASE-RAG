"""Root logging setup that shows extra={...} fields."""

import logging
import time
from contextvars import ContextVar

from .config import settings

# Set per HTTP request by RequestIdMiddleware; copied into asyncio tasks and
# to_thread calls, so every log line of a request carries it.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED_LOG_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {
    "message",
    "asctime",
    "service",
}


class ContextFilter(logging.Filter):
    """Stamp each record with the process's service name and request id.

    The request id is only set inside an HTTP request, so it shows up as a
    request_id=... extra field there and is absent elsewhere.

    Args:
        service: Process name shown on every line, such as "api".
    """

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        """Add service and, inside a request, request_id to the record."""
        record.service = self.service
        request_id = request_id_var.get()
        if request_id is not None:
            record.request_id = request_id
        return True


class ExtraFieldsFormatter(logging.Formatter):
    """Log formatter that appends a record's extra={...} fields to the line.

    The default formatter drops them, which hides fields such as
    cache_event and repo_id that the logs are searched by.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render a record, appending any non-standard attributes as key=value.

        Args:
            record: The log record to format.

        Returns:
            The base formatted line, followed by " | key=value ..." when
            the record carries extra fields.
        """
        base = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_ATTRS
        }
        if not extras:
            return base
        extras_str = " ".join(
            f"{key}={value}" for key, value in extras.items()
        )
        return f"{base} | {extras_str}"


def configure_logging(
    service: str = "cli", level: int | str | None = None
) -> None:
    """Configure root logging so extra={...} fields are visible.

    Call once at process start (the API app, the Celery worker or a CLI
    entry point), never from library modules. Replace any handlers an
    imported library already installed. Lines look like
    "2026-10-26T12:00:00.123Z INFO [api] module: message | key=value".

    Args:
        service: Process name on every line, so the API and worker sharing
            one container's stdout can be told apart.
        level: Root logging level, as a logging constant or a name such as
            "DEBUG". Defaults to settings.log_level (LOG_LEVEL, then "INFO").
    """
    if level is None:
        level = settings.log_level
    formatter = ExtraFieldsFormatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s [%(service)s] "
        "%(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter(service))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    # httpx logs every OpenAI/Qdrant/LangCache request at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
