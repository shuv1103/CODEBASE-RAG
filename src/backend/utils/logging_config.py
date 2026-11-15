"""Root logging setup with service, request id and extra={...} fields."""

import logging
import time
from contextvars import ContextVar

from utils.config import settings

# Set per request by RequestIdMiddleware; inherited by tasks and threads.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_RESERVED_LOG_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {
    "message",
    "asctime",
    "service",
}


class ContextFilter(logging.Filter):
    """Log filter that stamps each record with the service and request id.

    Args:
        service: Process name shown on every line, e.g. "api".
    """

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def filter(self, record: logging.LogRecord) -> bool:
        """Add service and, inside an HTTP request, request_id to the record.

        Args:
            record: The record being logged.

        Returns:
            True, so the record is always emitted.
        """
        record.service = self.service
        request_id = request_id_var.get()
        if request_id is not None:
            record.request_id = request_id
        return True


class ExtraFieldsFormatter(logging.Formatter):
    """Log formatter that appends a record's extra={...} fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a record, appending its non-standard attributes.

        Args:
            record: The record to format.

        Returns:
            The base line, plus " | key=value ..." when there are extra fields.
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
    """Configure root logging; call once at a process entry point.

    Lines look like "2026-10-26T12:00:00.123Z INFO [api] module: msg | k=v".
    Replaces any handlers a library installed and holds httpx at WARNING.

    Args:
        service: Process name on every line ("api", "worker", "cli").
        level: Logging level or name. Defaults to LOG_LEVEL.
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
