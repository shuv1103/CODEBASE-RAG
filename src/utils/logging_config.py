import logging
import os

_RESERVED_LOG_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()) | {
    "message",
    "asctime",
}


class ExtraFieldsFormatter(logging.Formatter):
    """Log formatter that appends a record's extra={...} fields to the line.

    logging.basicConfig's default formatter silently drops them — which is
    why calls like caching/semantic_cache_service.py's
    `logger.info("Exact cache hit", extra={"cache_event": "exact_hit",
    "repo_id": repo_id})` never showed `cache_event`/`repo_id` in the actual
    API logs, breaking grep-based debugging (e.g. distinguishing an
    exact-cache hit from a semantic-cache hit).
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
            key: value for key, value in record.__dict__.items() if key not in _RESERVED_LOG_RECORD_ATTRS
        }
        if not extras:
            return base
        extras_str = " ".join(f"{key}={value}" for key, value in extras.items())
        return f"{base} | {extras_str}"


def configure_logging(level: int | str | None = None) -> None:
    """Configure root logging so extra={...} fields are visible.

    Call once, at process start (API app, CLI entry points) — never from
    library modules. `force=True` replaces handlers a prior basicConfig call
    may have already installed (e.g. by an imported library).

    Args:
        level: Root logging level, as a logging constant or a name such as
            "DEBUG". Defaults to the LOG_LEVEL env var, then "INFO".
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(ExtraFieldsFormatter("%(levelname)s:%(name)s:%(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
