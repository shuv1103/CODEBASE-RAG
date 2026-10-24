"""Root logging setup that shows extra={...} fields."""

import logging

from .config import settings

_RESERVED_LOG_RECORD_ATTRS = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", (), None)).keys()
) | {
    "message",
    "asctime",
}


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
        extras_str = " ".join(f"{key}={value}" for key, value in extras.items())
        return f"{base} | {extras_str}"


def configure_logging(level: int | str | None = None) -> None:
    """Configure root logging so extra={...} fields are visible.

    Call once at process start (the API app or a CLI entry point), never
    from library modules. Replace any handlers an imported library already
    installed.

    Args:
        level: Root logging level, as a logging constant or a name such as
            "DEBUG". Defaults to settings.log_level (LOG_LEVEL, then "INFO").
    """
    if level is None:
        level = settings.log_level
    handler = logging.StreamHandler()
    handler.setFormatter(
        ExtraFieldsFormatter("%(levelname)s:%(name)s:%(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
