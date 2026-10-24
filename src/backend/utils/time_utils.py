"""Time helpers."""

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string.

    Returns:
        Timezone-aware ISO timestamp, e.g. "2026-07-17T10:02:11.123456+00:00".
    """
    return datetime.now(timezone.utc).isoformat()
