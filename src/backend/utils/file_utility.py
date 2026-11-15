"""File helpers: ingestion filter, language detection and safe reads."""

from pathlib import Path

from utils.config import (
    DEFAULT_ENCODING,
    EXTENSION_LANGUAGE_MAP,
    MAX_FILE_SIZE,
    SKIP_FILE_NAMES,
    SKIP_FILE_SUFFIXES,
    SUPPORTED_EXTENSIONS,
)

# str.endswith takes a tuple, not a set.
_SKIP_SUFFIXES = tuple(SKIP_FILE_SUFFIXES)


def is_supported_file(file_path: Path) -> bool:
    """Check whether a file should be ingested.

    Returns:
        True for a SUPPORTED_EXTENSIONS file that isn't a lockfile or a
        generated file such as app.min.js.
    """
    name = file_path.name
    return (
        file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        and name not in SKIP_FILE_NAMES
        and not name.lower().endswith(_SKIP_SUFFIXES)
    )


def detect_language(file_path: Path) -> str:
    """Detect a file's language from its extension.

    Args:
        file_path: Path of the source file.

    Returns:
        The EXTENSION_LANGUAGE_MAP name, or "unknown".

    Example:
        >>> detect_language(Path("app/Main.TSX"))
        'typescript'
    """
    return EXTENSION_LANGUAGE_MAP.get(file_path.suffix.lower(), "unknown")


def read_file_safe(file_path: Path) -> str | None:
    """Read a UTF-8 text file without raising.

    Returns:
        The text, or None if it exceeds MAX_FILE_SIZE or can't be read or
        decoded.
    """
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return None

        return file_path.read_text(encoding=DEFAULT_ENCODING)

    except (OSError, UnicodeDecodeError):
        return None
