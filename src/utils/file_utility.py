from pathlib import Path
from typing import Optional

from .config import (
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE,
    DEFAULT_ENCODING
)


def is_supported_file(file_path: Path) -> bool:
    """Check whether a file's extension is in the supported set.

    Args:
        file_path: Path of the file to check.

    Returns:
        True if the lowercased suffix is in SUPPORTED_EXTENSIONS, else False.
    """
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def get_file_size(file_path: Path) -> int:
    """Return the size of a file on disk.

    Args:
        file_path: Path of the file to measure.

    Returns:
        File size in bytes.
    """
    return file_path.stat().st_size


def read_file_safe(file_path: Path) -> Optional[str]:
    """Safely read a text file using the default encoding.

    Files larger than MAX_FILE_SIZE are skipped, and any read or decode
    error is swallowed rather than raised.

    Args:
        file_path: Path of the file to read.

    Returns:
        The file's text content, or None if the file is too large or
        cannot be read.
    """

    try:
        size = get_file_size(file_path)

        if size > MAX_FILE_SIZE:
            return None

        return file_path.read_text(encoding=DEFAULT_ENCODING)

    except Exception:
        return None


def get_relative_path(root: Path, file_path: Path) -> str:
    """Return a file's path relative to the repository root.

    Args:
        root: Repository root directory.
        file_path: Path of a file located under root.

    Returns:
        The relative path as a string.

    Raises:
        ValueError: If file_path is not located under root.
    """
    return str(file_path.relative_to(root))
