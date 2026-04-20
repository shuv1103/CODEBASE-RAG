from pathlib import Path
from typing import Optional

from .config import (
    SUPPORTED_EXTENSIONS,
    MAX_FILE_SIZE,
    DEFAULT_ENCODING
)


def is_supported_file(file_path: Path) -> bool:
    """Check if file extension is supported."""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def get_file_size(file_path: Path) -> int:
    """Return file size."""
    return file_path.stat().st_size


def read_file_safe(file_path: Path) -> Optional[str]:
    """
    Safely read text file using default encoding.
    Returns None if file cannot be read.
    """

    try:
        size = get_file_size(file_path)

        if size > MAX_FILE_SIZE:
            return None

        return file_path.read_text(encoding=DEFAULT_ENCODING)

    except Exception:
        return None


def get_relative_path(root: Path, file_path: Path) -> str:
    """Return relative path from repo root."""
    return str(file_path.relative_to(root))
