from pathlib import Path

EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".h": "c",
    ".cs": "csharp",
    ".dart": "dart",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".markdown": "markdown",
}


def detect_language(file_path: Path) -> str:
    """Detect a file's programming language from its extension.

    Args:
        file_path: Path of the source file.

    Returns:
        Language name from EXTENSION_LANGUAGE_MAP, or "unknown".

    Example:
        >>> detect_language(Path("app/Main.TSX"))
        'typescript'
    """
    return EXTENSION_LANGUAGE_MAP.get(file_path.suffix.lower(), "unknown")
