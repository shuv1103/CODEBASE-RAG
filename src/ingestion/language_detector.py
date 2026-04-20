from pathlib import Path

EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript_react",
    ".tsx": "typescript_react",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".h": "c",
    ".cs": "csharp",
}


def detect_language(file_path: Path) -> str:
    """Detect programming language from file extension."""
    return EXTENSION_LANGUAGE_MAP.get(file_path.suffix.lower(), "unknown")
