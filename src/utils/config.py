from pathlib import Path

# Supported source file extensions
SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
}

# Directories to skip during traversal
SKIP_DIRECTORIES = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".next",
    "coverage",
    "target",
    "bin",
    "obj",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    ".pytest_cache",
}

# Files to skip during traversal
SKIP_FILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
}

# File suffixes to skip during traversal
SKIP_FILE_SUFFIXES = {
    ".min.js",
    ".map",
    ".class",
    ".dll",
    ".exe",
    ".so",
    ".dylib",
    ".o",
    ".obj",
}

# Max file size to read (bytes)
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

# Default encoding
DEFAULT_ENCODING = "utf-8"
