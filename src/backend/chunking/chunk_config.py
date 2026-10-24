"""Chunking constants: supported languages, chunk size and batch size."""

AST_SUPPORTED_LANGUAGES = {
    "python",
    "java",
    "csharp",
    "typescript",
    "javascript",
    "rust",
    "go",
    "c",
    "cpp",
    "dart",
    "html",
    "css",
    "markdown",
}

# Detected language -> chonkie language name. Names currently match; add an
# entry here when one differs.
AST_LANGUAGE_ALIASES = {
    language: language for language in AST_SUPPORTED_LANGUAGES
}

DEFAULT_CODE_CHUNK_SIZE = 2300
DEFAULT_CODE_CHUNK_TOKENIZER = "character"
DEFAULT_CHUNK_BATCH_SIZE = 20
