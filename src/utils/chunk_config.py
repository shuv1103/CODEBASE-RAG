AST_SUPPORTED_LANGUAGES = {"python", "java", "csharp", "typescript"}

AST_LANGUAGE_ALIASES = {
    "python": "python",
    "java": "java",
    "csharp": "csharp",
    "typescript": "typescript",
}

DEFAULT_AST_MAX_CHUNK_SIZE = 2300 # Recommended max chunk size for AST-based chunking to balance context window limits and retrieval relevance(2000-2500 tokens)
DEFAULT_AST_CHUNK_OVERLAP = 400 # 10 - 20% of chunk size is recommended for chunk overlap 
DEFAULT_AST_CHUNK_EXPANSION = True
DEFAULT_METADATA_TEMPLATE = "default"
