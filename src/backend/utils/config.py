"""Static ingestion filters and the typed, validated runtime settings.

`settings` is built on import, so a missing or invalid variable stops
the process at startup with a ValidationError naming the field.
"""

from typing import Annotated, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# The single extension -> language table; add new languages here only.
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

SUPPORTED_EXTENSIONS = frozenset(EXTENSION_LANGUAGE_MAP)

SKIP_DIRECTORIES = frozenset(
    {
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
)

SKIP_FILE_NAMES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Cargo.lock",
    }
)

SKIP_FILE_SUFFIXES = frozenset(
    {
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
)

MAX_FILE_SIZE = 2 * 1024 * 1024

DEFAULT_ENCODING = "utf-8"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Every environment variable the app reads, typed and validated.

    Field names map case-insensitively to env vars (openai_api_key <-
    OPENAI_API_KEY); an empty variable counts as unset. Only OPENAI_API_KEY is
    required everywhere; process-specific values are checked where used.

    Raises:
        pydantic.ValidationError: On construction, if a value is missing or
            invalid (values are hidden from the message).

    Example:
        >>> from utils.config import settings
        >>> settings.top_k
        5
    """

    # hide_input_in_errors keeps API keys out of validation errors.
    model_config = SettingsConfigDict(
        env_ignore_empty=True, extra="ignore", hide_input_in_errors=True
    )

    # --- OpenAI / models ---
    openai_api_key: SecretStr
    # Required only where the LLM runs (API, evaluation).
    agent_model: str | None = None
    embedding_model: str = "text-embedding-3-large"
    embedding_batch_size: PositiveInt = 20
    # Used by the embeddings_main CLI only.
    embedding_output_dimension: PositiveInt | None = None
    backoff_max_retries: PositiveInt = 5
    backoff_max_wait_seconds: NonNegativeFloat = 30.0
    llm_num_retries: NonNegativeInt = 3
    # 0 turns the shared OpenAI budget off.
    openai_rpm_limit: NonNegativeInt = 0
    openai_budget_max_wait_seconds: NonNegativeFloat = 30.0

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: str = "code_chunks"
    qdrant_hnsw_m: PositiveInt = 16
    qdrant_hnsw_ef_construct: PositiveInt = 200
    qdrant_hnsw_full_scan_threshold: PositiveInt = 10000
    qdrant_hnsw_ef: PositiveInt = 128

    # --- Retrieval ---
    top_k: PositiveInt = 5

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"
    # Empty falls back to redis_url.
    celery_broker_url: str = ""
    celery_result_backend: str = ""
    # 0 keeps status records forever.
    repo_status_ttl_seconds: NonNegativeInt = 0
    redis_status_max_connections: PositiveInt = 5
    celery_broker_pool_limit: PositiveInt = 2
    celery_redis_max_connections: PositiveInt = 4

    # --- Answer cache (Redis LangCache) ---
    langcache_enabled: bool = True
    langcache_server_url: str = ""
    langcache_cache_id: str = ""
    langcache_api_key: SecretStr = SecretStr("")
    langcache_ttl: PositiveInt = 3600
    langcache_similarity_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0
    )

    # --- Ingestion limits ---
    max_repo_files: PositiveInt = 5000
    # Measured on the checked-out files.
    max_repo_bytes: PositiveInt = 200 * 1024 * 1024
    # GitHub's packed repo size, checked before cloning.
    max_repo_download_bytes: PositiveInt = 1024 * 1024 * 1024
    # Caps embedding spend per repo.
    max_repo_chunks: PositiveInt = 20000
    clone_timeout_seconds: PositiveInt = 120
    ingestion_timeout_seconds: PositiveInt = 900

    # --- API ---
    api_cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
    ]
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    # Serves /docs, /redoc and /openapi.json.
    api_docs_enabled: bool = False
    max_request_bytes: PositiveInt = 64 * 1024
    chat_rate_limit: str = "10/minute"
    create_repo_rate_limit: str = "5/minute"
    global_chat_rate_limit: str = "60/minute"
    global_create_repo_rate_limit: str = "20/hour"
    rate_limit_storage_uri: str = "memory://"
    # Proxies appending to X-Forwarded-For (Render: 1); 0 ignores the header.
    trusted_proxy_hops: NonNegativeInt = 0
    max_concurrent_llm_calls: PositiveInt = 4
    llm_queue_timeout_seconds: PositiveFloat = 30.0
    agent_registry_max_repos: PositiveInt = 20
    chat_session_ttl_seconds: PositiveInt = 3600
    log_level: LogLevel = "INFO"

    # --- Tracing (LangSmith) ---
    langchain_tracing_v2: bool = False
    langchain_api_key: SecretStr = SecretStr("")
    langchain_project: str = "default"

    # --- Local CLIs and evaluation (optional; checked by entry points) ---
    host_repo_path: str | None = None
    chunks_json_path: str | None = None
    evaluation_dataset_path: str | None = None
    # RAGAS judge and embedding models.
    openai_llm_model: str | None = None
    openai_embedding_model: str | None = None
    langsmith_eval_dataset_name: str = "codebase-rag-eval"
    langsmith_eval_experiment_name: str = "codebase-rag-eval"

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def split_origins(cls, value: str | list[str]) -> list[str]:
        """Parse API_CORS_ORIGINS from comma-separated text, dropping blanks.

        Args:
            value: Raw env string ("https://a.app, https://b.app") or a list.

        Returns:
            The stripped, non-empty origins.
        """
        items = value.split(",") if isinstance(value, str) else value
        return [origin.strip() for origin in items if origin.strip()]

    @field_validator("log_level", mode="before")
    @classmethod
    def upper_log_level(cls, value: str) -> str:
        """Accept LOG_LEVEL in any case ("debug" -> "DEBUG").

        Args:
            value: Raw LOG_LEVEL value.

        Returns:
            The upper-cased level name.
        """
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def default_celery_urls(self) -> "Settings":
        """Point unset Celery broker and result backend URLs at REDIS_URL.

        Returns:
            The settings, with both Celery URLs filled in.
        """
        self.celery_broker_url = self.celery_broker_url or self.redis_url
        self.celery_result_backend = (
            self.celery_result_backend or self.redis_url
        )
        return self

    @property
    def tracing_enabled(self) -> bool:
        """Whether tracing is on: LANGCHAIN_TRACING_V2 plus an API key."""
        return self.langchain_tracing_v2 and bool(
            self.langchain_api_key.get_secret_value()
        )

    @property
    def ingestion_hard_time_limit_seconds(self) -> int:
        """Celery hard time limit: INGESTION_TIMEOUT_SECONDS + 60 s."""
        return self.ingestion_timeout_seconds + 60

    @property
    def celery_visibility_timeout_seconds(self) -> int:
        """Broker visibility timeout: the hard limit + 30 min.

        --pool=solo ignores time limits, so a run can overrun by a batch; the
        margin keeps a running task from being redelivered.
        """
        return self.ingestion_hard_time_limit_seconds + 1800

    @property
    def ingestion_stale_after_seconds(self) -> int:
        """Silence after which a running ingestion counts as abandoned."""
        return self.ingestion_hard_time_limit_seconds + 60

    @property
    def ingestion_pending_stale_after_seconds(self) -> int:
        """Age (24 h) after which a "pending" ingestion counts as lost."""
        return 24 * 3600

    @property
    def plaintext_remote_redis_urls(self) -> list[str]:
        """Names of Redis URL settings using redis:// to a non-local host."""
        urls = {
            "REDIS_URL": self.redis_url,
            "CELERY_BROKER_URL": self.celery_broker_url,
            "CELERY_RESULT_BACKEND": self.celery_result_backend,
        }
        return [
            name
            for name, url in urls.items()
            if url.startswith("redis://")
            and urlparse(url).hostname
            not in {"localhost", "127.0.0.1", "redis"}
        ]


# LangSmith/LangChain/OpenAI read os.environ; real env vars beat .env.
load_dotenv()

settings = Settings()
