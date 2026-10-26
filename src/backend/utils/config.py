"""Project configuration: static ingestion filters and typed runtime settings.

Every environment variable the app reads is declared once, on `Settings`,
with its type, default and constraints. `settings` is built on first import,
so a missing or malformed value fails the process at startup with a clear
ValidationError rather than deep inside a request or ingestion task.
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

SUPPORTED_EXTENSIONS = frozenset(
    {
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
        ".dart",
        ".html",
        ".css",
        ".md",
        ".markdown",
    }
)

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
    """Typed, validated view of every environment variable the app reads.

    Field names map case-insensitively to environment variables
    (`openai_api_key` ← `OPENAI_API_KEY`). An empty variable counts as
    unset, so it falls back to the default (or fails, if required).

    OPENAI_API_KEY is required everywhere; everything else has a default.
    Values only some processes need (AGENT_MODEL, CLI and evaluation paths)
    are optional here and checked where they are used.

    Raises:
        pydantic.ValidationError: On construction, if a required variable is
            missing or any value has the wrong type or is out of range.

    Example:
        >>> from utils.config import settings
        >>> settings.top_k
        5
    """

    # Without hide_input_in_errors, a validation error logs every value read
    # from the environment, API keys included.
    model_config = SettingsConfigDict(
        env_ignore_empty=True, extra="ignore", hide_input_in_errors=True
    )

    # --- OpenAI / models ---
    openai_api_key: SecretStr
    # Required by the API and evaluation only (checked where the LLM is
    # built); the worker and CLIs never call the LLM.
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
    # Whole-repo size GitHub reports (all history, packed), checked before
    # cloning so a multi-GB repo is rejected without being downloaded.
    max_repo_download_bytes: PositiveInt = 1024 * 1024 * 1024
    # Caps embedding spend per repo.
    max_repo_chunks: PositiveInt = 20000
    clone_timeout_seconds: PositiveInt = 120
    ingestion_timeout_seconds: PositiveInt = 900

    # --- API ---
    api_cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", ""]
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
    # Proxies in front of the API that append to X-Forwarded-For (Render: 1).
    # 0 ignores the header and uses the TCP peer address.
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
        """Parse a comma-separated origin list, dropping blanks.

        Args:
            value: Raw env string (e.g. "https://a.app, https://b.app") or a
                list.

        Returns:
            The stripped, non-empty origins.
        """
        items = value.split(",") if isinstance(value, str) else value
        return [origin.strip() for origin in items if origin.strip()]

    @field_validator("log_level", mode="before")
    @classmethod
    def upper_log_level(cls, value: str) -> str:
        """Accept a log level in any case ("debug" -> "DEBUG").

        Args:
            value: Raw LOG_LEVEL value.

        Returns:
            The upper-cased level name.
        """
        return value.upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def default_celery_urls(self) -> "Settings":
        """Point the Celery broker and result backend at REDIS_URL when unset.

        Returns:
            The settings with both Celery URLs filled in.
        """
        self.celery_broker_url = self.celery_broker_url or self.redis_url
        self.celery_result_backend = (
            self.celery_result_backend or self.redis_url
        )
        return self

    @property
    def tracing_enabled(self) -> bool:
        """Whether LangSmith tracing is on.

        Needs LANGCHAIN_TRACING_V2 and an API key to both be set.
        """
        return self.langchain_tracing_v2 and bool(
            self.langchain_api_key.get_secret_value()
        )

    @property
    def ingestion_hard_time_limit_seconds(self) -> int:
        """Celery hard time limit.

        Backstop SIGKILL 60 s after the soft limit (INGESTION_TIMEOUT_SECONDS).
        """
        return self.ingestion_timeout_seconds + 60

    @property
    def celery_visibility_timeout_seconds(self) -> int:
        """Broker visibility timeout.

        The solo pool ignores Celery's time limits, so a task stops only at
        the pipeline's own timeout checks, which can overrun by one chunking
        batch. The 30-minute margin covers that, so a running task is never
        redelivered.
        """
        return self.ingestion_hard_time_limit_seconds + 1800

    @property
    def ingestion_stale_after_seconds(self) -> int:
        """Age after which a running ingestion is treated as abandoned.

        A running ingestion writes progress after every chunking and
        embedding batch, so a record silent this long has no live task (its
        worker was killed, for example).
        """
        return self.ingestion_hard_time_limit_seconds + 60

    @property
    def ingestion_pending_stale_after_seconds(self) -> int:
        """Age after which a "pending" ingestion is treated as lost.

        A queued task can wait behind several long ingestions, so only a
        message lost from the broker should ever reach this.
        """
        return 24 * 3600

    @property
    def plaintext_remote_redis_urls(self) -> list[str]:
        """Names of the Redis settings that reach a remote host without TLS.

        That is, a redis:// URL whose host is not local.
        """
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


# LangSmith, LangChain and OpenAI clients read os.environ, not Settings.
# Real environment variables keep priority over .env.
load_dotenv()

settings = Settings()
