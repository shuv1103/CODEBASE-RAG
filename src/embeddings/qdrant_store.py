import logging
import os
import uuid
from urllib.parse import urlparse

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "code_chunks"
REPO_ID_FIELD = "repo_id"


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client for the configured server.

    Repository Pattern: single source of truth for how this project connects
    to Qdrant. Both the write-side (QdrantVectorStore, below) and the
    read-side (retrieval.qdrant_retriever.QdrantRetriever) call this. The
    same code talks to a local `qdrant/qdrant` container (docker-compose.yml)
    or a Qdrant Cloud cluster — only QDRANT_URL / QDRANT_API_KEY differ.

    Returns:
        A QdrantClient for QDRANT_URL (default http://localhost:6333),
        authenticated with QDRANT_API_KEY when it is set.
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    # Host only, at DEBUG: keeps the cluster address out of normal (INFO) logs.
    logger.debug("Qdrant client for host %s", urlparse(url).hostname)
    return QdrantClient(url=url, api_key=api_key)


def get_collection_name() -> str:
    """Return the name of the single collection that holds every repo's chunks.

    Returns:
        QDRANT_COLLECTION, or "code_chunks" when unset.
    """
    return os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION_NAME)


def repo_filter(repo_id: str) -> models.Filter:
    """Build a filter that matches only one repo's points.

    Args:
        repo_id: Repository to match on the `repo_id` payload field.

    Returns:
        A Qdrant filter for `repo_id == repo_id`.
    """
    return models.Filter(
        must=[models.FieldCondition(key=REPO_ID_FIELD, match=models.MatchValue(value=repo_id))]
    )


def point_id(repo_id: str, chunk_id: str) -> str:
    """Derive a stable Qdrant point id for a chunk.

    Qdrant point ids must be unsigned integers or UUIDs, so the string
    chunk_id is hashed into a UUIDv5. The repo_id is part of the key because
    chunk ids are only unique within one repo (e.g. `README.md::chunk::0`),
    and all repos share one collection. Re-ingesting a repo yields the same
    ids, so upserts overwrite instead of duplicating.

    Args:
        repo_id: Repository the chunk belongs to.
        chunk_id: Chunk id, unique within the repo.

    Returns:
        The UUID as a string.

    Example:
        >>> point_id("c3460b930023", "README.md::chunk::0") == point_id("c3460b930023", "README.md::chunk::0")
        True
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{repo_id}:{chunk_id}"))


class QdrantVectorStore:
    """Write-side wrapper around the shared Qdrant collection.

    Every repo's chunks live in one collection, separated by a `repo_id`
    payload field that carries a tenant index (`is_tenant=True`), per
    Qdrant's multitenancy guidance — rather than one collection per repo.
    The collection is created on the first upsert, with cosine distance
    (HNSW index) and the vector size of the first batch.

    Args:
        repo_id: Repository this store is scoped to for count() and
            delete_repo(). Empty means the whole collection.
        collection_name: Collection to write to. Defaults to
            QDRANT_COLLECTION, then "code_chunks".

    Attributes:
        client: The underlying Qdrant client.
        collection_name: The target collection.
        repo_id: The repo scope for count() and delete_repo().

    Example:
        >>> store = QdrantVectorStore(repo_id="c3460b930023")
        >>> store.upsert(embedded_chunks)
        >>> print(store.count())
    """

    def __init__(self, repo_id: str = "", collection_name: str | None = None) -> None:
        self.client = get_qdrant_client()
        self.collection_name = collection_name or get_collection_name()
        self.repo_id = repo_id
        logger.debug("Qdrant collection %s (repo_id=%s)", self.collection_name, repo_id or "*")

    def _ensure_collection(self, vector_size: int) -> None:
        """Create the collection and its tenant index if they don't exist yet.

        Safe to call concurrently from several workers: if another process
        creates the collection first, the resulting error is ignored.

        Args:
            vector_size: Embedding dimension for a new collection.
        """
        if self.client.collection_exists(self.collection_name):
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=REPO_ID_FIELD,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=True,
                ),
            )
        except UnexpectedResponse:
            if not self.client.collection_exists(self.collection_name):
                raise

    def upsert(self, embedded_chunks):
        """Insert or update chunks, keyed by (repo_id, chunk_id).

        Args:
            embedded_chunks: EmbeddedChunk objects to write; a no-op if empty.
        """
        if not embedded_chunks:
            return

        self._ensure_collection(len(embedded_chunks[0].vector))

        points = [
            models.PointStruct(
                id=point_id(chunk.repo_id, chunk.chunk_id),
                vector=chunk.vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document": chunk.embedding_input_text,
                    "relative_path": chunk.relative_path,
                    "file_name": chunk.file_name,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "symbol_name": chunk.symbol_name,
                    "parent_symbol": chunk.parent_symbol,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    REPO_ID_FIELD: chunk.repo_id,
                    "source_url": chunk.source_url,
                    "embedding_model": chunk.embedding_model,
                    "embedding_dimension": chunk.embedding_dimension,
                    "token_estimate": chunk.token_estimate,
                    "created_at": chunk.created_at,
                },
            )
            for chunk in embedded_chunks
        ]

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def count(self) -> int:
        """Return how many chunks are stored for this store's repo.

        Returns:
            The exact point count for repo_id, or for the whole collection
            when repo_id is empty; 0 if the collection doesn't exist yet.
        """
        if not self.client.collection_exists(self.collection_name):
            return 0
        count_filter = repo_filter(self.repo_id) if self.repo_id else None
        return self.client.count(
            collection_name=self.collection_name, count_filter=count_filter, exact=True
        ).count

    def delete_repo(self) -> None:
        """Delete every point belonging to this store's repo.

        Raises:
            ValueError: If the store has no repo_id (refuses to wipe the
                whole collection).
        """
        if not self.repo_id:
            raise ValueError("delete_repo() needs a repo_id; refusing to delete the whole collection")
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=repo_filter(self.repo_id)),
            wait=True,
        )


if __name__ == "__main__":
    from utils.logging_config import configure_logging

    configure_logging()
    store = QdrantVectorStore()
    logger.info("Total chunks in store: %d", store.count())
