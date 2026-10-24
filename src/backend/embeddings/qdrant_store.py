"""Qdrant client, collection setup and the vector store for chunks."""

import logging
import uuid
from collections.abc import Sequence
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from utils.config import settings

from .embeddings_models import EmbeddedChunk

logger = logging.getLogger(__name__)

REPO_ID_FIELD = "repo_id"


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client for the configured server.

    Both QdrantVectorStore and the retriever connect through this, so a
    local container and Qdrant Cloud differ only in QDRANT_URL and
    QDRANT_API_KEY.

    Returns:
        A QdrantClient for QDRANT_URL (default http://localhost:6333),
        authenticated with QDRANT_API_KEY when it is set.
    """
    url = settings.qdrant_url
    # Host only, at DEBUG, to keep the cluster address out of normal logs.
    logger.debug("Qdrant client for host %s", urlparse(url).hostname)
    return QdrantClient(
        url=url, api_key=settings.qdrant_api_key.get_secret_value() or None
    )


def hnsw_config() -> models.HnswConfigDiff:
    """Return the HNSW index settings for the collection's vectors.

    m (links per node) and ef_construct (candidates checked per insert) set
    how well-connected the graph is; higher ef_construct raises recall at
    the cost of build time only. Below full_scan_threshold (KB of vectors
    matched by a search's filter), Qdrant scans exactly instead of walking
    the graph. payload_m is left unset, so the per-repo_id graphs use m.

    Returns:
        HnswConfigDiff from QDRANT_HNSW_M (16), QDRANT_HNSW_EF_CONSTRUCT (200)
        and QDRANT_HNSW_FULL_SCAN_THRESHOLD (10000).
    """
    return models.HnswConfigDiff(
        m=settings.qdrant_hnsw_m,
        ef_construct=settings.qdrant_hnsw_ef_construct,
        full_scan_threshold=settings.qdrant_hnsw_full_scan_threshold,
    )


def repo_filter(repo_id: str) -> models.Filter:
    """Build a filter that matches only one repo's points.

    Args:
        repo_id: Repository to match on the `repo_id` payload field.

    Returns:
        A Qdrant filter for `repo_id == repo_id`.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key=REPO_ID_FIELD, match=models.MatchValue(value=repo_id)
            )
        ]
    )


def point_id(repo_id: str, chunk_id: str) -> str:
    """Derive a stable Qdrant point id for a chunk.

    Qdrant ids must be unsigned integers or UUIDs, so the key is hashed
    into a UUIDv5. It includes repo_id because chunk ids are only unique
    within a repo, and all repos share one collection. Re-ingesting yields
    the same ids, so upserts overwrite instead of duplicating.

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

    All repos share one collection, separated by a tenant-indexed `repo_id`
    payload field, per Qdrant's multitenancy guidance. The collection is
    created on the first upsert, with cosine distance and the vector size of
    the first batch.

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

    def __init__(
        self, repo_id: str = "", collection_name: str | None = None
    ) -> None:
        self.client = get_qdrant_client()
        self.collection_name = collection_name or settings.qdrant_collection
        self.repo_id = repo_id
        self._hnsw_synced = False
        logger.debug(
            "Qdrant collection %s (repo_id=%s)",
            self.collection_name,
            repo_id or "*",
        )

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection and its tenant index if they don't exist yet.

        Safe to call from several workers at once: if another process
        creates the collection first, the error is ignored. An existing
        collection gets the configured HNSW settings once per store.

        Args:
            vector_size: Embedding dimension for a new collection.
        """
        if self._hnsw_synced:
            return
        if self.client.collection_exists(self.collection_name):
            self.sync_hnsw_config()
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                    hnsw_config=hnsw_config(),
                ),
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
        self._hnsw_synced = True

    def sync_hnsw_config(self) -> None:
        """Apply the configured HNSW settings to an existing collection.

        create_collection only configures new collections, so an older one
        keeps its settings until updated here. Qdrant rebuilds the graph in
        the background while searches keep working.
        """
        desired = hnsw_config()
        config = self.client.get_collection(self.collection_name).config
        vector_hnsw = getattr(config.params.vectors, "hnsw_config", None)
        fields = ("m", "ef_construct", "full_scan_threshold")
        # A per-vector setting overrides the collection-wide one.
        current = {
            f: (
                value
                if (value := getattr(vector_hnsw, f, None)) is not None
                else getattr(config.hnsw_config, f)
            )
            for f in fields
        }
        target = {f: getattr(desired, f) for f in fields}
        if current != target:
            self.client.update_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "": models.VectorParamsDiff(hnsw_config=desired)
                },
            )
            logger.info(
                "Updated HNSW config of %s: %s -> %s",
                self.collection_name,
                current,
                target,
            )
        self._hnsw_synced = True

    def upsert(self, embedded_chunks: Sequence[EmbeddedChunk]) -> None:
        """Insert or update chunks, keyed by (repo_id, chunk_id).

        Args:
            embedded_chunks: EmbeddedChunk objects to write; a no-op if empty.
        """
        if not embedded_chunks:
            return

        self.ensure_collection(len(embedded_chunks[0].vector))

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

        self.client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )

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
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True,
        ).count

    def delete_repo(self) -> None:
        """Delete every point belonging to this store's repo.

        Raises:
            ValueError: If the store has no repo_id (refuses to wipe the
                whole collection).
        """
        if not self.repo_id:
            raise ValueError(
                "delete_repo() needs a repo_id; "
                "refusing to delete the whole collection"
            )
        if not self.client.collection_exists(self.collection_name):
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=repo_filter(self.repo_id)
            ),
            wait=True,
        )


if __name__ == "__main__":
    from utils.logging_config import configure_logging

    configure_logging()
    store = QdrantVectorStore()
    logger.info("Total chunks in store: %d", store.count())
