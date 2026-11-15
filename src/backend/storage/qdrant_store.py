"""Qdrant client, collection setup and the vector store for chunks."""

import logging
import uuid
from collections.abc import Sequence
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from models.embeddings_models import EmbeddedChunk
from utils.config import settings

logger = logging.getLogger(__name__)

REPO_ID_FIELD = "repo_id"
# Id of the ingestion run that last wrote a point (see delete_stale).
INGEST_RUN_FIELD = "ingest_run_id"


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client for QDRANT_URL.

    Returns:
        A QdrantClient, authenticated with QDRANT_API_KEY when it is set.
    """
    url = settings.qdrant_url
    # Host only, at DEBUG, to keep the cluster address out of normal logs.
    logger.debug("Qdrant client for host %s", urlparse(url).hostname)
    return QdrantClient(
        url=url, api_key=settings.qdrant_api_key.get_secret_value() or None
    )


def hnsw_config() -> models.HnswConfigDiff:
    """Return the collection's HNSW index settings.

    Returns:
        HnswConfigDiff from QDRANT_HNSW_M, QDRANT_HNSW_EF_CONSTRUCT and
        QDRANT_HNSW_FULL_SCAN_THRESHOLD.
    """
    return models.HnswConfigDiff(
        m=settings.qdrant_hnsw_m,
        ef_construct=settings.qdrant_hnsw_ef_construct,
        full_scan_threshold=settings.qdrant_hnsw_full_scan_threshold,
    )


def repo_filter(repo_id: str) -> models.Filter:
    """Build a filter that matches only one repo's points.

    Args:
        repo_id: Value of the repo_id payload field to match.

    Returns:
        A Qdrant filter for repo_id == repo_id.
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key=REPO_ID_FIELD, match=models.MatchValue(value=repo_id)
            )
        ]
    )


def point_id(repo_id: str, chunk_id: str) -> str:
    """Return a stable Qdrant point id for a chunk.

    Re-ingesting yields the same ids, so upserts overwrite.

    Args:
        repo_id: Repo the chunk belongs to.
        chunk_id: Chunk id, unique within the repo.

    Returns:
        UUIDv5 of "<repo_id>:<chunk_id>", as a string.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{repo_id}:{chunk_id}"))


class QdrantVectorStore:
    """Write side of the shared chunk collection, filtered by repo_id.

    The collection is created on the first upsert (cosine distance).

    Args:
        repo_id: Scope for count() and delete_stale(); empty means the whole
            collection.
        collection_name: Defaults to QDRANT_COLLECTION.

    Example:
        >>> store = QdrantVectorStore(repo_id="c3460b930023")
        >>> store.upsert(embedded_chunks, run_id=run_id)
    """

    def __init__(
        self, repo_id: str = "", collection_name: str | None = None
    ) -> None:
        self.client = get_qdrant_client()
        self.collection_name = collection_name or settings.qdrant_collection
        self.repo_id = repo_id
        self._collection_ready = False
        logger.debug(
            "Qdrant collection %s (repo_id=%s)",
            self.collection_name,
            repo_id or "*",
        )

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if missing, then sync HNSW and indexes once.

        Safe when several workers race to create it.

        Args:
            vector_size: Embedding dimension for a new collection.

        Raises:
            UnexpectedResponse: If Qdrant rejects a creation for another
                reason.
        """
        if self._collection_ready:
            return
        if self.client.collection_exists(self.collection_name):
            self.sync_hnsw_config()
        else:
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                        hnsw_config=hnsw_config(),
                    ),
                )
            except UnexpectedResponse:
                # Another worker created it first; anything else re-raises.
                if not self.client.collection_exists(self.collection_name):
                    raise
        # Outside the try, so an index failure surfaces; the flag is set only
        # after it succeeds, so this store's next upsert retries it.
        self.ensure_payload_indexes()
        self._collection_ready = True

    def ensure_payload_indexes(self) -> None:
        """Create the repo_id (tenant) and ingest_run_id indexes if missing.

        Qdrant Cloud's strict mode rejects filters on unindexed fields.

        Raises:
            UnexpectedResponse: If Qdrant rejects an index creation.
        """
        schema = (
            self.client.get_collection(self.collection_name).payload_schema
            or {}
        )
        for field, is_tenant in (
            (REPO_ID_FIELD, True),
            (INGEST_RUN_FIELD, False),
        ):
            if field in schema:
                continue
            logger.info(
                "Creating %s index on collection %s",
                field,
                self.collection_name,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.KeywordIndexParams(
                    type=models.KeywordIndexType.KEYWORD,
                    is_tenant=is_tenant,
                ),
                wait=True,
            )

    def sync_hnsw_config(self) -> None:
        """Apply the configured HNSW settings to an existing collection.

        Only updates when the live settings differ; Qdrant rebuilds in the
        background while searches keep working.
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

    def upsert(
        self,
        embedded_chunks: Sequence[EmbeddedChunk],
        *,
        run_id: str | None = None,
    ) -> None:
        """Insert or overwrite chunks, keyed by point_id(repo_id, chunk_id).

        Args:
            embedded_chunks: Chunks to write; a no-op if empty.
            run_id: Ingestion run, stored as ingest_run_id for delete_stale().
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
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    REPO_ID_FIELD: chunk.repo_id,
                    INGEST_RUN_FIELD: run_id,
                },
            )
            for chunk in embedded_chunks
        ]

        self.client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )

    def count(self) -> int:
        """Return how many points the store's repo has.

        Returns:
            The exact count for repo_id (the whole collection when empty); 0 if
            the collection doesn't exist.
        """
        if not self.client.collection_exists(self.collection_name):
            return 0
        count_filter = repo_filter(self.repo_id) if self.repo_id else None
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=count_filter,
            exact=True,
        ).count

    def delete_stale(self, run_id: str) -> None:
        """Delete this repo's points that run_id didn't write.

        Called after the run's last batch, so a failed run keeps the old index.

        Args:
            run_id: The ingestion run whose points are kept.

        Raises:
            ValueError: If the store has no repo_id or run_id is empty.
        """
        if not self.repo_id or not run_id:
            raise ValueError(
                "delete_stale() needs a repo_id and a run_id; "
                "refusing to delete across the whole collection"
            )
        if not self.client.collection_exists(self.collection_name):
            return
        # Points without ingest_run_id don't match the must_not, so they go.
        stale = models.Filter(
            must=repo_filter(self.repo_id).must,
            must_not=[
                models.FieldCondition(
                    key=INGEST_RUN_FIELD,
                    match=models.MatchValue(value=run_id),
                )
            ],
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=stale),
            wait=True,
        )


def main() -> None:
    """Log the collection's point count (local debugging)."""
    from utils.logging_config import configure_logging

    configure_logging()
    store = QdrantVectorStore()
    logger.info("Total chunks in store: %d", store.count())


if __name__ == "__main__":
    main()
