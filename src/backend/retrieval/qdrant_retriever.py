"""Repo-scoped similarity search over the Qdrant collection."""

import logging
from pathlib import PurePath

from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from models.retrieval_models import RetrievalResponse, RetrievedChunk
from storage.qdrant_store import get_qdrant_client, repo_filter
from utils.config import settings

logger = logging.getLogger(__name__)


class QdrantRetriever:
    """Similarity search over the shared collection, filtered by repo_id.

    Args:
        repo_id: Repo to search; required unless all_repos is True.
        collection_name: Defaults to QDRANT_COLLECTION.
        all_repos: Search every repo unfiltered (adk web, evaluation only).

    Raises:
        ValueError: Unless exactly one of repo_id / all_repos is given.
    """

    def __init__(
        self,
        repo_id: str | None = None,
        collection_name: str | None = None,
        *,
        all_repos: bool = False,
    ) -> None:
        if all_repos and repo_id is not None:
            raise ValueError("Pass either repo_id or all_repos=True, not both")
        if not all_repos and not repo_id:
            raise ValueError(
                "repo_id is required (got an empty value); pass "
                "all_repos=True to deliberately search every repo"
            )
        self._client = get_qdrant_client()
        self._collection_name = collection_name or settings.qdrant_collection
        self._repo_id = repo_id
        self._search_params = models.SearchParams(
            hnsw_ef=settings.qdrant_hnsw_ef
        )

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> RetrievalResponse:
        """Return the chunks nearest to a query embedding.

        A missing collection (nothing ingested yet) returns no results.

        Args:
            query: Original query text, echoed in the response.
            query_embedding: The query's embedding vector.
            top_k: Most chunks to return.

        Returns:
            RetrievalResponse ordered by cosine similarity, highest first.
        """
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                query_filter=(
                    None
                    if self._repo_id is None
                    else repo_filter(self._repo_id)
                ),
                limit=top_k,
                search_params=self._search_params,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                logger.warning(
                    "Qdrant collection %s not found; returning no results",
                    self._collection_name,
                )
                return RetrievalResponse()
            raise

        return RetrievalResponse(
            results=[self.to_chunk(point) for point in response.points]
        )

    @staticmethod
    def to_chunk(point: models.ScoredPoint) -> RetrievedChunk:
        """Convert a Qdrant search hit into a RetrievedChunk.

        Args:
            point: A scored point from query_points.

        Returns:
            The chunk, with missing payload fields defaulted.
        """
        payload = point.payload or {}
        relative_path = payload.get("relative_path") or ""

        return RetrievedChunk(
            chunk_id=payload.get("chunk_id") or str(point.id),
            relative_path=relative_path,
            file_name=payload.get("file_name") or PurePath(relative_path).name,
            language=payload.get("language") or "",
            chunk_type=payload.get("chunk_type") or "",
            start_line=payload.get("start_line"),
            end_line=payload.get("end_line"),
            content=payload.get("document"),
            similarity_score=float(point.score),
        )
