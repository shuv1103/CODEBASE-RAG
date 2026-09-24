import os

from qdrant_client.http.exceptions import UnexpectedResponse

from embeddings.qdrant_store import get_collection_name, get_qdrant_client, repo_filter

from .retrieval_models import RetrievalResponse, RetrievedChunk


class QdrantRetriever:
    """Thin wrapper around Qdrant similarity search.

    Args:
        repo_id: Repository to search. None searches every repo in the
            collection (used by local dev tooling and evaluation).
        collection_name: Collection to query. Defaults to QDRANT_COLLECTION,
            then "code_chunks".
    """

    def __init__(self, repo_id: str | None = None, collection_name: str | None = None) -> None:
        self._client = get_qdrant_client()
        self._collection_name = collection_name or get_collection_name()
        self._repo_id = repo_id

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> RetrievalResponse:
        """Search Qdrant for the most relevant code chunks.

        A collection that doesn't exist yet (nothing ingested) yields an
        empty result instead of an error.

        Args:
            query: Original query text, echoed back in the response.
            query_embedding: Embedding vector of the query.
            top_k: Maximum number of chunks to return.

        Returns:
            RetrievalResponse with chunks ordered by cosine similarity,
            highest first.
        """
        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                query_filter=repo_filter(self._repo_id) if self._repo_id else None,
                limit=top_k,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return RetrievalResponse(query=query, top_k=top_k, results=[])
            raise

        chunks: list[RetrievedChunk] = []

        for point in response.points:
            payload = point.payload or {}

            relative_path = payload.get("relative_path") or ""
            file_name = payload.get("file_name") or os.path.basename(relative_path)

            chunks.append(
                RetrievedChunk(
                    chunk_id=payload.get("chunk_id") or str(point.id),
                    relative_path=relative_path,
                    file_name=file_name,
                    language=payload.get("language") or "",
                    chunk_type=payload.get("chunk_type") or "",
                    symbol_name=payload.get("symbol_name"),
                    parent_symbol=payload.get("parent_symbol"),
                    start_line=payload.get("start_line"),
                    end_line=payload.get("end_line"),
                    content=payload.get("document"),
                    embedding_input_text=payload.get("document"),
                    similarity_score=float(point.score),
                )
            )

        return RetrievalResponse(
            query=query,
            top_k=top_k,
            results=chunks,
        )
