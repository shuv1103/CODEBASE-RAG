import os
import chromadb

from .retrieval_models import RetrievalResponse, RetrievedChunk


class ChromaRetriever:
    """
    Thin wrapper around ChromaDB collection querying.
    """

    def __init__(self, persist_dir: str | None, collection_name: str) -> None:
        chroma_host = os.getenv("CHROMA_HOST")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

        if chroma_host:
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            self._client = chromadb.PersistentClient(path=persist_dir)

        self._collection = self._client.get_collection(name=collection_name)

    def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> RetrievalResponse:
        """
        Search ChromaDB for the most relevant code chunks.
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            metadata = metadata or {}

            relative_path = metadata.get("relative_path") or metadata.get("file_path") or ""
            file_name = metadata.get("file_name") or os.path.basename(relative_path)

            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    relative_path=relative_path,
                    file_name=file_name,
                    language=metadata.get("language", ""),
                    chunk_type=metadata.get("chunk_type", ""),
                    symbol_name=metadata.get("symbol_name"),
                    parent_symbol=metadata.get("parent_symbol"),
                    start_line=metadata.get("start_line"),
                    end_line=metadata.get("end_line"),
                    content=document,
                    embedding_input_text=metadata.get("embedding_input_text"),
                    similarity_score=float(distance),
                )
            )

        return RetrievalResponse(
            query=query,
            top_k=top_k,
            results=chunks,
        )
