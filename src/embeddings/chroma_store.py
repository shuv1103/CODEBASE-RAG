import os

import chromadb
from dotenv import load_dotenv


load_dotenv()


class ChromaVectorStore:
    def __init__(self):
        persist_dir = "/home/user/CODEBASE-RAG-AGENT/src /data"
        collection_name = "code_chunks"

        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        print(f"Chroma DB path: {os.path.abspath(persist_dir)}")
        print(f"Chroma collection: {collection_name}")

    def upsert(self, embedded_chunks):
        if not embedded_chunks:
            return

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for chunk in embedded_chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.embedding_input_text)
            embeddings.append(chunk.vector)

            metadatas.append({
                "relative_path": chunk.relative_path,
                "file_name": chunk.file_name,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "symbol_name": chunk.symbol_name or "",
                "parent_symbol": chunk.parent_symbol or "",
                "start_line": chunk.start_line if chunk.start_line is not None else -1,
                "end_line": chunk.end_line if chunk.end_line is not None else -1,
                "embedding_model": chunk.embedding_model,
                "embedding_dimension": chunk.embedding_dimension,
                "token_estimate": chunk.token_estimate if chunk.token_estimate is not None else -1,
                "created_at": chunk.created_at,
            })

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def count(self) -> int:
        return self.collection.count()
    

if __name__ == "__main__":
    store = ChromaVectorStore()
    print(f"Total chunks in store: {store.count()}")
    
