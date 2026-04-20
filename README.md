# CODEBASE-RAG-AGENT

Production-oriented **Codebase Retrieval-Augmented Generation (RAG)** pipeline for software repositories.

This project ingests source code, performs **AST-aware chunking**, embeds chunks into **ChromaDB**, retrieves semantically relevant code snippets for developer questions, and exposes a tool-driven **Google ADK agent** for grounded answers.

---

## Project Overview 

- **RAG Pipeline**: ingestion → chunking → embedding → retrieval → agent layers.
- **Structured Data Models**: Pydantic models across each stage for stronger reliability.
- **Context Quality Engineering**: AST-first chunking strategy to preserve code semantics.
- **LLM Grounding**: retrieval-first tool interface integrated with Google ADK.
- **Evaluation**: RAGAS-based retrieval/generation quality evaluation scaffold.
- **Production Grade**: environment-based configuration, batching, metadata-rich payloads, and persistent vector storage.

---

## System architecture

```text
Repository Files
   │
   ▼
[Ingestion Layer]
RepoLoader → FileDocument[] + ingestion stats
   │
   ▼
[Chunking Layer]
ASTCodeChunker (astchunk) → CodeChunk[] + chunk stats
   │
   ▼
[Embedding Layer]
EmbeddingService (Google embeddings) + formatter + Chroma upsert
   │
   ▼
[Retrieval Layer]
QueryEmbedder + ChromaRetriever + RetrievalService
   │
   ▼
[Agent Layer]
Google ADK LlmAgent + retrieval function tool
   │
   ▼
Grounded answers with file/context evidence
```

---

## Repository structure

```text
.
├── main.py
├── requirements.txt
├── req.txt
├── README.md
└── src
    ├── ingestion
    │   ├── language_detector.py
    │   └── repo_loader.py
    ├── chunking
    │   ├── chunk_models.py
    │   ├── chunker.py
    │   ├── chunk_persist.py
    │   └── chunk_main.py
    ├── embeddings
    │   ├── embeddings_models.py
    │   ├── embeddings_formatter.py
    │   ├── embeddings_client.py
    │   ├── chroma_store.py
    │   └── embeddings_main.py
    ├── retrieval
    │   ├── retrieval_models.py
    │   ├── chroma_retriever.py
    │   ├── query_embedder.py
    │   └── retrieval_service.py
    ├── codebase-rag-agent
    │   ├── tools.py
    │   └── agent.py
    ├── evaluation
    │   ├── loader.py
    │   ├── ragas_eval.py
    │   ├── run_eval.py
    │   └── dataset.json
    ├── utils
    │   ├── config.py
    │   ├── chunk_config.py
    │   └── file_utility.py
    └── data/chunks
        ├── code_chunks.jsonl
        └── chunk_manifest.json
```

---

## Deep-dive: each layer

### 1) Ingestion layer

- Scans repository tree and filters by supported extensions.
- Skips noisy/build/cache directories.
- Produces strongly typed `FileDocument` objects and ingestion stats.
- Detects language from extension (Python, JS/TS, Java, C#, C/C++, Rust, Go, etc.).

**Core files**: `src/ingestion/repo_loader.py`, `src/ingestion/language_detector.py`, `src/utils/file_utility.py`, `src/utils/config.py`

---

### 2) AST chunking layer

- Uses `astchunk` to split code along syntax-aware boundaries.
- Preserves chunk metadata (`symbol_name`, lines, language, relative path).
- Emits JSONL artifact (`content + metadata`) for downstream embedding.

**Design intent:** better retrieval relevance than naive fixed-token splitting.

**Core files**: `src/chunking/chunker.py`, `src/chunking/chunk_models.py`, `src/chunking/chunk_persist.py`, `src/utils/chunk_config.py`

---

### 3) Embedding + vectorization layer

- Builds embedding-ready text with code + metadata context.
- Generates vectors via Google embeddings.
- Upserts vectors + metadata into persistent Chroma collection.
- Tracks batch-level operational counters.

**Core files**: `src/embeddings/embeddings_formatter.py`, `src/embeddings/embeddings_client.py`, `src/embeddings/chroma_store.py`, `src/embeddings/embeddings_main.py`

---

### 4) Retrieval layer

- Converts user query into embedding.
- Executes vector similarity search in Chroma.
- Normalizes results into typed retrieval response.
- Can format retrieved chunks into LLM-ready grounded context blocks.

**Core files**: `src/retrieval/query_embedder.py`, `src/retrieval/chroma_retriever.py`, `src/retrieval/retrieval_service.py`, `src/retrieval/retrieval_models.py`

---

### 5) Agent layer (Google ADK)

- Defines a tool (`retrieve_code_context`) wrapping retrieval service.
- Registers tool in ADK `LlmAgent`.
- Agent instruction enforces grounded answering behavior.

**Core files**: `src/codebase-rag-agent/tools.py`, `src/codebase-rag-agent/agent.py`

---

### 6) Evaluation layer (RAGAS)

- Loads validated evaluation dataset.
- Computes RAG metrics:
  - Faithfulness
  - Answer relevancy
  - Context precision
  - Context recall

**Core files**: `src/evaluation/loader.py`, `src/evaluation/ragas_eval.py`, `src/evaluation/run_eval.py`

---

## Current feature highlights

- ✅ AST-based chunking pipeline.
- ✅ Metadata-rich chunks for explainable retrieval.
- ✅ ChromaDB integration for semantic search.
- ✅ Query-time retrieval service with contextual formatting.
- ✅ ADK-compatible tool-based agent wrapper.
- ✅ RAGAS evaluation bootstrap.

---

## Quickstart

### 1) Clone and setup

```bash
git clone <your-repo-url>
cd CODEBASE-RAG
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure environment

Create `.env` in project root (example):

```env
GOOGLE_API_KEY=your_key_here

# Embeddings
EMBEDDING_MODEL=models/text-embedding-004
CHUNKS_JSON_PATH=src/data/chunks/code_chunks.jsonl
EMBEDDING_BATCH_SIZE=20

# Retrieval
TOP_K=5

# Agent
AGENT_MODEL=gemini-2.0-flash

# Evaluation
EVALUATION_DATASET_PATH=src/evaluation/dataset.json
EVALUATION_LLM_MODEL=gemini-2.0-flash
```

---

## Pipeline runbook

### A) Ingest + chunk codebase

Use/adjust `repo_path` in `src/chunking/chunk_main.py`, then run:

```bash
python -m src.chunking.chunk_main
```

Output artifacts:

- `src/data/chunks/code_chunks.jsonl`
- `src/data/chunks/chunk_manifest.json`

### B) Embed + index chunks

```bash
python -m src.embeddings.embeddings_main
```

### C) Retrieve code context (programmatically)

```python
from src.retrieval.retrieval_service import RetrievalService

service = RetrievalService()
print(service.retrieve_as_context("Where is patient form validation implemented?", top_k=5))
```

### D) Run RAG evaluation

```bash
python -m src.evaluation.run_eval
```

---

## Engineering decisions

- **AST-first chunking over blind windowing** for semantic integrity.
- **Pydantic models** for schema discipline and safer refactors.
- **Metadata-enriched embeddings** to improve retrieval precision.
- **Tool-based ADK agent** to force retrieve-before-generate behavior.
- **Evaluation built in early** to support iterative quality tuning.
