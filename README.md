<div align="center">

# CODEBASE-RAG-AGENT

**Ask your codebase anything. Get grounded, evidence-backed answers.**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![Google ADK](https://img.shields.io/badge/Google%20ADK-1.31.1-4285F4?style=flat-square&logo=google&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-8E75B2?style=flat-square&logo=google&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)
![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-22c55e?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

[Overview](#overview) · [Architecture](#system-architecture) · [Agent Demo](#agent-in-action) · [Quickstart](#quickstart) · [Pipeline](#pipeline-runbook) · [Deep Dive](#deep-dive-each-layer) · [Evaluation](#6-evaluation-layer-ragas)

</div>

---

## Overview

Most code-search tools do text matching. This project builds a **semantic understanding layer** over an entire software repository using a production-grade RAG pipeline.

Point it at any codebase. It ingests every source file, chunks code along **AST boundaries** (not arbitrary token windows), embeds each chunk with Google's embedding model, and stores everything in **ChromaDB**. A **Google ADK agent** then answers developer questions by first retrieving the most relevant code evidence — and only then generating a response.

Ask *"Where is JWT authentication implemented?"* and get back the exact files, classes, and methods — not a hallucinated description.

**Core capabilities at a glance:**

| Capability | Details |
|---|---|
| AST-aware chunking | Splits at function/class boundaries, preserves dependency overlap |
| Metadata-rich vectors | Language, file path, symbol name, line range stored alongside embeddings |
| Persistent vector store | ChromaDB with batch upsert and incremental re-indexing |
| Grounded ADK agent | Tool-first architecture: `retrieve_code_context` fires before any generation |
| Full observability | ADK Events + Traces panel shows every tool call lifecycle |
| RAGAS evaluation | Faithfulness, answer relevancy, context precision & recall out of the box |
| Docker-native | Three-service Compose pipeline: chunk → embed → agent |

---

## System Architecture

```text
 ┌─────────────────────────────────────────────────────┐
 │                   Repository Files                   │
 └──────────────────────────┬──────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │     Ingestion Layer      │
              │  RepoLoader             │
              │  FileDocument[] + stats │
              └────────────┬────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │     Chunking Layer       │
              │  ASTCodeChunker          │
              │  CodeChunk[] + stats    │
              └────────────┬────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Embedding + Vector DB   │
              │  Google text-embedding   │
              │  ChromaDB upsert        │
              └────────────┬────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │     Retrieval Layer      │
              │  QueryEmbedder           │
              │  ChromaRetriever         │
              │  Typed retrieval results │
              └────────────┬────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │      Agent Layer         │
              │  Google ADK LlmAgent     │
              │  retrieve_code_context   │
              └────────────┬────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │  Grounded answers with file evidence   │
        └───────────────────────────────────────┘
```

---

## DEMO(Screenshot)

> Tested against a **Hospital Management System** codebase (React + Spring Boot) using the **Google ADK UI **.

<img width="1900" height="754" alt="image" src="https://github.com/user-attachments/assets/0da3e72b-5133-4c27-b152-4ac88037aa97" />


**What this screenshot demonstrates:**

The agent `codebase_rag_agent` is asked a real architectural question:

> *"Tell me in the current codebase where Authentication is implemented and what type of Authentication is used?"*

Here is what happens, step by step:

1. **Tool invocation** — The agent immediately fires the `retrieve_code_context` tool (events `#6` and `#7` in the Traces panel), searching the ChromaDB vector store for authentication-related code chunks.

2. **Grounded answer** — The agent identifies **JWT (JSON Web Token)** based authentication and cites three specific files retrieved from the index:
   - **`JwtTokenProvider.java`** — Core logic for generating, validating, and extracting JWT claims. Contains `generateToken`, `getAuthentication`, and `validateToken` methods.
   - **`JWTAuthorizationFilter.java`** — Spring Security filter that intercepts requests and validates JWTs using `JwtTokenProvider`.
   - **`WebSecurityConfig.java`** — Security chain configuration wiring the filter with `BCryptPasswordEncoder` and `httpBasic()`.

3. **Exact file paths returned** — The agent surfaces the full relative paths (`Hospital-Management-System-React-and-SpringBoot-master/backend/src/...`), not a generic description.

4. **Full transparency** — The ADK Events/Traces panel shows the complete tool call lifecycle, making the agent's reasoning auditable end-to-end.

This is **retrieval-grounded generation** working as designed: no hallucination, no guessing — only answers that can be traced back to code.

---

## Repository Structure

```text
CODEBASE-RAG/
├── Dockerfile
├── docker-compose.yml
├── req.txt
├── README.md
├── docs/
│   └── image.png                        # ADK UI live test screenshot
└── src/
    ├── ingestion/
    │   ├── repo_loader.py               # Directory traversal, FileDocument producer
    │   └── language_detector.py         # Extension → language mapping
    ├── chunking/
    │   ├── chunk_models.py              # Pydantic models: CodeChunk, ChunkManifest
    │   ├── chunker.py                   # ASTCodeChunker — astchunk integration
    │   ├── chunk_persist.py             # JSONL + manifest persistence
    │   └── chunk_main.py               # Pipeline entrypoint
    ├── embeddings/
    │   ├── embeddings_models.py         # Pydantic models: EmbeddingRecord, BatchStats
    │   ├── embeddings_formatter.py      # Builds embedding-ready text from chunk + metadata
    │   ├── embeddings_client.py         # Google embedding API wrapper
    │   ├── chroma_store.py             # ChromaDB collection management + batch upsert
    │   └── embeddings_main.py          # Pipeline entrypoint
    ├── retrieval/
    │   ├── retrieval_models.py          # Pydantic models: RetrievalResult, RetrievalResponse
    │   ├── query_embedder.py            # Converts query text to embedding vector
    │   ├── chroma_retriever.py          # Vector similarity search against ChromaDB
    │   └── retrieval_service.py         # Orchestrates query → retrieval → format
    ├── codebase_rag_agent/
    │   ├── tools.py                     # retrieve_code_context ADK tool definition
    │   └── agent.py                     # LlmAgent setup, instruction, tool registration
    ├── evaluation/
    │   ├── loader.py                    # Loads and validates evaluation dataset
    │   ├── ragas_eval.py                # RAGAS metric computation
    │   ├── run_eval.py                  # Evaluation entrypoint
    │   └── dataset.json                 # Ground-truth Q&A pairs
    ├── utils/
    │   ├── config.py                    # Environment variable loading (Pydantic Settings)
    │   ├── chunk_config.py              # Chunking tunable parameters
    │   └── file_utility.py             # Shared path and I/O helpers
    └── data/chunks/
        ├── code_chunks.jsonl            # Chunking output artifact
        └── chunk_manifest.json          # Chunk run metadata and stats
```

---

## Quickstart

### Option A — Local (Python venv)

```bash
git clone <your-repo-url>
cd CODEBASE-RAG
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r req.txt
```

Copy and populate the environment file:

```bash
cp .env.example .env               # edit with your keys (see Environment below)
```

Run the three pipeline stages in order:

```bash
# 1. Chunk your target codebase
python -m src.chunking.chunk_main

# 2. Embed chunks and index into ChromaDB
python -m src.embeddings.embeddings_main

# 3. Launch the ADK agent web server
adk web --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser — the ADK UI is ready.

---

### Option B — Docker Compose (recommended)

Set `HOST_REPO_PATH` in your `.env` to the absolute path of the repository you want to index, then run each stage:

```bash
# Stage 1: AST-chunk the target repository
docker compose run --rm chunk

# Stage 2: Embed chunks and persist to ChromaDB
docker compose run --rm embed

# Stage 3: Start the agent server (stays up)
docker compose up agent
```

The agent is served at `http://localhost:8000`.

---

## Environment Configuration

Create `.env` in the project root:

```env
# Google API
GOOGLE_API_KEY=your_google_api_key_here

# Docker: absolute path to the repo you want to index
HOST_REPO_PATH=/absolute/path/to/target/repo

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

## Pipeline Runbook

### A) Ingest and chunk a codebase

Set `repo_path` in `src/chunking/chunk_main.py` to your target repository, then:

```bash
python -m src.chunking.chunk_main
```

Output artifacts written to `src/data/chunks/`:

```
code_chunks.jsonl       ← one JSON object per chunk (content + metadata)
chunk_manifest.json     ← run stats: files scanned, chunks produced, languages
```

### B) Embed and index

```bash
python -m src.embeddings.embeddings_main
```

Reads `code_chunks.jsonl`, generates Google embeddings in configurable batches, and upserts into ChromaDB. Idempotent — re-running updates existing vectors.

### C) Retrieve programmatically

```python
from src.retrieval.retrieval_service import RetrievalService

service = RetrievalService()
context = service.retrieve_as_context(
    "Where is JWT authentication implemented?",
    top_k=5
)
print(context)
```

### D) Run evaluation

```bash
python -m src.evaluation.run_eval
```

Runs the RAGAS evaluation suite against `dataset.json` and prints metric scores.

---

## Deep Dive: Each Layer

### 1 — Ingestion

- Walks the repository tree and filters by supported file extensions.
- Skips build artifacts, dependency directories (`node_modules`, `__pycache__`, `.git`, etc.).
- Produces typed `FileDocument` objects with language, path, and raw content.
- Detects language from extension: Python, JS/TS, Java, C#, C/C++, Rust, Go, and more.

**Core files:** `src/ingestion/repo_loader.py`, `src/ingestion/language_detector.py`, `src/utils/config.py`

---

### 2 — AST Chunking

Naive token-window splitting destroys code semantics — a function split at line 50 loses its return statement and calling context. This layer uses `astchunk` to cut along syntax tree boundaries.

#### How it works

**Chunk size** — The chunker descends the AST from module → class → function, stopping as soon as a node fits within the configured size limit. Each chunk ideally represents a complete logical unit (a full function or class method). If the limit is too small, it degrades to finer-grained nodes like individual assignments or control statements — still syntactically valid, but less semantically rich.

**Chunk overlap** — After splitting, the chunker analyzes call dependencies. If chunk `[I]` calls function `[D]`, the chunk for `[I]` includes `[D]`'s signature or body as overlap. No chunk is isolated from the context it depends on.

**AST node taxonomy used:**

| Node | Represents |
|---|---|
| `[A]` | Module |
| `[B]` | Class definition (e.g., `OrderService`) |
| `[C]` | Method definition (e.g., `process_order`) |
| `[D]` | Function definition (e.g., `calculate_total`) |
| `[E]` | Assignment (`total = price * qty`) |
| `[F]` | Control statement (`if total > 100`) |
| `[G]` | Return statement |
| `[H]` | Variable assignment (`service = OrderService()`) |
| `[I]` | Function call (`service.calculate_total(...)`) |

**Result:** retrieval returns chunks that are both self-contained (chunk size) and contextually connected (overlap) — meaningfully better than fixed-window splitting for code Q&A.

**Core files:** `src/chunking/chunker.py`, `src/chunking/chunk_models.py`, `src/chunking/chunk_persist.py`, `src/utils/chunk_config.py`

---

### 3 — Embedding and Vectorization

- Formats each chunk into an embedding-ready string combining code content and metadata (language, file path, symbol name).
- Calls Google's `text-embedding-004` model in configurable batches.
- Upserts vectors and full metadata payloads into a persistent ChromaDB collection.
- Tracks batch-level operational counters for observability.

**Core files:** `src/embeddings/embeddings_formatter.py`, `src/embeddings/embeddings_client.py`, `src/embeddings/chroma_store.py`, `src/embeddings/embeddings_main.py`

---

### 4 — Retrieval

- Embeds the user's natural language query using the same Google embedding model.
- Executes cosine similarity search against the ChromaDB collection.
- Normalizes raw Chroma results into typed `RetrievalResult` / `RetrievalResponse` objects.
- Formats retrieved chunks into structured, LLM-ready context blocks with file path and line attribution.

**Core files:** `src/retrieval/query_embedder.py`, `src/retrieval/chroma_retriever.py`, `src/retrieval/retrieval_service.py`, `src/retrieval/retrieval_models.py`

---

### 5 — Agent Layer (Google ADK)

- Defines a single tool `retrieve_code_context` that wraps the `RetrievalService`.
- Registers the tool in a Google ADK `LlmAgent` backed by Gemini 2.0 Flash.
- Agent system instruction enforces **retrieve-before-generate** behavior — the model cannot answer without first invoking the retrieval tool.
- The ADK web UI provides Events and Traces panels for full tool-call transparency.

**Core files:** `src/codebase_rag_agent/tools.py`, `src/codebase_rag_agent/agent.py`

---

### 6 — Evaluation Layer (RAGAS)

Evaluation is built in from the start rather than bolted on later.

**Metrics computed:**

| Metric | What it measures |
|---|---|
| Faithfulness | Does the answer stay within the retrieved context? |
| Answer relevancy | How directly does the answer address the question? |
| Context precision | Are the retrieved chunks actually relevant to the question? |
| Context recall | Does the retrieved context cover the ground-truth answer? |

Add ground-truth Q&A pairs to `src/evaluation/dataset.json` and run `python -m src.evaluation.run_eval` to score the pipeline end-to-end.

**Core files:** `src/evaluation/loader.py`, `src/evaluation/ragas_eval.py`, `src/evaluation/run_eval.py`

---

## Engineering Decisions

**AST-first chunking over sliding windows** — Token windows split at arbitrary positions, destroying function bodies and class hierarchies. AST boundaries guarantee every chunk is a syntactically valid, semantically meaningful unit. Retrieval quality depends entirely on chunk quality.

**Pydantic models at every layer boundary** — Each layer (ingestion → chunking → embedding → retrieval) has its own typed models. Schema mismatches surface at object construction, not at generation time. Safer refactors, faster debugging.

**Metadata-enriched embeddings** — Embedding plain code content is table stakes. Including language, file path, and symbol name in the embedding text improves retrieval precision for cross-file and cross-language queries.

**Tool-based ADK agent** — Wrapping retrieval as a mandatory tool forces the LLM into a retrieve-then-generate pattern. Without this constraint, instruction-following alone is insufficient to prevent hallucination on large codebases.

**Evaluation from day one** — RAGAS metrics are set up before the pipeline is "done." This makes quality regressions from chunking parameter changes or embedding model swaps immediately visible.

**Docker-native pipeline** — Each stage (chunk, embed, agent) is a standalone Compose service. The target repository is mounted read-only. The ChromaDB data volume is shared across services. No state leaks between runs.
