<div align="center">

# CODEBASE-RAG-AGENT

**Ask your codebase anything. Get grounded, evidence-backed answers.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=flat-square&logo=fastapi&logoColor=white)
![Google ADK](https://img.shields.io/badge/Google%20ADK-2.9.2-4285F4?style=flat-square&logo=google&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4.1%20%7C%20text--embedding--3--large-412991?style=flat-square&logo=openai&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.19-DC244C?style=flat-square)
![Celery](https://img.shields.io/badge/Celery-5.6-37814A?style=flat-square&logo=celery&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-LangCache-DD0031?style=flat-square&logo=redis&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![Pytest](https://img.shields.io/badge/pytest-passing-22c55e?style=flat-square&logo=pytest&logoColor=white)

[Overview](#overview) · [Architecture](#system-architecture) · [Agent Demo](#agent-in-action) · [Quickstart](#quickstart) · [Pipeline](#pipeline-runbook) · [Deep Dive](#deep-dive-each-layer) · [Evaluation](#6-evaluation-layer-ragas)

</div>

---

## Overview

Most code-search tools do text matching. This project builds a **semantic understanding layer** over an entire software repository using a production-grade RAG pipeline.

Point it at any codebase. It ingests every source file, chunks code along **AST boundaries** (not arbitrary token windows), embeds each chunk with Google's embedding model, and stores everything in **Qdrant**. A **Google ADK agent** then answers developer questions by first retrieving the most relevant code evidence — and only then generating a response.

Ask *"Where is JWT authentication implemented?"* and get back the exact files, classes, and methods — not a hallucinated description.

**Core capabilities at a glance:**

| Capability | Details |
|---|---|
| AST-aware chunking | Splits at function/class boundaries, preserves dependency overlap |
| Metadata-rich vectors | Language, file path, symbol name, line range stored alongside embeddings |
| Persistent vector store | Qdrant (HNSW, cosine): one collection, per-repo `repo_id` filtering, idempotent batch upserts |
| Grounded ADK agent | Tool-first architecture: `retrieve_code_context` fires before any generation |
| Full observability | LangSmith integration shows every tool call, LLM usage, etc lifecycle |
| RAGAS evaluation | Faithfulness, answer relevancy, context precision & recall |
| Docker-native | Three-service Compose pipeline: chunk → embed → agent |

---

## System Architecture Diagram


<img width="1117" height="897" alt="image" src="https://github.com/user-attachments/assets/e4b4798f-ab0b-4432-af6e-caecbd0975ef" />


---

## DEMO(Screenshot)

> Tested against a **Hospital Management System** codebase (React + Spring Boot) using the **Google ADK UI **.

<img width="1900" height="754" alt="image" src="https://github.com/user-attachments/assets/0da3e72b-5133-4c27-b152-4ac88037aa97" />


**What this screenshot demonstrates:**

The agent `codebase_rag_agent` is asked a real architectural question:

> *"Tell me in the current codebase where Authentication is implemented and what type of Authentication is used?"*

Here is what happens, step by step:

1. **Tool invocation** — The agent immediately fires the `retrieve_code_context` tool (events `#6` and `#7` in the Traces panel), searching the Qdrant vector store for authentication-related code chunks.

2. **Grounded answer** — The agent identifies **JWT (JSON Web Token)** based authentication and cites three specific files retrieved from the index:
   - **`JwtTokenProvider.java`** — Core logic for generating, validating, and extracting JWT claims. Contains `generateToken`, `getAuthentication`, and `validateToken` methods.
   - **`JWTAuthorizationFilter.java`** — Spring Security filter that intercepts requests and validates JWTs using `JwtTokenProvider`.
   - **`WebSecurityConfig.java`** — Security chain configuration wiring the filter with `BCryptPasswordEncoder` and `httpBasic()`.

3. **Exact file paths returned** — The agent surfaces the full relative paths (`Hospital-Management-System-React-and-SpringBoot-master/backend/src/...`), not a generic description.

4. **Full transparency** — The ADK Events/Traces panel shows the complete tool call lifecycle, making the agent's reasoning auditable end-to-end.

This is **retrieval-grounded generation** working as designed: no hallucination, no guessing — only answers that can be traced back to code.

---

## Repository Structure

> Updated 2026-11-16 to match the current codebase. `src/backend/` is the Python import root (imports are flat, e.g. `from utils.config import settings`).

```text
CODEBASE-RAG/
├── Dockerfile                           # Backend image (API + Celery worker), runs as non-root
├── docker-compose.yml                   # api :8000, celery_worker, frontend :5173
├── requirements.txt                     # Pinned runtime dependencies
├── requirements-dev.txt                 # pytest, ruff
├── ruff.toml / pytest.ini
├── scripts/                             # Local CLIs (not used by the API)
│   ├── chunk_main.py                    # Chunk a local repo to JSONL
│   ├── embeddings_main.py               # Embed JSONL chunks into Qdrant
│   ├── reingest_ready_repos.py          # Re-ingest every ready repo (dry run unless --apply)
│   └── locustfile.py                    # Load test
├── tests/                               # pytest suite (offline, no live services)
└── src/
    ├── backend/
    │   ├── api/
    │   │   ├── main.py                  # FastAPI app, CORS, middleware, logging
    │   │   ├── routes/v1/repo_routes.py # /health, /repos, /repos/{id}, /repos/{id}/chat
    │   │   ├── schemas/repo_schemas.py  # Request/response models, input length caps
    │   │   └── core/
    │   │       ├── dependencies.py      # RagWorkflowFacade (dedup, enqueue, cache-then-agent chat)
    │   │       ├── agent_runtime.py     # AgentRegistry (per-repo runners), LLMConcurrencyLimiter
    │   │       ├── middleware.py        # Request id, body size limit, generic 500
    │   │       ├── exceptions.py        # Domain errors -> HTTP status handlers
    │   │       └── rate_limit.py        # slowapi per-IP + global limits
    │   ├── worker/
    │   │   ├── celery_app.py            # Celery config (acks_late, prefetch 1)
    │   │   └── tasks.py                 # ingest_repo_task, task_failure handler
    │   ├── ingestion/
    │   │   ├── git_cloner.py            # GitHub size check, shallow clone, file/byte caps
    │   │   ├── repo_loader.py           # Filtered FileDocuments (skips symlinks)
    │   │   └── ingest_pipeline.py       # clone -> chunk -> embed -> upsert, timeouts, cleanup
    │   ├── chunking/
    │   │   ├── chunker.py               # ASTCodeChunker: chonkie CodeChunker in a subprocess pool
    │   │   └── chunk_config.py          # Chunk size 2,300, character tokenizer, batch 20
    │   ├── embeddings/
    │   │   ├── embeddings_formatter.py  # Embedding input text: location headers + code
    │   │   └── embeddings_client.py     # OpenAI embeddings with exponential backoff
    │   ├── storage/
    │   │   ├── qdrant_store.py          # Collection, HNSW config, tenant index, upsert, delete_stale
    │   │   ├── status_store.py          # RepoStatus + StatusStore ABC (in-memory)
    │   │   └── redis_status_store.py    # Production status store (atomic SET NX, CAS on attempt_id)
    │   ├── retrieval/
    │   │   ├── qdrant_retriever.py      # repo_id-filtered search (hnsw_ef)
    │   │   └── retrieval_service.py     # Embed query -> search -> format context
    │   ├── caching/
    │   │   ├── semantic_cache_service.py    # Lookup/store policy, follow-up keys, normalize_query
    │   │   ├── semantic_cache_repository.py # LangCacheRepository
    │   │   └── pii_detector.py              # PII/credential gate
    │   ├── codebase_rag_agent/
    │   │   ├── agent.py                 # build_agent: ADK LlmAgent + instructions + callbacks
    │   │   ├── tools.py                 # retrieve_code_context tool
    │   │   └── tracing.py               # LangSmith tracing (redacted)
    │   ├── models/                      # Pydantic models shared across layers
    │   │   ├── ingestion_models.py      # FileDocument, IngestionStats, IngestionResult
    │   │   ├── chunk_models.py          # ChunkMetadata, CodeChunk, ChunkingStats, ChunkingResult
    │   │   ├── embeddings_models.py     # EmbeddedChunk
    │   │   └── retrieval_models.py      # RetrievedChunk, RetrievalResponse
    │   ├── evaluation/
    │   │   ├── run_eval.py              # Entry point: python -m evaluation.run_eval
    │   │   ├── ragas_eval.py            # RAGAS metrics on live agent answers
    │   │   ├── metrics.py               # Recall@K, latency P50/P95/P99, cost per query
    │   │   ├── loader.py                # Dataset loading + validation
    │   │   └── dataset.json             # 15 labeled Q&A records
    │   └── utils/
    │       ├── config.py                # Settings (pydantic-settings), language table, filters
    │       ├── file_utility.py          # is_supported_file, detect_language, safe reads
    │       ├── rate_limiter.py          # OpenAIRequestBudget, ExponentialBackoff
    │       ├── model_config.py          # LiteLLM model factory
    │       ├── logging_config.py        # Structured logs with request_id
    │       ├── repo_id.py               # normalize_github_url, repo_id = sha1(url)[:12]
    │       ├── iter_utils.py / time_utils.py
    └── frontend/                        # React + Vite + TypeScript
        └── src/
            ├── api.ts                   # The only API client
            ├── App.tsx, types.ts
            └── components/              # IngestForm, StatusPanel, ChatPanel
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

# 2. Embed chunks and index into Qdrant
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

# Stage 2: Embed chunks and persist to Qdrant
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

Reads `code_chunks.jsonl`, generates Google embeddings in configurable batches, and upserts into Qdrant. Idempotent — re-running updates existing vectors.

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

### 1 — Repository Ingestion

- **Validate and identify**: Accepts public GitHub repository URLs and creates a stable repo_id from the normalized URL.
- **Cloning**: The latest version of github repo is cloned.
- **Controls Processing Size**: Checks the Github repository size before cloning(max 1 GB) then limits processing to 5000 files, 2 MB/file, 20,000 chunks. These limits controlled **processing time** and **embedding costs**.
- Filtering: Filters source files, skips dependency files like .env, node_modules, lockfiles, binaries, and other executable files.
- Status Tracking: Tracks cloning, chunking and embedding progress.

**Core files:** `ingestion/git_cloner.py`, `ingestion/repo_loader.py`, `ingestion/ingest_pipeline.py`

---

### 2 — AST Chunking

<img width="869" height="745" alt="image" src="https://github.com/user-attachments/assets/bda84470-e5f9-4a94-8a57-70a44f31bb51" />

Naive token-window splitting destroys code semantics — a function split at a random line loses its actual context. This layer uses `astchunk` to cut along syntax tree boundaries.

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

Each chunk is converted into an embedding, a vector that lets the system find code related to a question by meaning, not only by exact words.

- **Add source context:** The embedding input is `File:`, `Path:`, `Language:` and `Lines:` headers followed by the code.
- **Generate embeddings:** OpenAI `text-embedding-3-large` (3,072 dimensions), in batches of 20. Rate-limit (429) errors are retried with exponential backoff, up to 5 retries, and a shared OpenAI RPM budget in Redis can be enabled with `OPENAI_RPM_LIMIT`.
- **Store in Qdrant:** Saves each vector with its code and metadata in one collection configured for cosine similarity and an HNSW index (`m=16`, `ef_construct=200`).
- **Separate repositories:** All repositories share one collection; `repo_id` has a tenant payload index and every search filters on it.
- **Avoid duplicate points:** The point ID is a deterministic UUIDv5 of `repo_id:chunk_id`. Each run tags its points with an `ingest_run_id`; the repository's points from older runs are deleted only **after** the new run's last batch is stored, so a failed re-ingest leaves the previous index intact.

**Core files:** `embeddings/embeddings_formatter.py`, `embeddings/embeddings_client.py`, `storage/qdrant_store.py`, `models/embeddings_models.py`

---

### 4 — Retrieval

1. Converts the question into an embedding with the same model used for the code.
2. Searches Qdrant by cosine similarity for the top `TOP_K` chunks (`hnsw_ef=128`).
3. Filters by `repo_id`, so only the selected repository is searched. A missing `repo_id` raises an error instead of searching every repo.
4. Returns ranked chunks with file paths, line ranges and similarity scores.
5. Formats them into `[Chunk N]` context blocks for the agent.

The current implementation uses dense vector search only; BM25 keyword search, hybrid search and a reranking model are not implemented.

**Core files:** `retrieval/retrieval_service.py`, `retrieval/qdrant_retriever.py`, `models/retrieval_models.py`

---

### 5 — Agent Layer (Google ADK)

- Defines a single tool `retrieve_code_context` that wraps the `RetrievalService`.
- Registers the tool in a Google ADK `LlmAgent` backed by gpt-4.1.
- Agent system instruction enforces **retrieve-before-generate** behavior — the model cannot answer without first invoking the retrieval tool.

**Core files:** `codebase_rag_agent/agent.py`, `codebase_rag_agent/tools.py`, `codebase_rag_agent/tracing.py`, `api/core/agent_runtime.py`

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

**1. AST-first chunking over sliding windows** — Token windows split at arbitrary positions, destroying function bodies and class hierarchies. AST boundaries guarantee every chunk is a syntactically valid, semantically meaningful unit. Retrieval quality depends entirely on chunk quality.

**2. Pydantic models at every layer boundary** — Each layer (ingestion → chunking → embedding → retrieval) has its own typed models. Schema mismatches surface at object construction, not at generation time. Safer refactors, faster debugging.

**3. Metadata-enriched embeddings** — Embedding plain code content is table stakes. Including language, file path, and symbol name in the embedding text improves retrieval precision for cross-file and cross-language queries.

**4. Tool-based ADK agent** — Wrapping retrieval as a mandatory tool forces the LLM into a retrieve-then-generate pattern. Without this constraint, instruction-following alone is insufficient to prevent hallucination on large codebases.

**5. Docker-native pipeline** — Each stage (chunk, embed, agent) is a standalone Compose service. The target repository is mounted read-only. The Qdrant data volume is shared across services. No state leaks between runs.
