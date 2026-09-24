# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY src/ ./src/

# Run as an unprivileged user, not root. The code stays root-owned (read-only
# for the app); only src/data/ — the one directory the app writes to, besides
# /tmp for repo clones — belongs to the app user.
RUN groupadd --system app \
    && useradd --system --gid app --create-home --home-dir /home/app app \
    && mkdir -p /app/src/data/chunks \
    && chown -R app:app /app/src/data

WORKDIR /app/src

USER app

# chonkie's CodeChunker downloads every tree-sitter grammar (~500 MB, about a
# minute) the first time it runs, into the running user's ~/.cache. Do it
# once at build time so each new container (every deploy / restart) starts
# with the grammars baked in instead of stalling its first ingestion.
RUN python -c "import tree_sitter_language_pack as t; t.download_all()"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
