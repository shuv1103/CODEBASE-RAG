FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

COPY src/ ./src/

RUN mkdir -p /app/src/data/chunks

WORKDIR /app/src

CMD ["adk", "web", "--host", "0.0.0.0", "--port", "8000"]
