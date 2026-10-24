"""FastAPI application: middleware, error handlers and routes."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.config import settings
from utils.logging_config import configure_logging

from .core.body_limit import BodySizeLimitMiddleware
from .core.exceptions import register_exception_handlers
from .core.rate_limit import limiter
from .routes.v1.status_routes import router

configure_logging()

if not settings.agent_model:
    raise RuntimeError("AGENT_MODEL must be set to run the API")

app = FastAPI(
    title="Codebase RAG API",
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
)
app.state.limiter = limiter

app.add_middleware(
    BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
