"""FastAPI application: middleware, error handlers and routes."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.core.exceptions import register_exception_handlers
from api.core.middleware import (
    BodySizeLimitMiddleware,
    RequestIdMiddleware,
    UnhandledErrorMiddleware,
)
from api.core.rate_limit import limiter
from api.routes.v1.repo_routes import router
from utils.config import settings
from utils.logging_config import configure_logging

configure_logging(service="api")

if not settings.agent_model:
    raise RuntimeError("AGENT_MODEL must be set to run the API")

app = FastAPI(
    title="Codebase RAG API",
    docs_url="/docs" if settings.api_docs_enabled else None,
    redoc_url="/redoc" if settings.api_docs_enabled else None,
    openapi_url="/openapi.json" if settings.api_docs_enabled else None,
)
app.state.limiter = limiter

# Added first so it is innermost: a 500 still gets CORS and request-id headers.
app.add_middleware(UnhandledErrorMiddleware)
app.add_middleware(
    BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Added last so it is outermost: even 413s and CORS rejections get an id.
app.add_middleware(RequestIdMiddleware)

register_exception_handlers(app)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
