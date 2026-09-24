import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.logging_config import configure_logging

from .exceptions import register_exception_handlers
from .routes import router

load_dotenv()

configure_logging()

app = FastAPI(title="Codebase RAG API")

# Fail closed: with API_CORS_ORIGINS unset, only the local dev frontend may
# call the API — never "*". Production sets it to the deployed frontend's
# origin (comma-separated for several).
cors_origins = [
    origin.strip()
    for origin in os.getenv("API_CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
