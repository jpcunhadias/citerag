"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import collections, health, rag
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Documentation Search API",
    description="REST API for RAG-based documentation search system",
    version="1.0.0",
)

# Configure CORS for Streamlit client
# CORS origins are configured via CORS_ORIGINS environment variable
# Default: http://localhost:8501 (for local Streamlit development)
# Production: Set to specific origins, e.g., "https://app.example.com,https://admin.example.com"
allowed_origins = [origin.strip() for origin in config.CORS_ORIGINS.split(",") if origin.strip()]

# Ensure at least one origin is configured
if not allowed_origins:
    logger.warning(f"No CORS origins configured. Using default: {config.DEFAULT_CORS_ORIGIN}")
    allowed_origins = [config.DEFAULT_CORS_ORIGIN]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(collections.router)
app.include_router(rag.router)


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize services on startup."""
    logger.info("FastAPI application starting up")
    logger.info("Services will be initialized lazily on first request")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup on shutdown."""
    logger.info("FastAPI application shutting down")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "message": "RAG Documentation Search API",
        "docs": "/docs",
        "health": "/health",
    }
