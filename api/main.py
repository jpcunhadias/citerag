"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import collections, health, rag

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
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

