"""Pydantic models for FastAPI request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field

from src.models import Citation, SearchResult


class AskRequest(BaseModel):
    """Request model for RAG ask endpoint."""

    query: str = Field(..., description="User query/question")
    collection: str = Field(..., description="Qdrant collection name")
    top_k: int = Field(25, ge=1, le=100, description="Number of initial search results")
    top_n: int = Field(5, ge=1, le=10, description="Number of results to rerank and use for context")
    rerank: bool = Field(True, description="Whether to apply reranking")
    debug: bool = Field(False, description="Whether to include context_used in response")


class AskResponse(BaseModel):
    """Response model for RAG ask endpoint."""

    answer: str = Field(..., description="Generated answer text")
    citations: list[Citation] = Field(..., description="List of citations referenced in the answer")
    context_used: Optional[str] = Field(None, description="Formatted context used for generation (only if debug=True)")
    used_chunk_ids: list[str] = Field(..., description="List of chunk IDs used in the answer")


class SearchRequest(BaseModel):
    """Request model for search endpoint."""

    query: str = Field(..., description="Search query text")
    collection: str = Field(..., description="Qdrant collection name")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    filters: Optional[dict[str, str]] = Field(None, description="Optional filters (e.g., {'library': 'pandas', 'version': '2.0'})")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    results: list[SearchResult] = Field(..., description="List of search results")


class CollectionsResponse(BaseModel):
    """Response model for collections endpoint."""

    collections: list[str] = Field(..., description="List of available collection names")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="Overall health status")
    services: dict[str, str] = Field(..., description="Status of individual services (qdrant, ollama)")

