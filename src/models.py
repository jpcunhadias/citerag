"""Pydantic models for type-safe data structures."""

from typing import Optional

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Query model for search requests."""

    text: str = Field(..., description="The search query text")
    library: Optional[str] = Field(None, description="Filter by library name (e.g., 'pandas')")
    version: Optional[str] = Field(None, description="Filter by library version")


class SearchResult(BaseModel):
    """Model for a single search result."""

    score: float = Field(..., description="Relevance score")
    text: str = Field(..., description="Text snippet from the document")
    url: Optional[str] = Field(None, description="Source URL if available")
    file_path: Optional[str] = Field(None, description="Local file path if available")
    header: Optional[str] = Field(None, description="Section header (e.g., 'DataFrame.merge')")
    library: Optional[str] = Field(None, description="Library name")
    version: Optional[str] = Field(None, description="Library version")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")

