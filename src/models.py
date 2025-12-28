"""Pydantic models for type-safe data structures."""

import hashlib
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.config import CHUNKER_FINGERPRINT


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


class DocumentChunk(BaseModel):
    """Model for a document chunk with embeddings."""

    chunk_id: str = Field(..., description="Stable hash-based chunk identifier")
    text: str = Field(..., description="Chunk text content")
    dense_vector: Optional[list[float]] = Field(None, description="Dense embedding vector")
    sparse_vector: Optional[dict[int, float]] = Field(None, description="Sparse embedding vector")
    metadata: dict[str, Any] = Field(..., description="Metadata including source_path, header, title, library, version")


def normalize_text_for_hashing(text: str) -> str:
    """
    Normalize text deterministically for chunk ID generation.

    Args:
        text: Raw text to normalize.

    Returns:
        Normalized text with consistent formatting.
    """
    # Convert \r\n to \n and trim trailing spaces from each line
    text = text.replace("\r\n", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text_no_trailing_space = "\n".join(lines)

    # Collapse excessive blank lines (3+ newlines -> 2 newlines)
    return re.sub(r"\n{3,}", "\n\n", text_no_trailing_space)


def generate_chunk_id(
    canonical_source_id: str,
    chunker_fingerprint: str,
    normalized_chunk_text: str,
) -> str:
    """
    Generate a stable chunk ID based on source, chunker config, and normalized text.

    Args:
        canonical_source_id: Relative POSIX path from input root.
        chunker_fingerprint: Chunker fingerprint from config.
        normalized_chunk_text: Text normalized deterministically.

    Returns:
        Hex digest string of SHA256 hash.
    """
    # Ensure text is normalized
    normalized_text = normalize_text_for_hashing(normalized_chunk_text)
    # Create hash input
    hash_input = f"{canonical_source_id}|{chunker_fingerprint}|{normalized_text}"
    # Generate SHA256 hash
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

