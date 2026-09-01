"""Pydantic models for type-safe data structures."""

import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Query model for search requests."""

    text: str = Field(..., description="The search query text")
    library: str | None = Field(None, description="Filter by library name (e.g., 'pandas')")
    version: str | None = Field(None, description="Filter by library version")


class SearchResult(BaseModel):
    """Model for a single search result."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Fused relevance score (higher = better)")
    text: str = Field(..., description="Text snippet from the document")
    source_path: str = Field(..., description="Source file path")
    canonical_source_id: str = Field(
        ..., description="Canonical source identifier (relative POSIX path)"
    )
    header: str | None = Field(None, description="Section header (e.g., 'DataFrame.merge')")
    library: str | None = Field(
        None, description="Library name (optional but recommended for filtering/display)"
    )
    version: str | None = Field(None, description="Library version (optional but recommended)")
    title: str | None = Field(None, description="Document title (optional but recommended)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Catch-all for other fields")


class Citation(BaseModel):
    """Model for a citation reference in RAG responses."""

    label: str = Field(..., description="Citation label (e.g., '[1]')")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    canonical_source_id: str = Field(
        ..., description="Canonical source identifier (relative POSIX path)"
    )
    source_path: str = Field(..., description="Source file path")
    header: str | None = Field(None, description="Section header (e.g., 'DataFrame.merge')")
    title: str | None = Field(None, description="Document title")
    score: float | None = Field(None, description="Relevance score from search/rerank")


class RAGResponse(BaseModel):
    """Model for RAG-generated answer with citations."""

    answer: str = Field(..., description="Generated answer text")
    citations: list[Citation] = Field(..., description="List of citations referenced in the answer")
    context_used: str | None = Field(None, description="Formatted context used for generation")
    used_chunk_ids: list[str] = Field(..., description="List of chunk IDs used in the answer")


class DocumentChunk(BaseModel):
    """Model for a document chunk with embeddings."""

    chunk_id: str = Field(..., description="Stable hash-based chunk identifier")
    text: str = Field(..., description="Chunk text content")
    dense_vector: list[float] | None = Field(None, description="Dense embedding vector")
    sparse_vector: dict[int, float] | None = Field(None, description="Sparse embedding vector")
    metadata: dict[str, Any] = Field(
        ..., description="Metadata including source_path, header, title, library, version"
    )


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
