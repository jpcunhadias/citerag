"""Unit tests for Pydantic models (Citation, RAGResponse)."""

import pytest

from src.models import Citation, RAGResponse


class TestCitation:
    """Tests for Citation model."""

    def test_citation_with_all_fields(self):
        """Test creating Citation with all fields."""
        citation = Citation(
            label="1",
            chunk_id="chunk123",
            canonical_source_id="docs/api.md",
            source_path="/path/to/docs/api.md",
            header="API Reference",
            title="API Documentation",
            score=0.95,
        )

        assert citation.label == "1"
        assert citation.chunk_id == "chunk123"
        assert citation.canonical_source_id == "docs/api.md"
        assert citation.source_path == "/path/to/docs/api.md"
        assert citation.header == "API Reference"
        assert citation.title == "API Documentation"
        assert citation.score == 0.95

    def test_citation_with_optional_fields_none(self):
        """Test creating Citation with optional fields as None."""
        citation = Citation(
            label="2",
            chunk_id="chunk456",
            canonical_source_id="docs/intro.md",
            source_path="/path/to/docs/intro.md",
            header=None,
            title=None,
            score=None,
        )

        assert citation.label == "2"
        assert citation.chunk_id == "chunk456"
        assert citation.header is None
        assert citation.title is None
        assert citation.score is None

    def test_citation_without_optional_fields(self):
        """Test creating Citation without optional fields."""
        citation = Citation(
            label="3",
            chunk_id="chunk789",
            canonical_source_id="docs/guide.md",
            source_path="/path/to/docs/guide.md",
        )

        assert citation.label == "3"
        assert citation.chunk_id == "chunk789"
        assert citation.header is None
        assert citation.title is None
        assert citation.score is None

    def test_citation_score_field(self):
        """Test Citation score field accepts float values."""
        citation = Citation(
            label="4",
            chunk_id="chunk999",
            canonical_source_id="test.md",
            source_path="test.md",
            score=-10.5,  # Negative scores are valid (reranker scores)
        )

        assert citation.score == -10.5


class TestRAGResponse:
    """Tests for RAGResponse model."""

    def test_rag_response_with_all_fields(self):
        """Test creating RAGResponse with all fields."""
        citations = [
            Citation(
                label="1",
                chunk_id="chunk1",
                canonical_source_id="doc1.md",
                source_path="doc1.md",
            ),
            Citation(
                label="2",
                chunk_id="chunk2",
                canonical_source_id="doc2.md",
                source_path="doc2.md",
            ),
        ]

        response = RAGResponse(
            answer="This is the answer with [1] citation.",
            citations=citations,
            context_used="[1] Some context...",
            used_chunk_ids=["chunk1", "chunk2"],
        )

        assert response.answer == "This is the answer with [1] citation."
        assert len(response.citations) == 2
        assert response.context_used == "[1] Some context..."
        assert response.used_chunk_ids == ["chunk1", "chunk2"]

    def test_rag_response_without_context_used(self):
        """Test creating RAGResponse without context_used."""
        citations = [
            Citation(
                label="1",
                chunk_id="chunk1",
                canonical_source_id="doc1.md",
                source_path="doc1.md",
            )
        ]

        response = RAGResponse(
            answer="Answer",
            citations=citations,
            used_chunk_ids=["chunk1"],
        )

        assert response.answer == "Answer"
        assert response.context_used is None
        assert len(response.citations) == 1

    def test_rag_response_with_empty_citations(self):
        """Test creating RAGResponse with empty citations."""
        response = RAGResponse(
            answer="I couldn't find this in the indexed documentation.",
            citations=[],
            used_chunk_ids=[],
        )

        assert len(response.citations) == 0
        assert len(response.used_chunk_ids) == 0

    def test_rag_response_citations_match_chunk_ids(self):
        """Test that citations and used_chunk_ids are consistent."""
        citations = [
            Citation(
                label="1",
                chunk_id="chunk1",
                canonical_source_id="doc1.md",
                source_path="doc1.md",
            ),
            Citation(
                label="2",
                chunk_id="chunk2",
                canonical_source_id="doc2.md",
                source_path="doc2.md",
            ),
        ]

        response = RAGResponse(
            answer="Answer",
            citations=citations,
            used_chunk_ids=["chunk1", "chunk2"],
        )

        # Verify consistency
        assert len(response.citations) == len(response.used_chunk_ids)
        citation_chunk_ids = [c.chunk_id for c in response.citations]
        assert set(citation_chunk_ids) == set(response.used_chunk_ids)

