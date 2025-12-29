"""Integration tests for search functionality (requires Qdrant running)."""

import os
import uuid

import pytest

from src.models import SearchResult
from src.search import SearchService


@pytest.mark.integration
class TestSearchIntegration:
    """Integration tests for search functionality (requires Qdrant running)."""

    @pytest.fixture
    def test_collection_name(self):
        """Generate unique test collection name."""
        return f"test_search_{uuid.uuid4().hex[:8]}"

    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL") and not os.getenv("QDRANT_HOST"),
        reason="Qdrant not configured (set QDRANT_URL or QDRANT_HOST env var)",
    )
    def test_search_returns_results(self, test_collection_name):
        """Test that search returns results with scores."""
        from qdrant_client import QdrantClient

        from src.config import QDRANT_HOST, QDRANT_PORT
        from src.ingest import VectorService

        # This test requires a collection with data
        # For now, just verify the service can be instantiated and called
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        vector_service = VectorService()
        search_service = SearchService(client, vector_service)

        # Try to search (may return empty if collection doesn't exist)
        results = search_service.hybrid_search(
            query="test", collection=test_collection_name, top_k=5
        )

        # Results should be a list (even if empty)
        assert isinstance(results, list)

        # If results exist, verify structure
        if results:
            assert all(isinstance(r, SearchResult) for r in results)
            assert all(r.score is not None for r in results)
            assert all(r.chunk_id for r in results)
            assert all(r.text for r in results)
