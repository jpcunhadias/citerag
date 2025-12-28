"""Integration tests for the full RAG pipeline."""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.llm import OllamaClient
from src.rag import RAGService
from src.rerank import RerankerService
from src.search import SearchService


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for the full RAG pipeline (requires Qdrant and Ollama)."""

    @pytest.fixture
    def test_collection_name(self):
        """Generate unique test collection name."""
        return f"test_rag_{uuid.uuid4().hex[:8]}"

    @pytest.fixture
    def qdrant_available(self):
        """Check if Qdrant is available."""
        try:
            from qdrant_client import QdrantClient

            from src.config import QDRANT_HOST, QDRANT_PORT

            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            return True
        except Exception:
            return False

    @pytest.fixture
    def ollama_available(self):
        """Check if Ollama is available."""
        try:
            import requests

            from src.config import OLLAMA_BASE_URL

            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL") and not os.getenv("QDRANT_HOST"),
        reason="Qdrant not configured (set QDRANT_URL or QDRANT_HOST env var)",
    )
    def test_rag_pipeline_with_mocked_llm(
        self, test_collection_name, qdrant_available
    ):
        """Test full RAG pipeline with mocked LLM (requires Qdrant)."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from qdrant_client import QdrantClient

        from src.config import QDRANT_HOST, QDRANT_PORT
        from src.ingest import VectorService

        # Initialize real services (except LLM)
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        vector_service = VectorService()
        search_service = SearchService(qdrant_client, vector_service)
        reranker_service = RerankerService()

        # Mock LLM client
        mock_llm = MagicMock(spec=OllamaClient)
        mock_llm.generate.return_value = "This is a test answer with [1] citation."

        rag_service = RAGService(
            search_service=search_service,
            reranker_service=reranker_service,
            llm_client=mock_llm,
        )

        # Try to search (may return empty if collection doesn't exist)
        # This tests the pipeline structure even if collection is empty
        try:
            response = rag_service.ask(
                query="test query",
                collection=test_collection_name,
                top_k=5,
                top_n=3,
                rerank=True,
                debug=False,
            )

            # Verify response structure
            assert isinstance(response.answer, str)
            assert isinstance(response.citations, list)
            assert isinstance(response.used_chunk_ids, list)

            # If we got results, verify structure
            if response.citations:
                assert len(response.citations) == len(response.used_chunk_ids)
                for citation in response.citations:
                    assert citation.label is not None
                    assert citation.chunk_id is not None
                    assert citation.score is not None

        except Exception as e:
            # If collection doesn't exist, that's okay for integration test
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                pytest.skip(f"Collection {test_collection_name} does not exist")
            else:
                raise

    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL") and not os.getenv("QDRANT_HOST"),
        reason="Qdrant not configured",
    )
    def test_rag_pipeline_empty_context_returns_refusal(
        self, test_collection_name, qdrant_available
    ):
        """Test that RAG pipeline returns refusal when no context is found."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")

        from qdrant_client import QdrantClient

        from src.config import QDRANT_HOST, QDRANT_PORT
        from src.ingest import VectorService

        # Initialize real services
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        vector_service = VectorService()
        search_service = SearchService(qdrant_client, vector_service)
        reranker_service = RerankerService()

        # Mock LLM (should not be called)
        mock_llm = MagicMock(spec=OllamaClient)

        rag_service = RAGService(
            search_service=search_service,
            reranker_service=reranker_service,
            llm_client=mock_llm,
        )

        # Query that won't match anything
        response = rag_service.ask(
            query="xyzabc123nonexistentquery",
            collection=test_collection_name,
            top_k=5,
            top_n=3,
            rerank=True,
            debug=False,
        )

        # Should return refusal without calling LLM
        assert (
            response.answer == "I couldn't find this in the indexed documentation."
        )
        assert len(response.citations) == 0
        assert len(response.used_chunk_ids) == 0
        mock_llm.generate.assert_not_called()

    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL") and not os.getenv("QDRANT_HOST"),
        reason="Qdrant not configured",
    )
    @pytest.mark.skipif(
        not os.getenv("OLLAMA_AVAILABLE"),
        reason="Ollama not available (set OLLAMA_AVAILABLE=1 to enable)",
    )
    def test_rag_pipeline_full_integration(
        self, test_collection_name, qdrant_available, ollama_available
    ):
        """Test full RAG pipeline with real services (requires Qdrant and Ollama)."""
        if not qdrant_available:
            pytest.skip("Qdrant not available")
        if not ollama_available:
            pytest.skip("Ollama not available")

        from qdrant_client import QdrantClient

        from src.config import QDRANT_HOST, QDRANT_PORT
        from src.ingest import VectorService
        from src.llm import OllamaClient

        # Initialize all real services
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        vector_service = VectorService()
        search_service = SearchService(qdrant_client, vector_service)
        reranker_service = RerankerService()
        llm_client = OllamaClient()

        rag_service = RAGService(
            search_service=search_service,
            reranker_service=reranker_service,
            llm_client=llm_client,
        )

        # Try with a known collection (e.g., smoke_docs)
        known_collections = ["smoke_docs", "pandas_docs"]
        collection_to_use = None

        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        for coll in known_collections:
            if coll in collection_names:
                collection_to_use = coll
                break

        if not collection_to_use:
            pytest.skip("No known test collection available")

        # Execute RAG pipeline
        response = rag_service.ask(
            query="merge",
            collection=collection_to_use,
            top_k=10,
            top_n=3,
            rerank=True,
            debug=False,
        )

        # Verify response structure
        assert isinstance(response.answer, str)
        assert len(response.answer) > 0
        assert isinstance(response.citations, list)
        assert isinstance(response.used_chunk_ids, list)

        # If we got citations, verify they're consistent
        if response.citations:
            assert len(response.citations) == len(response.used_chunk_ids)
            for citation in response.citations:
                assert citation.label is not None
                assert citation.chunk_id is not None
                assert citation.score is not None
                assert citation.chunk_id in response.used_chunk_ids

    def test_rag_pipeline_budget_enforcement(self):
        """Test that RAG pipeline enforces context budget."""
        from src.rag import RAGService
        from src.models import SearchResult

        # Create mock services
        mock_search = MagicMock()
        mock_reranker = MagicMock()
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Answer [1]"

        rag_service = RAGService(mock_search, mock_reranker, mock_llm)

        # Create chunks that exceed budget
        with patch("src.rag.RAG_MAX_CONTEXT_CHARS", 200):
            chunks = [
                SearchResult(
                    chunk_id=f"chunk{i}",
                    score=0.9,
                    text="x" * 100,  # Each chunk is ~100 chars
                    source_path="doc.md",
                    canonical_source_id="doc.md",
                )
                for i in range(5)
            ]

            mock_search.hybrid_search.return_value = chunks

            response = rag_service.ask(
                query="test",
                collection="test_collection",
                top_k=5,
                top_n=5,
                rerank=False,
                debug=False,
            )

            # Should only include chunks that fit in budget
            # With budget of 200, should fit ~2 chunks (each is ~100 + "[1] " + "\n\n")
            assert len(response.citations) <= 3  # Allow some margin
            assert len(response.used_chunk_ids) == len(response.citations)

    def test_rag_pipeline_citation_compliance(self):
        """Test that RAG pipeline enforces citation compliance."""
        from src.rag import RAGService
        from src.models import SearchResult

        # Create mock services
        mock_search = MagicMock()
        mock_reranker = MagicMock()
        mock_llm = MagicMock()

        rag_service = RAGService(mock_search, mock_reranker, mock_llm)

        chunks = [
            SearchResult(
                chunk_id="chunk1",
                score=0.9,
                text="Test text",
                source_path="doc.md",
                canonical_source_id="doc.md",
            )
        ]

        mock_search.hybrid_search.return_value = chunks

        # Test 1: Answer without citations should be replaced
        mock_llm.generate.return_value = "This answer has no citations."
        response = rag_service.ask(
            query="test",
            collection="test_collection",
            top_k=5,
            top_n=3,
            rerank=False,
            debug=False,
        )
        assert (
            response.answer == "I couldn't find this in the indexed documentation."
        )

        # Test 2: Answer with citations should be kept
        mock_llm.generate.return_value = "This answer has [1] citation."
        response = rag_service.ask(
            query="test",
            collection="test_collection",
            top_k=5,
            top_n=3,
            rerank=False,
            debug=False,
        )
        assert "[1]" in response.answer
        assert (
            response.answer != "I couldn't find this in the indexed documentation."
        )

        # Test 3: Exact refusal string should be kept
        refusal = "I couldn't find this in the indexed documentation."
        mock_llm.generate.return_value = refusal
        response = rag_service.ask(
            query="test",
            collection="test_collection",
            top_k=5,
            top_n=3,
            rerank=False,
            debug=False,
        )
        assert response.answer == refusal

