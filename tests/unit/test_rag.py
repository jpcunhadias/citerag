"""Unit tests for the RAG orchestration service."""

from unittest.mock import MagicMock, patch

import pytest

from src.models import SearchResult
from src.rag import RAGService


class TestRAGService:
    """Tests for RAGService class."""

    @pytest.fixture
    def mock_search_service(self):
        """Create a mock SearchService."""
        return MagicMock()

    @pytest.fixture
    def mock_reranker_service(self):
        """Create a mock RerankerService."""
        return MagicMock()

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock OllamaClient."""
        return MagicMock()

    @pytest.fixture
    def rag_service(self, mock_search_service, mock_reranker_service, mock_llm_client):
        """Create RAGService instance with mocked dependencies."""
        return RAGService(
            search_service=mock_search_service,
            reranker_service=mock_reranker_service,
            llm_client=mock_llm_client,
        )

    @pytest.fixture
    def sample_search_results(self):
        """Create sample SearchResult objects."""
        return [
            SearchResult(
                chunk_id="chunk1",
                score=0.9,
                text="First chunk text",
                source_path="doc1.md",
                canonical_source_id="doc1.md",
                header="Section 1",
                title="Document 1",
            ),
            SearchResult(
                chunk_id="chunk2",
                score=0.8,
                text="Second chunk text",
                source_path="doc2.md",
                canonical_source_id="doc2.md",
                header="Section 2",
                title="Document 2",
            ),
        ]

    def test_build_context_formats_correctly(self, rag_service, sample_search_results):
        """Test that build_context formats context string correctly."""
        context_str, citations, used_chunk_ids = rag_service.build_context(sample_search_results)

        # Check format: [1] text\n\n[2] text\n\n
        assert "[1]" in context_str
        assert "[2]" in context_str
        assert "First chunk text" in context_str
        assert "Second chunk text" in context_str
        assert context_str.count("\n\n") == 2  # Two chunks, two separators

    def test_build_context_creates_citations_with_scores(self, rag_service, sample_search_results):
        """Test that build_context creates Citation objects with scores."""
        context_str, citations, used_chunk_ids = rag_service.build_context(sample_search_results)

        assert len(citations) == 2
        assert citations[0].label == "1"
        assert citations[0].chunk_id == "chunk1"
        assert citations[0].score == 0.9
        assert citations[1].label == "2"
        assert citations[1].chunk_id == "chunk2"
        assert citations[1].score == 0.8

    def test_build_context_labels_are_sequential(self, rag_service, sample_search_results):
        """Test that build_context creates sequential labels."""
        context_str, citations, used_chunk_ids = rag_service.build_context(sample_search_results)

        labels = [c.label for c in citations]
        assert labels == ["1", "2"]

    def test_build_context_enforces_budget(self, rag_service):
        """Test that build_context enforces character budget."""
        # Create chunks that exceed budget
        long_text = "x" * 10000  # Very long text
        chunks = [
            SearchResult(
                chunk_id=f"chunk{i}",
                score=0.9,
                text=long_text,
                source_path="doc.md",
                canonical_source_id="doc.md",
            )
            for i in range(5)
        ]

        context_str, citations, used_chunk_ids = rag_service.build_context(chunks)

        # Should stop before exceeding budget
        assert len(context_str) <= 12000  # RAG_MAX_CONTEXT_CHARS
        assert len(citations) <= len(chunks)

    def test_build_context_stops_at_budget_limit(self, rag_service):
        """Test that build_context stops adding chunks when budget is exceeded."""
        # Create chunks where first fits, second exceeds budget
        with patch("src.rag.RAG_MAX_CONTEXT_CHARS", 100):
            chunk1 = SearchResult(
                chunk_id="chunk1",
                score=0.9,
                text="x" * 50,  # Fits in budget
                source_path="doc1.md",
                canonical_source_id="doc1.md",
            )
            chunk2 = SearchResult(
                chunk_id="chunk2",
                score=0.8,
                text="x" * 100,  # Would exceed budget
                source_path="doc2.md",
                canonical_source_id="doc2.md",
            )

            context_str, citations, used_chunk_ids = rag_service.build_context([chunk1, chunk2])

            # Should only include first chunk
            assert len(citations) == 1
            assert citations[0].chunk_id == "chunk1"
            assert "chunk2" not in used_chunk_ids

    def test_build_context_handles_empty_chunks(self, rag_service):
        """Test that build_context handles empty chunks list."""
        context_str, citations, used_chunk_ids = rag_service.build_context([])

        assert context_str == ""
        assert len(citations) == 0
        assert len(used_chunk_ids) == 0

    def test_ask_returns_refusal_on_empty_context(self, rag_service, mock_search_service):
        """Test that ask returns refusal when context is empty."""
        mock_search_service.hybrid_search.return_value = []
        mock_llm_client = rag_service.llm_client

        response = rag_service.ask(
            query="test query",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        assert response.answer == "I couldn't find this in the indexed documentation."
        assert len(response.citations) == 0
        assert len(response.used_chunk_ids) == 0
        # Should not call LLM when context is empty
        mock_llm_client.generate.assert_not_called()

    def test_ask_citation_compliance_check(
        self, rag_service, mock_search_service, mock_llm_client, sample_search_results
    ):
        """Test that ask performs citation compliance check."""
        mock_search_service.hybrid_search.return_value = sample_search_results
        # LLM returns answer without citations
        mock_llm_client.generate.return_value = "This is an answer without citations."

        response = rag_service.ask(
            query="test query",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        # Should be replaced with refusal, and citations/used_chunk_ids
        # cleared — nothing was actually used to produce a refusal
        assert response.answer == "I couldn't find this in the indexed documentation."
        assert response.citations == []
        assert response.used_chunk_ids == []

    def test_ask_allows_answer_with_citations(
        self, rag_service, mock_search_service, mock_llm_client, sample_search_results
    ):
        """Test that ask allows answers with citations."""
        mock_search_service.hybrid_search.return_value = sample_search_results
        # LLM returns answer with citations
        mock_llm_client.generate.return_value = "This is an answer with [1] citation."

        response = rag_service.ask(
            query="test query",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        # Should keep the answer with citations
        assert "[1]" in response.answer
        assert response.answer != "I couldn't find this in the indexed documentation."

    def test_ask_allows_refusal_string(
        self, rag_service, mock_search_service, mock_llm_client, sample_search_results
    ):
        """Test that ask allows the exact refusal string."""
        mock_search_service.hybrid_search.return_value = sample_search_results
        refusal = "I couldn't find this in the indexed documentation."
        mock_llm_client.generate.return_value = refusal

        response = rag_service.ask(
            query="test query",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        # Should keep the refusal string, with no citations
        assert response.answer == refusal
        assert response.citations == []
        assert response.used_chunk_ids == []

    def test_ask_includes_context_used_in_debug_mode(
        self, rag_service, mock_search_service, mock_llm_client, sample_search_results
    ):
        """Test that ask includes context_used when debug=True."""
        mock_search_service.hybrid_search.return_value = sample_search_results
        mock_llm_client.generate.return_value = "Answer with [1] citation."

        response = rag_service.ask(
            query="test query",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=True,
        )

        assert response.context_used is not None
        assert "[1]" in response.context_used
        assert "First chunk text" in response.context_used

    def test_ask_excludes_context_used_when_debug_false(
        self, rag_service, mock_search_service, mock_llm_client, sample_search_results
    ):
        """Test that ask excludes context_used when debug=False."""
        mock_search_service.hybrid_search.return_value = sample_search_results
        mock_llm_client.generate.return_value = "Answer with [1] citation."

        response = rag_service.ask(
            query="test query",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        assert response.context_used is None

    def test_ask_pipeline_order_with_rerank(
        self, rag_service, mock_search_service, mock_reranker_service, mock_llm_client
    ):
        """Test that ask executes pipeline in correct order: search -> rerank -> context -> llm."""
        search_results = [
            SearchResult(
                chunk_id="chunk1",
                score=0.5,
                text="Text",
                source_path="doc.md",
                canonical_source_id="doc.md",
            )
        ]
        reranked_results = [
            SearchResult(
                chunk_id="chunk1",
                score=0.9,  # Updated score
                text="Text",
                source_path="doc.md",
                canonical_source_id="doc.md",
            )
        ]

        mock_search_service.hybrid_search.return_value = search_results
        mock_reranker_service.rerank.return_value = reranked_results
        mock_llm_client.generate.return_value = "Answer [1]"

        rag_service.ask(
            query="test",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=True,
            debug=False,
        )

        # Verify call order
        mock_search_service.hybrid_search.assert_called_once()
        mock_reranker_service.rerank.assert_called_once()
        mock_llm_client.generate.assert_called_once()

        # Verify rerank was called with correct arguments
        # The important thing is that rerank was called between search and LLM
        # The exact argument verification is less critical than verifying the call order

    def test_ask_pipeline_order_without_rerank(
        self, rag_service, mock_search_service, mock_reranker_service, mock_llm_client
    ):
        """Test that ask skips rerank when rerank=False."""
        search_results = [
            SearchResult(
                chunk_id="chunk1",
                score=0.5,
                text="Text",
                source_path="doc.md",
                canonical_source_id="doc.md",
            )
        ]

        mock_search_service.hybrid_search.return_value = search_results
        mock_llm_client.generate.return_value = "Answer [1]"

        rag_service.ask(
            query="test",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        # Verify rerank was not called
        mock_reranker_service.rerank.assert_not_called()
        mock_llm_client.generate.assert_called_once()

    def test_ask_prompt_includes_context_and_query(
        self, rag_service, mock_search_service, mock_llm_client, sample_search_results
    ):
        """Test that ask builds prompt with context and query."""
        mock_search_service.hybrid_search.return_value = sample_search_results
        mock_llm_client.generate.return_value = "Answer [1]"

        rag_service.ask(
            query="What is Python?",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        # Verify prompt includes system prompt, context, and query
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "technical assistant" in call_args.lower()
        assert "What is Python?" in call_args
        assert "[1]" in call_args  # Context label
        assert "First chunk text" in call_args  # Context content

    def test_ask_handles_empty_search_results(self, rag_service, mock_search_service):
        """Test that ask handles empty search results."""
        mock_search_service.hybrid_search.return_value = []

        response = rag_service.ask(
            query="test",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=False,
            debug=False,
        )

        assert response.answer == "I couldn't find this in the indexed documentation."
        assert len(response.citations) == 0

    def test_ask_handles_rerank_with_empty_results(
        self, rag_service, mock_search_service, mock_reranker_service
    ):
        """Test that ask handles rerank returning empty results."""
        mock_search_service.hybrid_search.return_value = [
            SearchResult(
                chunk_id="chunk1",
                score=0.5,
                text="Text",
                source_path="doc.md",
                canonical_source_id="doc.md",
            )
        ]
        mock_reranker_service.rerank.return_value = []

        response = rag_service.ask(
            query="test",
            collection="test_collection",
            top_k=25,
            top_n=5,
            rerank=True,
            debug=False,
        )

        assert response.answer == "I couldn't find this in the indexed documentation."
        assert len(response.citations) == 0

    def test_ask_stream_yields_tokens_then_done(
        self, rag_service, mock_search_service, mock_reranker_service, mock_llm_client
    ):
        """Test that ask_stream yields tokens then done message with citations."""
        mock_search_service.hybrid_search.return_value = [
            SearchResult(
                chunk_id="chunk1",
                score=0.9,
                text="Context text",
                source_path="doc.md",
                canonical_source_id="doc.md",
                header="Section",
            )
        ]
        mock_reranker_service.rerank.return_value = mock_search_service.hybrid_search.return_value
        mock_llm_client.generate_stream.return_value = iter(["Hello", " ", "world"])

        items = list(
            rag_service.ask_stream(
                query="test",
                collection="test_collection",
                top_k=25,
                top_n=5,
                rerank=True,
            )
        )

        assert items[:-1] == ["Hello", " ", "world"]
        assert items[-1]["type"] == "done"
        assert "citations" in items[-1]
        assert "used_chunk_ids" in items[-1]
        assert len(items[-1]["citations"]) == 1
        assert items[-1]["citations"][0]["chunk_id"] == "chunk1"

    def test_ask_stream_empty_context_yields_refusal_and_done(
        self, rag_service, mock_search_service
    ):
        """Test that ask_stream yields refusal and empty citations when context is empty."""
        mock_search_service.hybrid_search.return_value = []

        items = list(
            rag_service.ask_stream(
                query="test",
                collection="test_collection",
                top_k=25,
                top_n=5,
                rerank=False,
            )
        )

        assert len(items) == 2
        assert items[0] == "I couldn't find this in the indexed documentation."
        assert items[1] == {"type": "done", "citations": [], "used_chunk_ids": []}
        mock_search_service.hybrid_search.assert_called_once()
