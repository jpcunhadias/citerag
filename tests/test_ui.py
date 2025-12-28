"""Unit tests for the Streamlit UI module."""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Mock heavy dependencies before importing src modules
sys.modules['FlagEmbedding'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['langchain_community'] = MagicMock()
sys.modules['langchain_text_splitters'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.documents'] = MagicMock()

from src.models import Citation, RAGResponse
from src.ui import build_citations_data, get_collections, get_services


class TestBuildCitationsData:
    """Tests for build_citations_data helper function."""

    def test_build_citations_data_with_header(self):
        """Test building citation data when citations have headers."""
        citations = [
            Citation(
                label="[1]",
                chunk_id="chunk1",
                canonical_source_id="file1.md",
                score=0.95,
                source_path="docs/file1.md",
                header="Introduction",
                title="File 1",
            ),
            Citation(
                label="[2]",
                chunk_id="chunk2",
                canonical_source_id="file2.md",
                score=0.87,
                source_path="docs/file2.md",
                header="Getting Started",
                title="File 2",
            ),
        ]

        result = build_citations_data(citations)

        assert len(result) == 2
        assert result[0]["ID"] == "[1]"
        assert result[0]["Score"] == "0.9500"
        assert result[0]["File"] == "docs/file1.md"
        assert result[0]["Header"] == "Introduction"

        assert result[1]["ID"] == "[2]"
        assert result[1]["Score"] == "0.8700"
        assert result[1]["File"] == "docs/file2.md"
        assert result[1]["Header"] == "Getting Started"

    def test_build_citations_data_without_header(self):
        """Test building citation data when citations have no header but have title."""
        citations = [
            Citation(
                label="[1]",
                chunk_id="chunk1",
                canonical_source_id="file1.md",
                score=0.95,
                source_path="docs/file1.md",
                header=None,
                title="File 1",
            ),
        ]

        result = build_citations_data(citations)

        assert len(result) == 1
        assert result[0]["Header"] == "File 1"

    def test_build_citations_data_without_header_and_title(self):
        """Test building citation data when citations have neither header nor title."""
        citations = [
            Citation(
                label="[1]",
                chunk_id="chunk1",
                canonical_source_id="file1.md",
                score=0.95,
                source_path="docs/file1.md",
                header=None,
                title=None,
            ),
        ]

        result = build_citations_data(citations)

        assert len(result) == 1
        assert result[0]["Header"] == "-"

    def test_build_citations_data_with_none_score(self):
        """Test building citation data when citations have None score."""
        citations = [
            Citation(
                label="[1]",
                chunk_id="chunk1",
                canonical_source_id="file1.md",
                score=None,
                source_path="docs/file1.md",
                header="Introduction",
                title="File 1",
            ),
        ]

        result = build_citations_data(citations)

        assert len(result) == 1
        assert result[0]["Score"] == "-"

    def test_build_citations_data_empty_list(self):
        """Test building citation data with empty list."""
        citations = []

        result = build_citations_data(citations)

        assert len(result) == 0
        assert result == []


class TestGetServices:
    """Tests for get_services function."""

    @patch("src.ui.OllamaClient")
    @patch("src.ui.RerankerService")
    @patch("src.ui.SearchService")
    @patch("src.ui.VectorService")
    @patch("src.ui.QdrantClient")
    def test_get_services_initialization(
        self,
        mock_qdrant,
        mock_vector_service,
        mock_search_service,
        mock_reranker_service,
        mock_ollama_client,
    ):
        """Test that get_services initializes all services correctly."""
        # Create mock instances
        mock_qdrant_instance = MagicMock()
        mock_vector_instance = MagicMock()
        mock_search_instance = MagicMock()
        mock_reranker_instance = MagicMock()
        mock_ollama_instance = MagicMock()

        # Configure mocks to return instances
        mock_qdrant.return_value = mock_qdrant_instance
        mock_vector_service.return_value = mock_vector_instance
        mock_search_service.return_value = mock_search_instance
        mock_reranker_service.return_value = mock_reranker_instance
        mock_ollama_client.return_value = mock_ollama_instance

        # Call the function
        search_svc, reranker_svc, llm_client = get_services()

        # Verify all services were initialized
        mock_qdrant.assert_called_once()
        mock_vector_service.assert_called_once()
        mock_search_service.assert_called_once()
        mock_reranker_service.assert_called_once()
        mock_ollama_client.assert_called_once()

        # Verify return values
        assert search_svc == mock_search_instance
        assert reranker_svc == mock_reranker_instance
        assert llm_client == mock_ollama_instance


class TestGetCollections:
    """Tests for get_collections function."""

    @patch("src.ui.QdrantClient")
    def test_get_collections_success(self, mock_qdrant):
        """Test successful retrieval of collections."""
        # Create mock collection objects
        mock_collection1 = MagicMock()
        mock_collection1.name = "pandas_docs"
        mock_collection2 = MagicMock()
        mock_collection2.name = "numpy_docs"

        # Configure mock client
        mock_client = MagicMock()
        mock_collections_response = MagicMock()
        mock_collections_response.collections = [mock_collection1, mock_collection2]
        mock_client.get_collections.return_value = mock_collections_response
        mock_qdrant.return_value = mock_client

        # Clear cache before test
        import streamlit as st
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()

        # Call the function
        collections = get_collections()

        # Verify results
        assert len(collections) == 2
        assert "pandas_docs" in collections
        assert "numpy_docs" in collections

    @patch("src.ui.QdrantClient")
    @patch("src.ui.logger")
    def test_get_collections_error_fallback(self, mock_logger, mock_qdrant):
        """Test fallback when collection retrieval fails."""
        # Clear cache before test
        import streamlit as st
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()

        # Configure mock to raise an exception
        mock_qdrant.side_effect = Exception("Connection failed")

        # Call the function
        collections = get_collections()

        # Verify fallback behavior
        assert collections == ["pandas_docs"]
        mock_logger.error.assert_called_once()


class TestSessionStateManagement:
    """Tests for session state initialization."""

    @patch("src.ui.st")
    def test_init_state_creates_messages(self, mock_st):
        """Test that init_state creates messages list if not present."""
        from src.ui import init_state

        # Mock session state without 'messages'
        mock_st.session_state = {}

        # Call init_state
        init_state()

        # Verify 'messages' was created
        assert "messages" in mock_st.session_state
        assert mock_st.session_state["messages"] == []

    @patch("src.ui.st")
    def test_init_state_preserves_existing_messages(self, mock_st):
        """Test that init_state preserves existing messages."""
        from src.ui import init_state

        # Mock session state with existing 'messages'
        existing_messages = [{"role": "user", "content": "test"}]
        mock_st.session_state = {"messages": existing_messages}

        # Call init_state
        init_state()

        # Verify existing messages were preserved
        assert mock_st.session_state["messages"] == existing_messages


class TestCitationDataFrameCreation:
    """Integration tests for citation dataframe creation."""

    def test_citations_to_dataframe_conversion(self):
        """Test that citations can be converted to pandas DataFrame."""
        citations = [
            Citation(
                label="[1]",
                chunk_id="chunk1",
                canonical_source_id="file1.md",
                score=0.95,
                source_path="docs/file1.md",
                header="Introduction",
                title="File 1",
            ),
            Citation(
                label="[2]",
                chunk_id="chunk2",
                canonical_source_id="file2.md",
                score=0.87,
                source_path="docs/file2.md",
                header="Getting Started",
                title="File 2",
            ),
        ]

        citations_data = build_citations_data(citations)
        df = pd.DataFrame(citations_data)

        # Verify DataFrame structure
        assert len(df) == 2
        assert list(df.columns) == ["ID", "Score", "File", "Header"]
        assert df["ID"].tolist() == ["[1]", "[2]"]
        assert df["Score"].tolist() == ["0.9500", "0.8700"]
        assert df["File"].tolist() == ["docs/file1.md", "docs/file2.md"]
        assert df["Header"].tolist() == ["Introduction", "Getting Started"]


class TestErrorHandling:
    """Tests for error handling in UI functions."""

    @patch("src.ui.QdrantClient")
    def test_get_collections_handles_connection_error(self, mock_qdrant):
        """Test that get_collections handles connection errors gracefully."""
        # Clear cache before test
        import streamlit as st
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()

        mock_qdrant.side_effect = ConnectionError("Cannot connect to Qdrant")

        # Should not raise, should return fallback
        collections = get_collections()

        assert collections == ["pandas_docs"]

    @patch("src.ui.QdrantClient")
    def test_get_collections_handles_generic_exception(self, mock_qdrant):
        """Test that get_collections handles generic exceptions gracefully."""
        # Clear cache before test
        import streamlit as st
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()

        mock_qdrant.side_effect = RuntimeError("Unexpected error")

        # Should not raise, should return fallback
        collections = get_collections()

        assert collections == ["pandas_docs"]
