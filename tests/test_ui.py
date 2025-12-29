"""Unit tests for the Streamlit UI module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.models import Citation
from src.ui import build_citations_data, get_collections, init_state
from src.api_client import APIClient, APIClientError

# Constants for tests
FALLBACK_COLLECTIONS = ["pandas_docs"]


@pytest.fixture
def mock_api_client():
    """Fixture to create a mock APIClient."""
    return MagicMock(spec=APIClient)


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
        ]
        result = build_citations_data(citations)
        assert result[0]["Header"] == "Introduction"

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
        assert result[0]["Header"] == "File 1"

# Using patch as a decorator for the whole class
@patch("streamlit.cache_data", lambda **kwargs: lambda func: func)
class TestGetCollections:
    """Tests for get_collections function."""

    def test_get_collections_success(self, mock_api_client):
        """Test successful retrieval of collections."""
        mock_api_client.get_collections.return_value = ["pandas_docs", "numpy_docs"]
        collections = get_collections(mock_api_client)
        assert collections == ["pandas_docs", "numpy_docs"]
        mock_api_client.get_collections.assert_called_once()

    @patch("src.ui.logger")
    def test_get_collections_error_fallback(self, mock_logger, mock_api_client):
        """Test fallback when collection retrieval fails."""
        mock_api_client.get_collections.side_effect = APIClientError("Connection failed")
        collections = get_collections(mock_api_client)
        assert collections == FALLBACK_COLLECTIONS
        mock_logger.error.assert_called_once()


class TestSessionStateManagement:
    """Tests for session state initialization."""

    @patch("src.ui.st")
    def test_init_state_creates_messages_and_client(self, mock_st):
        """Test that init_state creates messages list and api_client if not present."""
        mock_st.session_state = {}
        init_state()
        assert "messages" in mock_st.session_state
        assert "api_client" in mock_st.session_state
        assert isinstance(mock_st.session_state["api_client"], APIClient)

    @patch("src.ui.st")
    def test_init_state_preserves_existing_state(self, mock_st):
        """Test that init_state preserves existing messages and client."""
        existing_messages = [{"role": "user", "content": "test"}]
        mock_client = MagicMock()
        mock_st.session_state = {"messages": existing_messages, "api_client": mock_client}
        init_state()
        assert mock_st.session_state["messages"] == existing_messages
        assert mock_st.session_state["api_client"] == mock_client