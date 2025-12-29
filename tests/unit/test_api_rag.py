"""Tests for the RAG API endpoints."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.main import app
from src.llm import OllamaConnectionError
from src.models import Citation, RAGResponse, SearchResult

client = TestClient(app)


def test_search_success(monkeypatch):
    """Test successful search."""
    # Mock search service
    mock_search_service = MagicMock()
    mock_search_result = SearchResult(
        chunk_id="test_chunk",
        score=0.9,
        text="This is a test chunk.",
        source_path="/path/to/doc.md",
        canonical_source_id="doc.md",
        header="Test Header",
        library="test_lib",
        version="1.0",
        title="Test Document",
    )
    mock_search_service.hybrid_search.return_value = [mock_search_result]

    # Mock get_search_service to return our mock service
    monkeypatch.setattr("api.routes.rag.get_search_service", lambda: mock_search_service)

    # Make request
    response = client.post(
        "/api/search",
        json={"query": "test query", "collection": "test_collection", "top_k": 1},
    )

    # Assert
    assert response.status_code == 200
    assert response.json() == {"results": [mock_search_result.model_dump()]}
    mock_search_service.hybrid_search.assert_called_once_with(
        query="test query", collection="test_collection", top_k=1, filters=None
    )


def test_search_error(monkeypatch):
    """Test search with an error in the service."""
    # Mock search service to raise an exception
    mock_search_service = MagicMock()
    mock_search_service.hybrid_search.side_effect = Exception("Test search error")

    # Mock get_search_service
    monkeypatch.setattr("api.routes.rag.get_search_service", lambda: mock_search_service)

    # Make request
    response = client.post(
        "/api/search",
        json={"query": "test query", "collection": "test_collection", "top_k": 1},
    )

    # Assert
    assert response.status_code == 500
    assert "Search failed: Test search error" in response.json()["detail"]


def test_ask_success(monkeypatch):
    """Test successful ask."""
    mock_rag_service = MagicMock()
    mock_rag_response = RAGResponse(
        answer="This is a test answer.",
        citations=[
            Citation(
                label="[1]",
                chunk_id="test_chunk",
                canonical_source_id="doc.md",
                source_path="/path/to/doc.md",
                header="Test Header",
                title="Test Document",
                score=0.9,
            )
        ],
        used_chunk_ids=["test_chunk"],
    )
    mock_rag_service.ask.return_value = mock_rag_response

    monkeypatch.setattr("api.routes.rag.RAGService", lambda **kwargs: mock_rag_service)

    response = client.post(
        "/api/ask",
        json={"query": "test query", "collection": "test_collection"},
    )

    assert response.status_code == 200
    # Pydantic models with lists need careful comparison
    response_data = response.json()
    expected_data = mock_rag_response.model_dump()
    assert response_data["answer"] == expected_data["answer"]
    assert response_data["citations"] == expected_data["citations"]
    assert response_data["used_chunk_ids"] == expected_data["used_chunk_ids"]


def test_ask_ollama_error(monkeypatch):
    """Test ask with an Ollama connection error."""
    mock_rag_service = MagicMock()
    mock_rag_service.ask.side_effect = OllamaConnectionError("Test Ollama error")

    monkeypatch.setattr("api.routes.rag.RAGService", lambda **kwargs: mock_rag_service)

    response = client.post(
        "/api/ask",
        json={"query": "test query", "collection": "test_collection"},
    )

    assert response.status_code == 503
    assert "Ollama service unavailable: Test Ollama error" in response.json()["detail"]


def test_ask_general_error(monkeypatch):
    """Test ask with a general error."""
    mock_rag_service = MagicMock()
    mock_rag_service.ask.side_effect = Exception("Test general error")

    monkeypatch.setattr("api.routes.rag.RAGService", lambda **kwargs: mock_rag_service)

    response = client.post(
        "/api/ask",
        json={"query": "test query", "collection": "test_collection"},
    )

    assert response.status_code == 500
    assert "RAG pipeline failed: Test general error" in response.json()["detail"]
