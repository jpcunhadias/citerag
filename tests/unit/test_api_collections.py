"""Tests for the API collections endpoint."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_get_collections_success(monkeypatch):
    """Test successful retrieval of collections."""
    # Mock QdrantClient
    mock_qdrant_client = MagicMock()

    # Mock CollectionInfo
    mock_collection_info1 = MagicMock()
    mock_collection_info1.name = "test_collection_1"
    mock_collection_info2 = MagicMock()
    mock_collection_info2.name = "test_collection_2"

    # Mock CollectionsResponse
    mock_collections_response = MagicMock()
    mock_collections_response.collections = [mock_collection_info1, mock_collection_info2]

    mock_qdrant_client.get_collections.return_value = mock_collections_response
    monkeypatch.setattr("api.routes.collections.QdrantClient", lambda **kwargs: mock_qdrant_client)

    # Make request
    response = client.get("/collections")

    # Assert
    assert response.status_code == 200
    assert response.json() == {"collections": ["test_collection_1", "test_collection_2"]}


def test_get_collections_qdrant_error(monkeypatch):
    """Test Qdrant connection error."""
    # Mock QdrantClient to raise an exception
    mock_qdrant_client = MagicMock()
    mock_qdrant_client.get_collections.side_effect = ConnectionError("Test connection error")
    monkeypatch.setattr("api.routes.collections.QdrantClient", lambda **kwargs: mock_qdrant_client)

    # Make request
    response = client.get("/collections")

    # Assert
    assert response.status_code == 503
    assert "Unable to connect to Qdrant" in response.json()["detail"]
