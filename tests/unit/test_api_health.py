"""Tests for the API health endpoint."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check_healthy(monkeypatch):
    """Test health check when all services are healthy."""
    monkeypatch.setattr("api.routes.health.check_qdrant", lambda: "healthy")
    monkeypatch.setattr("api.routes.health.check_ollama", lambda: "healthy")

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "services": {"qdrant": "healthy", "ollama": "healthy"},
    }


def test_health_check_unhealthy_qdrant(monkeypatch):
    """Test health check when Qdrant is unhealthy."""
    monkeypatch.setattr("api.routes.health.check_qdrant", lambda: "unhealthy")
    monkeypatch.setattr("api.routes.health.check_ollama", lambda: "healthy")

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "services": {"qdrant": "unhealthy", "ollama": "healthy"},
    }


def test_health_check_unhealthy_ollama(monkeypatch):
    """Test health check when Ollama is unhealthy."""
    monkeypatch.setattr("api.routes.health.check_qdrant", lambda: "healthy")
    monkeypatch.setattr("api.routes.health.check_ollama", lambda: "unhealthy")

    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unhealthy",
        "services": {"qdrant": "healthy", "ollama": "unhealthy"},
    }


def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "CiteRAG API",
        "docs": "/docs",
        "health": "/health",
    }
