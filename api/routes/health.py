"""Health check endpoint."""

import logging

import requests
from fastapi import APIRouter, Response
from qdrant_client import QdrantClient

from api.models import HealthResponse
from src.config import OLLAMA_BASE_URL, QDRANT_HOST, QDRANT_PORT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


def check_qdrant() -> str:
    """
    Check Qdrant connection status.

    Returns:
        Status string: "healthy" or "unhealthy"
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        return "healthy"
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        return "unhealthy"


def check_ollama() -> str:
    """
    Check Ollama connection status.

    Returns:
        Status string: "healthy" or "unhealthy"
    """
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return "healthy"
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return "unhealthy"


@router.get("", response_model=HealthResponse)
async def health_check(response: Response) -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with overall status and individual service statuses
        Status code 200 for healthy, 503 for unhealthy
    """
    qdrant_status = check_qdrant()
    ollama_status = check_ollama()

    overall_status = (
        "healthy" if qdrant_status == "healthy" and ollama_status == "healthy" else "unhealthy"
    )

    # Set appropriate status code for load balancers and monitoring tools
    if overall_status == "unhealthy":
        response.status_code = 503

    return HealthResponse(
        status=overall_status,
        services={
            "qdrant": qdrant_status,
            "ollama": ollama_status,
        },
    )
