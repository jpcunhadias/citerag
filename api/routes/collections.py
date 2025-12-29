"""Collections endpoint."""

import logging

from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient

from api.models import CollectionsResponse
from src.config import QDRANT_HOST, QDRANT_PORT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("", response_model=CollectionsResponse)
async def get_collections() -> CollectionsResponse:
    """
    Get list of available Qdrant collections.

    Returns:
        CollectionsResponse with list of collection names

    Raises:
        HTTPException: If unable to connect to Qdrant
    """
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        logger.info(f"Retrieved {len(collection_names)} collections")
        return CollectionsResponse(collections=collection_names)
    except Exception as e:
        logger.error(f"Error fetching collections: {e}")
        raise HTTPException(status_code=503, detail=f"Unable to connect to Qdrant: {str(e)}") from e

