"""FastAPI dependencies for service injection."""

import logging
from functools import lru_cache

from qdrant_client import QdrantClient

from src.config import QDRANT_HOST, QDRANT_PORT
from src.ingest import VectorService
from src.llm import OllamaClient
from src.rerank import RerankerService
from src.search import SearchService

logger = logging.getLogger(__name__)


# Global service instances (singleton pattern)
_search_service: SearchService | None = None
_reranker_service: RerankerService | None = None
_llm_client: OllamaClient | None = None


def get_search_service() -> SearchService:
    """
    Get or create SearchService instance (singleton).

    Returns:
        SearchService instance
    """
    global _search_service
    if _search_service is None:
        logger.info("Initializing SearchService (singleton)")
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        vector_service = VectorService()
        _search_service = SearchService(qdrant_client, vector_service)
    return _search_service


def get_reranker_service() -> RerankerService:
    """
    Get or create RerankerService instance (singleton).

    Returns:
        RerankerService instance
    """
    global _reranker_service
    if _reranker_service is None:
        logger.info("Initializing RerankerService (singleton)")
        _reranker_service = RerankerService()
    return _reranker_service


def get_llm_client() -> OllamaClient:
    """
    Get or create OllamaClient instance (singleton).

    Returns:
        OllamaClient instance
    """
    global _llm_client
    if _llm_client is None:
        logger.info("Initializing OllamaClient (singleton)")
        _llm_client = OllamaClient()
    return _llm_client

