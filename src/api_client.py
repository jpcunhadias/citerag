"""HTTP client for FastAPI backend."""

import logging
from typing import Optional

import httpx

from src.config import API_BASE_URL
from src.models import Citation, RAGResponse, SearchResult

logger = logging.getLogger(__name__)


class APIClientError(Exception):
    """Exception raised when API client operations fail."""

    pass


class APIClient:
    """Client for making HTTP requests to FastAPI backend."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize APIClient.

        Args:
            base_url: Base URL for FastAPI backend. If None, uses API_BASE_URL from config.
        """
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        logger.info(f"Initialized APIClient with base_url={self.base_url}")

    def ask(
        self,
        query: str,
        collection: str,
        top_k: int = 25,
        top_n: int = 5,
        rerank: bool = True,
        debug: bool = False,
    ) -> RAGResponse:
        """
        Execute RAG pipeline via API.

        Args:
            query: User query/question
            collection: Qdrant collection name
            top_k: Number of initial search results
            top_n: Number of results to rerank and use for context
            rerank: Whether to apply reranking
            debug: Whether to include context_used in response

        Returns:
            RAGResponse with answer, citations, and metadata

        Raises:
            APIClientError: If API request fails
        """
        url = f"{self.base_url}/api/ask"
        payload = {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "top_n": top_n,
            "rerank": rerank,
            "debug": debug,
        }

        try:
            logger.info(f"Sending ask request to {url}")
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

            # Convert API response to RAGResponse
            return RAGResponse(
                answer=result["answer"],
                citations=[Citation(**citation) for citation in result["citations"]],
                context_used=result.get("context_used"),
                used_chunk_ids=result["used_chunk_ids"],
            )

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"API request failed with status {e.response.status_code}: {e.response.text}"
            )
            logger.error(error_msg)
            raise APIClientError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to API at {self.base_url}: {str(e)}"
            logger.error(error_msg)
            raise APIClientError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in API client: {str(e)}"
            logger.error(error_msg)
            raise APIClientError(error_msg) from e

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filters: Optional[dict[str, str]] = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search via API.

        Args:
            query: Search query text
            collection: Qdrant collection name
            top_k: Number of results to return
            filters: Optional filters (e.g., {'library': 'pandas', 'version': '2.0'})

        Returns:
            List of SearchResult objects

        Raises:
            APIClientError: If API request fails
        """
        url = f"{self.base_url}/api/search"
        payload = {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "filters": filters,
        }

        try:
            logger.info(f"Sending search request to {url}")
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                result = response.json()

            # Convert API response to SearchResult objects
            return [SearchResult(**item) for item in result["results"]]

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"API request failed with status {e.response.status_code}: {e.response.text}"
            )
            logger.error(error_msg)
            raise APIClientError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to API at {self.base_url}: {str(e)}"
            logger.error(error_msg)
            raise APIClientError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in API client: {str(e)}"
            logger.error(error_msg)
            raise APIClientError(error_msg) from e

    def get_collections(self) -> list[str]:
        """
        Get list of available collections via API.

        Returns:
            List of collection names

        Raises:
            APIClientError: If API request fails
        """
        url = f"{self.base_url}/collections"

        try:
            logger.info(f"Fetching collections from {url}")
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url)
                response.raise_for_status()
                result = response.json()

            return result["collections"]

        except httpx.HTTPStatusError as e:
            error_msg = (
                f"API request failed with status {e.response.status_code}: {e.response.text}"
            )
            logger.error(error_msg)
            raise APIClientError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Failed to connect to API at {self.base_url}: {str(e)}"
            logger.error(error_msg)
            raise APIClientError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error in API client: {str(e)}"
            logger.error(error_msg)
            raise APIClientError(error_msg) from e
