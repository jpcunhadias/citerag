"""HTTP client for FastAPI backend."""

import json
import logging
from collections.abc import Iterator

import httpx

from src.config import API_BASE_URL
from src.models import Citation, RAGResponse, SearchResult

logger = logging.getLogger(__name__)


class StreamAskResult:
    """
    Result of streaming ask. Provides token iterator and citations after stream completes.

    Use with st.write_stream(result) - after iteration, citations and used_chunk_ids
    are populated from the final NDJSON message.
    """

    def __init__(
        self,
        token_iterator: Iterator[str],
        citations_ref: list,
        used_chunk_ids_ref: list,
    ):
        self._token_iterator = token_iterator
        self.citations: list[Citation] = citations_ref
        self.used_chunk_ids: list[str] = used_chunk_ids_ref

    def __iter__(self) -> Iterator[str]:
        return self._token_iterator


class APIClientError(Exception):
    """Exception raised when API client operations fail."""

    pass


class APIClient:
    """Client for making HTTP requests to FastAPI backend."""

    def __init__(self, base_url: str | None = None):
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
        use_hyde: bool = False,
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
            use_hyde: Retrieve using a generated hypothetical answer instead of the raw query

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
            "use_hyde": use_hyde,
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

    def ask_stream(
        self,
        query: str,
        collection: str,
        top_k: int = 25,
        top_n: int = 5,
        rerank: bool = True,
        use_hyde: bool = False,
    ) -> StreamAskResult:
        """
        Execute RAG pipeline via streaming API. Returns StreamAskResult for st.write_stream.

        Citations and used_chunk_ids are populated after the stream is fully consumed.

        Args:
            query: User query/question
            collection: Qdrant collection name
            top_k: Number of initial search results
            top_n: Number of results to rerank and use for context
            rerank: Whether to apply reranking
            use_hyde: Retrieve using a generated hypothetical answer instead of the raw query

        Returns:
            StreamAskResult with __iter__ yielding token strings; citations populated when done.
        """
        url = f"{self.base_url}/api/ask/stream"
        payload = {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "top_n": top_n,
            "rerank": rerank,
            "use_hyde": use_hyde,
        }

        citations: list[Citation] = []
        used_chunk_ids: list[str] = []

        def token_generator() -> Iterator[str]:
            try:
                with httpx.Client(timeout=300.0) as client:
                    with client.stream("POST", url, json=payload) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            if obj.get("type") == "token":
                                content = obj.get("content", "")
                                if content:
                                    yield content
                            elif obj.get("type") == "done":
                                citations.extend([Citation(**c) for c in obj.get("citations", [])])
                                used_chunk_ids.extend(obj.get("used_chunk_ids", []))
                                return
                            elif obj.get("type") == "error":
                                raise APIClientError(obj.get("detail", "Stream error"))
            except httpx.HTTPStatusError as e:
                raise APIClientError(
                    f"API request failed: {e.response.status_code} {e.response.text}"
                ) from e
            except httpx.RequestError as e:
                raise APIClientError(
                    f"Failed to connect to API at {self.base_url}: {str(e)}"
                ) from e

        return StreamAskResult(token_generator(), citations, used_chunk_ids)

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
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
