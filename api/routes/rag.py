"""RAG endpoints: ask and search."""

import asyncio
import json
import logging
import queue
import threading

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm_client, get_reranker_service, get_search_service
from api.models import AskRequest, AskResponse, SearchRequest, SearchResponse
from src.llm import OllamaConnectionError
from src.rag import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["rag"])


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Execute RAG pipeline: search, rerank, build context, and generate answer.

    Args:
        request: AskRequest with query, collection, and parameters

    Returns:
        AskResponse with answer, citations, and metadata

    Raises:
        HTTPException: If RAG pipeline fails or services unavailable
    """
    try:
        # Get services (singletons)
        search_service = get_search_service()
        reranker_service = get_reranker_service()
        llm_client = get_llm_client()

        # Create RAG service
        rag_service = RAGService(
            search_service=search_service,
            reranker_service=reranker_service,
            llm_client=llm_client,
        )

        # Execute RAG pipeline in thread pool (services are sync)
        logger.info(
            f"Processing ask request: query='{request.query[:50]}...', "
            f"collection={request.collection}"
        )
        response = await asyncio.to_thread(
            rag_service.ask,
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
            top_n=request.top_n,
            rerank=request.rerank,
            debug=request.debug,
        )

        # Convert to API response model
        return AskResponse(
            answer=response.answer,
            citations=response.citations,
            context_used=response.context_used if request.debug else None,
            used_chunk_ids=response.used_chunk_ids,
        )

    except OllamaConnectionError as e:
        logger.error(f"Ollama connection error: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama service unavailable: {str(e)}") from e
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {str(e)}") from e


@router.post("/ask/stream")
async def ask_stream(request: AskRequest) -> StreamingResponse:
    """
    Execute RAG pipeline and stream LLM tokens as NDJSON.

    Each line is JSON: {"type": "token", "content": "..."} or
    {"type": "done", "citations": [...], "used_chunk_ids": [...]}.
    """

    def run_stream(q: queue.Queue) -> None:
        try:
            search_service = get_search_service()
            reranker_service = get_reranker_service()
            llm_client = get_llm_client()
            rag_service = RAGService(
                search_service=search_service,
                reranker_service=reranker_service,
                llm_client=llm_client,
            )
            for item in rag_service.ask_stream(
                query=request.query,
                collection=request.collection,
                top_k=request.top_k,
                top_n=request.top_n,
                rerank=request.rerank,
            ):
                q.put(item)
        except Exception as e:
            q.put({"type": "error", "detail": str(e)})
        finally:
            q.put(None)

    async def ndjson_generator():
        q: queue.Queue = queue.Queue()
        thread = threading.Thread(target=run_stream, args=(q,))
        thread.start()
        loop = asyncio.get_event_loop()
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, dict):
                if item.get("type") == "error":
                    raise HTTPException(status_code=500, detail=item.get("detail", "Unknown error"))
                yield json.dumps(item) + "\n"
            else:
                yield json.dumps({"type": "token", "content": item}) + "\n"
        thread.join()

    return StreamingResponse(
        ndjson_generator(),
        media_type="application/x-ndjson",
    )


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Perform hybrid search on documents.

    Args:
        request: SearchRequest with query, collection, top_k, and optional filters

    Returns:
        SearchResponse with list of search results

    Raises:
        HTTPException: If search fails or services unavailable
    """
    try:
        # Get search service (singleton)
        search_service = get_search_service()

        # Execute search in thread pool (service is sync)
        logger.info(
            f"Processing search request: query='{request.query[:50]}...', "
            f"collection={request.collection}"
        )
        results = await asyncio.to_thread(
            search_service.hybrid_search,
            query=request.query,
            collection=request.collection,
            top_k=request.top_k,
            filters=request.filters,
        )

        return SearchResponse(results=results)

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e
