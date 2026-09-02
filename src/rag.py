"""RAG orchestration service: search, rerank, and generate answers."""

import logging
import re
from collections.abc import Iterator

from src.config import RAG_MAX_CONTEXT_CHARS, RAG_REFUSAL_MESSAGE
from src.llm import OllamaClient
from src.models import Citation, RAGResponse, SearchResult
from src.rerank import RerankerService
from src.search import SearchService

logger = logging.getLogger(__name__)


class RAGService:
    """Service for RAG pipeline: search, rerank, context building, and LLM generation."""

    def __init__(
        self,
        search_service: SearchService,
        reranker_service: RerankerService,
        llm_client: OllamaClient,
    ):
        """
        Initialize RAGService.

        Args:
            search_service: SearchService instance for hybrid search.
            reranker_service: RerankerService instance for reranking.
            llm_client: OllamaClient instance for LLM generation.
        """
        self.search_service = search_service
        self.reranker_service = reranker_service
        self.llm_client = llm_client

    def build_context(self, chunks: list[SearchResult]) -> tuple[str, list[Citation], list[str]]:
        """
        Build formatted context string from search results with character budget.

        Args:
            chunks: List of SearchResult objects to include in context.

        Returns:
            Tuple of (context_str, citations_list, used_chunk_ids):
            - context_str: Formatted context string with labeled chunks
            - citations_list: List of Citation objects for accepted chunks (with scores included)
            - used_chunk_ids: List of chunk IDs that were included
        """
        context_parts: list[str] = []
        citations: list[Citation] = []
        used_chunk_ids: list[str] = []
        current_len = 0

        for i, chunk in enumerate(chunks, start=1):
            # Format entry: [i] chunk.text\n\n
            entry = f"[{i}] {chunk.text}\n\n"

            # Budget check: stop if adding this entry would exceed limit
            if current_len + len(entry) > RAG_MAX_CONTEXT_CHARS:
                logger.info(
                    f"Context budget exceeded at chunk {i}/{len(chunks)} "
                    f"(current: {current_len}, entry: {len(entry)}, limit: {RAG_MAX_CONTEXT_CHARS})"
                )
                break

            # Add entry to context
            context_parts.append(entry)
            current_len += len(entry)

            # Create Citation object with all fields from SearchResult (including score)
            citation = Citation(
                label=str(i),
                chunk_id=chunk.chunk_id,
                canonical_source_id=chunk.canonical_source_id,
                source_path=chunk.source_path,
                header=chunk.header,
                title=chunk.title,
                score=chunk.score,
            )
            citations.append(citation)
            used_chunk_ids.append(chunk.chunk_id)

        context_str = "".join(context_parts)
        logger.info(
            f"Built context: {len(citations)} chunks, {len(context_str)} chars "
            f"(budget: {RAG_MAX_CONTEXT_CHARS})"
        )

        return context_str, citations, used_chunk_ids

    def _generate_hyde_passage(self, query: str) -> str:
        """
        Generate a hypothetical passage answering the query (HyDE).

        Used only to produce a better retrieval embedding for vague or
        keyword-poor queries — a hypothetical answer tends to sit closer in
        embedding space to real matching documents than the bare question
        does. The real query is still used for the final answer prompt; this
        passage is never shown to the user.

        Args:
            query: User query text.

        Returns:
            LLM-generated hypothetical documentation passage.
        """
        hyde_prompt = (
            "Write a short passage (2-4 sentences) that directly answers the "
            "following question, written as if it were an excerpt from "
            "technical documentation. State it as fact even if you are not "
            "certain of exact details — this passage is only used to improve "
            "document retrieval, it is not shown to the user.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        return self.llm_client.generate(hyde_prompt)

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
        Execute RAG pipeline: search, rerank, build context, and generate answer.

        Args:
            query: User query text.
            collection: Qdrant collection name.
            top_k: Number of initial search results to retrieve.
            top_n: Number of results to rerank and use for context.
            rerank: Whether to apply reranking (default: True).
            debug: Whether to include context_used in response (default: False).
            use_hyde: Retrieve using a generated hypothetical answer instead of
                the raw query (default: False). The final answer is still
                grounded in the real query.

        Returns:
            RAGResponse object with answer, citations, and metadata.
        """
        logger.info(
            f"RAG pipeline started: query='{query[:50]}...', "
            f"top_k={top_k}, top_n={top_n}, rerank={rerank}, use_hyde={use_hyde}"
        )

        # Step 1: Hybrid search (optionally retrieving with a HyDE passage
        # instead of the raw query; the answer prompt below always uses the
        # real query)
        search_query = query
        if use_hyde:
            search_query = self._generate_hyde_passage(query)
            logger.info(f"HyDE passage for retrieval: '{search_query[:80]}...'")

        results = self.search_service.hybrid_search(
            query=search_query, collection=collection, top_k=top_k
        )
        logger.info(f"Search returned {len(results)} results")

        # Step 2: Rerank if enabled
        if rerank and results:
            results = self.reranker_service.rerank(query=query, results=results, top_n=top_n)
            logger.info(f"Reranking returned {len(results)} results")

        # Step 3: Build context
        context_str, citations, used_chunk_ids = self.build_context(results)

        # Step 4: Circuit breaker - if context is empty, return early
        if not context_str.strip():
            logger.warning("Empty context - returning default response")
            return RAGResponse(
                answer=RAG_REFUSAL_MESSAGE,
                citations=[],
                context_used=None if not debug else "",
                used_chunk_ids=[],
            )

        # Step 5: Build prompt
        system_prompt = (
            "You are a technical assistant. Answer using ONLY the context provided. "
            "Context is labeled [1], [2]... Every factual statement must cite a label. "
            f"If the answer is not in the context, say '{RAG_REFUSAL_MESSAGE}' "
            "Do not guess."
        )
        full_prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        # Step 6: Generate answer using LLM
        logger.info("Generating answer with LLM...")
        answer = self.llm_client.generate(full_prompt)

        # Step 7: Citation compliance check
        if answer.strip() != RAG_REFUSAL_MESSAGE:
            # Check if answer contains citation labels [1], [2], etc.
            citation_pattern = r"\[\d+\]"
            if not re.search(citation_pattern, answer):
                logger.warning(
                    "Answer does not contain citations but is not a refusal. "
                    "Replacing with refusal string."
                )
                answer = RAG_REFUSAL_MESSAGE

        # Step 8: Return RAGResponse. A refusal answer — whether the LLM said
        # it directly or the compliance check above forced it — should carry
        # no citations, since nothing was actually used to produce it.
        is_refusal = answer.strip() == RAG_REFUSAL_MESSAGE
        return RAGResponse(
            answer=answer,
            citations=[] if is_refusal else citations,
            context_used=context_str if debug else None,
            used_chunk_ids=[] if is_refusal else used_chunk_ids,
        )

    def ask_stream(
        self,
        query: str,
        collection: str,
        top_k: int = 25,
        top_n: int = 5,
        rerank: bool = True,
        use_hyde: bool = False,
    ) -> Iterator[str | dict]:
        """
        Execute RAG pipeline and stream LLM tokens. Skips citation compliance check.

        Yields token strings, then a final dict:
        {"type": "done", "citations": [...], "used_chunk_ids": [...]}.

        Args:
            query: User query text.
            collection: Qdrant collection name.
            top_k: Number of initial search results to retrieve.
            top_n: Number of results to rerank and use for context.
            rerank: Whether to apply reranking (default: True).
            use_hyde: Retrieve using a generated hypothetical answer instead of
                the raw query (default: False). The final answer is still
                grounded in the real query.

        Yields:
            str: Token chunks from the LLM, or a dict with type="done" and citations.
        """
        logger.info(
            f"RAG stream started: query='{query[:50]}...', "
            f"top_k={top_k}, top_n={top_n}, rerank={rerank}, use_hyde={use_hyde}"
        )

        # Step 1: Hybrid search (optionally retrieving with a HyDE passage
        # instead of the raw query; the answer prompt below always uses the
        # real query)
        search_query = query
        if use_hyde:
            search_query = self._generate_hyde_passage(query)
            logger.info(f"HyDE passage for retrieval: '{search_query[:80]}...'")

        results = self.search_service.hybrid_search(
            query=search_query, collection=collection, top_k=top_k
        )
        logger.info(f"Search returned {len(results)} results")

        # Step 2: Rerank if enabled
        if rerank and results:
            results = self.reranker_service.rerank(query=query, results=results, top_n=top_n)
            logger.info(f"Reranking returned {len(results)} results")

        # Step 3: Build context
        context_str, citations, used_chunk_ids = self.build_context(results)

        # Step 4: Circuit breaker - if context is empty, yield refusal and done
        if not context_str.strip():
            logger.warning("Empty context - yielding default response")
            yield RAG_REFUSAL_MESSAGE
            yield {"type": "done", "citations": [], "used_chunk_ids": []}
            return

        # Step 5: Build prompt
        system_prompt = (
            "You are a technical assistant. Answer using ONLY the context provided. "
            "Context is labeled [1], [2]... Every factual statement must cite a label. "
            f"If the answer is not in the context, say '{RAG_REFUSAL_MESSAGE}' "
            "Do not guess."
        )
        full_prompt = f"{system_prompt}\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

        # Step 6: Stream LLM tokens (no citation compliance check for streaming)
        logger.info("Streaming answer with LLM...")
        yield from self.llm_client.generate_stream(full_prompt)

        # Step 7: Yield done message with citations
        citations_data = [c.model_dump() for c in citations]
        yield {"type": "done", "citations": citations_data, "used_chunk_ids": used_chunk_ids}
