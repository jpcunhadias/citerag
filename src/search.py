"""Search and retrieval logic: hybrid search, re-ranking, and RAG generation."""

import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    ScoredPoint,
)

from src.config import QDRANT_HOST, QDRANT_PORT
from src.ingest import VectorService
from src.models import SearchResult
from src.utils.qdrant import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    convert_sparse_dict_to_qdrant_sparsevector,
)

logger = logging.getLogger(__name__)


class SearchService:
    """Service for performing hybrid search using Qdrant."""

    def __init__(self, qdrant_client: QdrantClient, vector_service: VectorService):
        """
        Initialize SearchService.

        Args:
            qdrant_client: Qdrant client instance.
            vector_service: VectorService for generating embeddings.
        """
        self.client = qdrant_client
        self.vector_service = vector_service
        self.rrf_k = 60  # Constant for RRF fusion
        self.prefetch_multiplier = 5  # Multiplier for prefetch candidates
        self.min_prefetch_limit = 50  # Minimum prefetch limit

    def reciprocal_rank_fusion(
        self,
        dense_results: list[ScoredPoint],
        sparse_results: list[ScoredPoint],
        top_k: int,
    ) -> list[ScoredPoint]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Args:
            dense_results: Results from dense search.
            sparse_results: Results from sparse search.
            top_k: Number of final results to return.

        Returns:
            List of ScoredPoint objects sorted by fused RRF score descending.
        """
        # Create score dictionary: {point_id: rrf_score}
        score_dict: dict = {}
        point_dict: dict = {}  # Store full ScoredPoint objects

        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            point_id = result.id
            if point_id not in score_dict:
                score_dict[point_id] = 0.0
                point_dict[point_id] = result
            score_dict[point_id] += rrf_score

        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            point_id = result.id
            if point_id not in score_dict:
                score_dict[point_id] = 0.0
                point_dict[point_id] = result
            score_dict[point_id] += rrf_score

        # Create list of ScoredPoint objects with fused scores
        fused_results = []
        for point_id, fused_score in score_dict.items():
            point = point_dict[point_id]
            # Create new ScoredPoint with fused score
            fused_point = ScoredPoint(
                id=point.id,
                version=point.version,
                score=fused_score,
                payload=point.payload,
                vector=point.vector,
            )
            fused_results.append(fused_point)

        # Sort by fused score descending
        fused_results.sort(key=lambda x: x.score, reverse=True)

        # Return top_k
        return fused_results[:top_k]

    def hybrid_search(
        self,
        query: str,
        collection: str,
        top_k: int = 25,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search using dense and sparse vectors with fusion.

        Args:
            query: User query text.
            collection: Qdrant collection name.
            top_k: Number of results to return.
            filters: Optional filters dict supporting only {"library": str, "version": str}.

        Returns:
            List of SearchResult objects sorted by relevance score.
        """
        logger.info(f"Performing hybrid search for query: '{query}' (top_k={top_k})")

        # Step 1: Generate embeddings
        dense_vector, sparse_dict = self.vector_service.embed_query(query)

        # Step 2: Convert dense vector to list (match ingestion behavior)
        if isinstance(dense_vector, np.ndarray):
            dense_list = dense_vector.tolist()
        else:
            dense_list = list(dense_vector)

        # Step 3: Convert sparse dict to SparseVector
        sparse_vector = convert_sparse_dict_to_qdrant_sparsevector(sparse_dict)

        # Step 4: Build filters if provided
        qdrant_filter = None
        if filters:
            # Validate filter keys
            allowed_keys = {"library", "version"}
            for key in filters:
                if key not in allowed_keys:
                    raise ValueError(
                        f"Invalid filter key: '{key}'. Allowed keys are: {allowed_keys}"
                    )

            filter_conditions = []
            if "library" in filters and filters["library"] is not None:
                filter_conditions.append(
                    FieldCondition(key="library", match=MatchValue(value=filters["library"]))
                )
            if "version" in filters and filters["version"] is not None:
                filter_conditions.append(
                    FieldCondition(key="version", match=MatchValue(value=filters["version"]))
                )
            if filter_conditions:
                qdrant_filter = Filter(must=filter_conditions)

        # Step 5: Try prefetch + fusion approach
        try:
            from qdrant_client import models

            # Calculate prefetch limit for retrieving candidate points
            prefetch_limit = max(self.min_prefetch_limit, top_k * self.prefetch_multiplier)

            # Construct prefetch queries for dense and sparse vectors
            prefetches = [
                models.Prefetch(
                    query=dense_list,
                    using=DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ]

            # Construct fusion query
            fusion_query = models.FusionQuery(fusion=models.Fusion.RRF)

            # Execute query using query_points with prefetch and fusion
            results = self.client.query_points(
                collection_name=collection,
                prefetch=prefetches,
                query=fusion_query,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points

            logger.info(f"Used Qdrant prefetch + fusion API, returned {len(results)} results")
        except (AttributeError, TypeError, ValueError) as e:
            # Fallback: fusion API not available, use Python RRF
            logger.info(f"Fusion API not available ({e}), falling back to Python RRF merge")

            # Calculate prefetch_k for fallback
            prefetch_k = max(self.min_prefetch_limit, top_k * self.prefetch_multiplier)

            # Execute separate dense search
            dense_results = self.client.query_points(
                collection_name=collection,
                query=dense_list,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points

            # Execute separate sparse search
            sparse_results = self.client.query_points(
                collection_name=collection,
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_k,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points

            # Merge using Python RRF
            results = self.reciprocal_rank_fusion(dense_results, sparse_results, top_k)

            logger.info(
                f"Used Python RRF fallback, returned {len(results)} results from "
                f"{len(dense_results)} dense + {len(sparse_results)} sparse candidates"
            )

        # Step 6: Map ScoredPoint results to SearchResult objects
        search_results = []
        for point in results:
            payload = point.payload or {}

            # Extract required fields
            chunk_id = payload.get("chunk_id", "")
            text = payload.get("text", "")
            source_path = payload.get("source_path", "")
            canonical_source_id = payload.get("canonical_source_id", "")

            # Extract optional fields
            header = payload.get("header")
            library = payload.get("library")
            version = payload.get("version")
            title = payload.get("title")

            # Build metadata dict from remaining payload fields
            metadata = {
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "chunk_id",
                    "text",
                    "source_path",
                    "canonical_source_id",
                    "header",
                    "library",
                    "version",
                    "title",
                }
            }

            search_result = SearchResult(
                chunk_id=chunk_id,
                score=point.score,
                text=text,
                source_path=source_path,
                canonical_source_id=canonical_source_id,
                header=header,
                library=library,
                version=version,
                title=title,
                metadata=metadata,
            )
            search_results.append(search_result)

        logger.info(f"Returning {len(search_results)} search results")
        return search_results


def hybrid_search(query: str, k: int = 25, filters: Optional[dict] = None) -> list[SearchResult]:
    """
    Perform hybrid search (sparse + dense) in Qdrant.

    This is a convenience wrapper that creates SearchService with default config.

    Args:
        query: User query text.
        k: Number of results to return.
        filters: Optional metadata filters (e.g., {"library": "pandas"}).

    Returns:
        List of SearchResult objects.
    """
    from src.config import QDRANT_COLLECTION_NAME

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_service = VectorService()
    search_service = SearchService(qdrant_client, vector_service)

    return search_service.hybrid_search(
        query=query, collection=QDRANT_COLLECTION_NAME, top_k=k, filters=filters
    )


def rerank_results(query: str, candidates: list[SearchResult]) -> list[SearchResult]:
    """
    Re-rank search results using BGE reranker.

    Args:
        query: Original user query.
        candidates: List of candidate SearchResult objects.

    Returns:
        Re-ranked list of SearchResult objects.
    """
    # TODO: Implement re-ranking using BGE-reranker-v2-m3
    logger.info(f"Re-ranking {len(candidates)} candidates")
    return candidates


def retrieve_context(
    query: str, top_k: int = 5, filters: Optional[dict] = None
) -> list[SearchResult]:
    """
    Complete retrieval pipeline: hybrid search + re-ranking.

    Args:
        query: User query text.
        top_k: Number of final results to return.
        filters: Optional metadata filters.

    Returns:
        Top-k re-ranked SearchResult objects.
    """
    # Step 1: Hybrid search for top ~25
    candidates = hybrid_search(query, k=25, filters=filters)

    # Step 2: Re-rank
    reranked = rerank_results(query, candidates)

    # Step 3: Return top-k
    return reranked[:top_k]


def generate_answer(query: str, context: list[SearchResult]) -> str:
    """
    Generate answer using Ollama LLM with provided context.

    Args:
        query: User query.
        context: List of SearchResult objects to use as context.

    Returns:
        Generated answer string.
    """
    # TODO: Implement Ollama API call with structured prompt
    logger.info(f"Generating answer for query with {len(context)} context chunks")
    return ""
