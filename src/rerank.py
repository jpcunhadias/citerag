"""Reranking service using BGE reranker model."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from FlagEmbedding import FlagReranker

from src.config import RERANKER_MODEL_NAME
from src.devices import get_device

if TYPE_CHECKING:
    from src.models import SearchResult

logger = logging.getLogger(__name__)


class RerankerService:
    """Service for reranking search results using BGE reranker."""

    def __init__(self):
        """Initialize RerankerService with FlagReranker model."""
        device = get_device()
        device_str = device.type  # Returns "cuda" or "cpu"
        logger.info(
            f"Initializing FlagReranker with model={RERANKER_MODEL_NAME}, device={device_str}"
        )
        self.reranker = FlagReranker(RERANKER_MODEL_NAME, device=device_str)
        # Ensure model is in eval mode
        if hasattr(self.reranker, "model"):
            self.reranker.model.eval()
        logger.info("Reranker initialized and set to eval mode")

    def rerank(self, query: str, results: list["SearchResult"], top_n: int) -> list["SearchResult"]:
        """
        Rerank search results using cross-encoder reranker.

        Args:
            query: User query text.
            results: List of SearchResult objects to rerank.
            top_n: Number of top results to return.

        Returns:
            List of SearchResult objects sorted by reranker score (descending).
        """
        if not results:
            logger.warning("Empty results list provided to rerank")
            return []

        logger.info(f"Reranking {len(results)} results for query: '{query[:50]}...'")

        # Create query-document pairs using the existing text field
        pairs = [[query, result.text] for result in results]

        # Compute reranker scores
        scores = self.reranker.compute_score(pairs)

        # Handle different return types (list, numpy array, tensor, or single value)
        if isinstance(scores, np.ndarray):
            scores = scores.tolist()
        elif not isinstance(scores, list):
            scores = [scores]

        # Update each SearchResult.score in place with the new reranker score
        for result, score in zip(results, scores):
            result.score = float(score)

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        # Return top_n results (or all if top_n > len(results))
        return sorted_results[:top_n]
