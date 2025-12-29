"""Unit tests for the reranking service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import SearchResult
from src.rerank import RerankerService


class TestRerankerService:
    """Tests for RerankerService class."""

    @pytest.fixture
    def mock_flag_reranker(self):
        """Create a mock FlagReranker."""
        mock_reranker = MagicMock()
        mock_reranker.model = MagicMock()
        mock_reranker.model.eval = MagicMock()
        return mock_reranker

    @pytest.fixture
    def sample_results(self):
        """Create sample SearchResult objects for testing."""
        return [
            SearchResult(
                chunk_id="chunk1",
                score=0.1,
                text="Apple is a fruit",
                source_path="test1.md",
                canonical_source_id="test1.md",
            ),
            SearchResult(
                chunk_id="chunk2",
                score=0.2,
                text="Car is a vehicle",
                source_path="test2.md",
                canonical_source_id="test2.md",
            ),
            SearchResult(
                chunk_id="chunk3",
                score=0.3,
                text="Python is a programming language",
                source_path="test3.md",
                canonical_source_id="test3.md",
            ),
        ]

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_reranker_initialization(self, mock_get_device, mock_flag_reranker_class):
        """Test that RerankerService initializes correctly."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        mock_reranker.model = MagicMock()
        mock_reranker.model.eval = MagicMock()
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        assert service.reranker == mock_reranker
        mock_flag_reranker_class.assert_called_once()
        mock_reranker.model.eval.assert_called_once()

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_reranker_initialization_cpu(self, mock_get_device, mock_flag_reranker_class):
        """Test that RerankerService works with CPU device."""
        mock_device = MagicMock()
        mock_device.type = "cpu"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        mock_reranker.model = MagicMock()
        mock_reranker.model.eval = MagicMock()
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        assert service.reranker == mock_reranker
        # Should still call eval even on CPU
        mock_reranker.model.eval.assert_called_once()

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_reranker_initialization_no_model_attr(self, mock_get_device, mock_flag_reranker_class):
        """Test that RerankerService handles reranker without model attribute."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        # No model attribute
        mock_flag_reranker_class.return_value = mock_reranker

        # Should not raise an error
        service = RerankerService()
        assert service.reranker == mock_reranker

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_updates_scores_in_place(
        self, mock_get_device, mock_flag_reranker_class, sample_results
    ):
        """Test that rerank updates scores in place."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        # Return list of scores (higher score for first result)
        mock_reranker.compute_score.return_value = [0.9, 0.5, 0.3]
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()
        original_scores = [r.score for r in sample_results]

        reranked = service.rerank("fruit", sample_results, top_n=3)

        # Scores should be updated
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.5
        assert reranked[2].score == 0.3
        # Original scores should be different
        assert reranked[0].score != original_scores[0]

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_sorts_by_score_descending(
        self, mock_get_device, mock_flag_reranker_class, sample_results
    ):
        """Test that rerank sorts results by score descending."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        # Return scores in descending order
        mock_reranker.compute_score.return_value = [0.9, 0.5, 0.3]
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        reranked = service.rerank("query", sample_results, top_n=3)

        # Should be sorted descending
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.5
        assert reranked[2].score == 0.3
        assert reranked[0].score >= reranked[1].score >= reranked[2].score

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_handles_empty_results(self, mock_get_device, mock_flag_reranker_class):
        """Test that rerank handles empty results list."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        reranked = service.rerank("query", [], top_n=5)

        assert reranked == []
        mock_reranker.compute_score.assert_not_called()

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_respects_top_n_limit(
        self, mock_get_device, mock_flag_reranker_class, sample_results
    ):
        """Test that rerank respects top_n limit."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = [0.9, 0.5, 0.3]
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        reranked = service.rerank("query", sample_results, top_n=2)

        assert len(reranked) == 2
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.5

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_handles_top_n_greater_than_results(
        self, mock_get_device, mock_flag_reranker_class, sample_results
    ):
        """Test that rerank handles top_n greater than number of results."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = [0.9, 0.5, 0.3]
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        reranked = service.rerank("query", sample_results, top_n=10)

        assert len(reranked) == 3  # Should return all results

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_handles_numpy_array_scores(
        self, mock_get_device, mock_flag_reranker_class, sample_results
    ):
        """Test that rerank handles numpy array scores."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        # Return numpy array instead of list
        mock_reranker.compute_score.return_value = np.array([0.9, 0.5, 0.3])
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        reranked = service.rerank("query", sample_results, top_n=3)

        assert len(reranked) == 3
        assert reranked[0].score == 0.9
        assert isinstance(reranked[0].score, float)

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_handles_single_score(self, mock_get_device, mock_flag_reranker_class):
        """Test that rerank handles single score (not a list)."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        # Return single float instead of list
        mock_reranker.compute_score.return_value = 0.8
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        single_result = [
            SearchResult(
                chunk_id="chunk1",
                score=0.1,
                text="Test text",
                source_path="test.md",
                canonical_source_id="test.md",
            )
        ]

        reranked = service.rerank("query", single_result, top_n=1)

        assert len(reranked) == 1
        assert reranked[0].score == 0.8

    @patch("src.rerank.FlagReranker")
    @patch("src.rerank.get_device")
    def test_rerank_creates_correct_pairs(
        self, mock_get_device, mock_flag_reranker_class, sample_results
    ):
        """Test that rerank creates correct query-document pairs."""
        mock_device = MagicMock()
        mock_device.type = "cuda"
        mock_get_device.return_value = mock_device

        mock_reranker = MagicMock()
        mock_reranker.compute_score.return_value = [0.9, 0.5, 0.3]
        mock_flag_reranker_class.return_value = mock_reranker

        service = RerankerService()

        query = "test query"
        service.rerank(query, sample_results, top_n=3)

        # Verify compute_score was called with correct pairs
        mock_reranker.compute_score.assert_called_once()
        call_args = mock_reranker.compute_score.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0] == [query, sample_results[0].text]
        assert call_args[1] == [query, sample_results[1].text]
        assert call_args[2] == [query, sample_results[2].text]
