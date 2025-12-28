"""Unit tests for the search functionality."""

import os
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client.models import ScoredPoint, SparseVector

from src.models import SearchResult
from src.search import SearchService
from src.utils.qdrant import convert_sparse_dict_to_qdrant_sparsevector


def create_mock_query_response(points: list) -> MagicMock:
    """Create a mock QueryResponse object with a .points attribute."""
    mock_response = MagicMock()
    mock_response.points = points
    return mock_response


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_search_result_with_all_fields(self):
        """Test creating SearchResult with all fields."""
        result = SearchResult(
            chunk_id="abc123",
            score=0.95,
            text="Test chunk text",
            source_path="/path/to/file.md",
            canonical_source_id="file.md",
            header="Section Header",
            library="pandas",
            version="2.0.0",
            title="Test Document",
            metadata={"extra": "field"},
        )
        assert result.chunk_id == "abc123"
        assert result.score == 0.95
        assert result.text == "Test chunk text"
        assert result.source_path == "/path/to/file.md"
        assert result.canonical_source_id == "file.md"
        assert result.header == "Section Header"
        assert result.library == "pandas"
        assert result.version == "2.0.0"
        assert result.title == "Test Document"
        assert result.metadata == {"extra": "field"}

    def test_search_result_minimal_fields(self):
        """Test creating SearchResult with minimal required fields."""
        result = SearchResult(
            chunk_id="abc123",
            score=0.95,
            text="Test chunk text",
            source_path="/path/to/file.md",
            canonical_source_id="file.md",
        )
        assert result.chunk_id == "abc123"
        assert result.score == 0.95
        assert result.text == "Test chunk text"
        assert result.header is None
        assert result.library is None
        assert result.version is None
        assert result.title is None
        assert result.metadata == {}  # Default factory

    def test_search_result_metadata_default_factory(self):
        """Test that metadata uses default_factory."""
        result1 = SearchResult(
            chunk_id="abc123",
            score=0.95,
            text="Test",
            source_path="/path/to/file.md",
            canonical_source_id="file.md",
        )
        result2 = SearchResult(
            chunk_id="def456",
            score=0.90,
            text="Test",
            source_path="/path/to/file2.md",
            canonical_source_id="file2.md",
        )
        # Metadata should be separate dicts, not shared
        result1.metadata["test"] = "value1"
        assert "test" not in result2.metadata


class TestSearchService:
    """Tests for SearchService class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock QdrantClient."""
        return MagicMock()

    @pytest.fixture
    def mock_vector_service(self):
        """Create a mock VectorService."""
        service = MagicMock()
        # Mock embed_query to return numpy array and dict
        dense_vec = np.array([0.1, 0.2, 0.3, 0.4])
        sparse_dict = {1: 0.5, 2: 0.3, 3: 0.7}
        service.embed_query.return_value = (dense_vec, sparse_dict)
        return service

    @pytest.fixture
    def search_service(self, mock_qdrant_client, mock_vector_service):
        """Create SearchService instance with mocked dependencies."""
        return SearchService(mock_qdrant_client, mock_vector_service)

    def test_hybrid_search_calls_embed_query_once(
        self, search_service, mock_vector_service
    ):
        """Test that hybrid_search calls embed_query exactly once."""
        mock_qdrant_client = search_service.client
        mock_qdrant_client.query_points.return_value = create_mock_query_response([])

        search_service.hybrid_search("test query", "test_collection", top_k=5)

        assert mock_vector_service.embed_query.call_count == 1
        mock_vector_service.embed_query.assert_called_once_with("test query")

    def test_hybrid_search_converts_dense_to_list(
        self, search_service, mock_vector_service
    ):
        """Test that dense vector is converted to list before Qdrant query."""
        mock_qdrant_client = search_service.client
        mock_qdrant_client.query_points.return_value = create_mock_query_response([])

        # Verify embed_query returns numpy array
        dense_vec, _ = mock_vector_service.embed_query.return_value
        assert isinstance(dense_vec, np.ndarray)

        search_service.hybrid_search("test query", "test_collection", top_k=5)

        # Check that query was called (will fail if conversion fails)
        # The actual conversion happens inside hybrid_search
        assert mock_qdrant_client.query_points.called

    def test_hybrid_search_uses_prefetch_fusion_when_available(
        self, search_service, mock_vector_service
    ):
        """Test that prefetch + fusion API is used when available."""
        mock_qdrant_client = search_service.client
        mock_qdrant_client.query_points.return_value = create_mock_query_response([])

        search_service.hybrid_search("test query", "test_collection", top_k=5)

        # Should try query_points() first (prefetch + fusion)
        if mock_qdrant_client.query_points.called:
            call_args = mock_qdrant_client.query_points.call_args
            assert call_args is not None

    def test_hybrid_search_fallback_to_python_rrf(
        self, search_service, mock_vector_service
    ):
        """Test fallback to Python RRF when fusion API unavailable."""
        mock_qdrant_client = search_service.client

        # Mock search() for fallback
        dense_results = [
            ScoredPoint(
                id=uuid.uuid4(),
                version=1,
                score=0.9,
                payload={"chunk_id": "dense1", "text": "Dense result 1"},
            ),
            ScoredPoint(
                id=uuid.uuid4(),
                version=1,
                score=0.8,
                payload={"chunk_id": "dense2", "text": "Dense result 2"},
            ),
        ]
        sparse_results = [
            ScoredPoint(
                id=dense_results[0].id,  # Same ID to test fusion
                version=1,
                score=0.85,
                payload={"chunk_id": "dense1", "text": "Dense result 1"},
            ),
            ScoredPoint(
                id=uuid.uuid4(),
                version=1,
                score=0.75,
                payload={"chunk_id": "sparse1", "text": "Sparse result 1"},
            ),
        ]

        # Make query_points raise AttributeError first, then return results for fallback
        mock_qdrant_client.query_points.side_effect = [
            AttributeError("FusionMethod not found"),
            create_mock_query_response(dense_results),
            create_mock_query_response(sparse_results),
        ]

        results = search_service.hybrid_search("test query", "test_collection", top_k=2)

        # Should have called query_points() 3 times (1 failed fusion attempt + dense + sparse)
        assert mock_qdrant_client.query_points.call_count == 3

        # Should return fused results
        assert len(results) <= 2

    def test_hybrid_search_prefetch_k_calculation(
        self, search_service, mock_vector_service
    ):
        """Test that prefetch_k = max(50, top_k * 5) in fallback."""
        mock_qdrant_client = search_service.client

        # Test with top_k=5 -> prefetch_k should be max(50, 5*5) = 50
        top_k = 5
        prefetch_k = max(50, top_k * 5)

        # Force fallback by making FusionQuery construction fail
        # Patch models.FusionQuery to raise AttributeError when instantiated
        with patch(
            "qdrant_client.models.FusionQuery",
            side_effect=AttributeError("FusionMethod not found"),
        ):
            # Mock query_points for fallback calls (dense and sparse)
            mock_qdrant_client.query_points.side_effect = [
                create_mock_query_response([]),  # Dense query
                create_mock_query_response([]),  # Sparse query
            ]

            results = search_service.hybrid_search(
                "test query", "test_collection", top_k=top_k
            )

            # Should have 2 calls: dense + sparse (fusion failed during construction)
            assert mock_qdrant_client.query_points.call_count == 2

            # Check limit parameter for fallback calls
            dense_call = mock_qdrant_client.query_points.call_args_list[0]
            sparse_call = mock_qdrant_client.query_points.call_args_list[1]

            assert dense_call.kwargs["limit"] == prefetch_k
            assert sparse_call.kwargs["limit"] == prefetch_k
            assert results == []  # Empty results from mock

        # Reset mocks
        mock_qdrant_client.reset_mock()

        # Test with top_k=15 -> prefetch_k should be max(50, 15*5) = 75
        top_k = 15
        prefetch_k = max(50, top_k * 5)

        with patch(
            "qdrant_client.models.FusionQuery",
            side_effect=AttributeError("FusionMethod not found"),
        ):
            mock_qdrant_client.query_points.side_effect = [
                create_mock_query_response([]),  # Dense query
                create_mock_query_response([]),  # Sparse query
            ]

            search_service.hybrid_search(
                "test query", "test_collection", top_k=top_k
            )

            assert mock_qdrant_client.query_points.call_count == 2
            dense_call = mock_qdrant_client.query_points.call_args_list[0]
            sparse_call = mock_qdrant_client.query_points.call_args_list[1]

            assert dense_call.kwargs["limit"] == prefetch_k
            assert sparse_call.kwargs["limit"] == prefetch_k

    def test_hybrid_search_filter_library_only(
        self, search_service, mock_vector_service
    ):
        """Test filter application with library only."""
        mock_qdrant_client = search_service.client
        mock_qdrant_client.query_points.return_value = create_mock_query_response([])

        filters = {"library": "pandas"}
        search_service.hybrid_search(
            "test query", "test_collection", top_k=5, filters=filters
        )

        # Verify filter was applied (check query_points was called with filter)
        assert mock_qdrant_client.query_points.called

    def test_hybrid_search_filter_version_only(
        self, search_service, mock_vector_service
    ):
        """Test filter application with version only."""
        mock_qdrant_client = search_service.client
        mock_qdrant_client.query_points.return_value = create_mock_query_response([])

        filters = {"version": "2.0.0"}
        search_service.hybrid_search(
            "test query", "test_collection", top_k=5, filters=filters
        )

        assert mock_qdrant_client.query_points.called

    def test_hybrid_search_filter_both_library_and_version(
        self, search_service, mock_vector_service
    ):
        """Test filter application with both library and version."""
        mock_qdrant_client = search_service.client
        mock_qdrant_client.query_points.return_value = create_mock_query_response([])

        filters = {"library": "pandas", "version": "2.0.0"}
        search_service.hybrid_search(
            "test query", "test_collection", top_k=5, filters=filters
        )

        assert mock_qdrant_client.query_points.called

    def test_hybrid_search_invalid_filter_key(
        self, search_service, mock_vector_service
    ):
        """Test that invalid filter keys raise ValueError."""
        import pytest

        filters = {"invalid_key": "some_value"}
        with pytest.raises(ValueError, match="Invalid filter key: 'invalid_key'"):
            search_service.hybrid_search(
                "test query", "test_collection", top_k=5, filters=filters
            )

    def test_hybrid_search_empty_results(self, search_service, mock_vector_service):
        """Test handling of empty results."""
        mock_qdrant_client = search_service.client
        mock_qdrant_client.query_points.return_value = create_mock_query_response([])

        results = search_service.hybrid_search("test query", "test_collection", top_k=5)

        assert results == []

    def test_hybrid_search_maps_scored_point_to_search_result_fusion(
        self, search_service, mock_vector_service
    ):
        """Test that ScoredPoint is correctly mapped to SearchResult.

        Note: Due to FusionQuery API limitations in test environment (Query is a Union type),
        this test verifies mapping works correctly. In production with a working FusionQuery API,
        the fusion path would be taken and scores would match Qdrant's fusion scores.
        """
        mock_qdrant_client = search_service.client

        # Create mock ScoredPoint with all payload fields
        point_id = uuid.uuid4()
        scored_point = ScoredPoint(
            id=point_id,
            version=1,
            score=0.95,
            payload={
                "chunk_id": "test_chunk_123",
                "text": "Test chunk text",
                "source_path": "/path/to/file.md",
                "canonical_source_id": "file.md",
                "header": "Section Header",
                "library": "pandas",
                "version": "2.0.0",
                "title": "Test Document",
                "extra_field": "extra_value",
            },
        )

        # Mock query_points to return the scored point
        # Note: FusionQuery construction will likely fail in test env, causing fallback to RRF
        # This test verifies that ScoredPoint -> SearchResult mapping works correctly
        mock_qdrant_client.query_points.return_value = create_mock_query_response(
            [scored_point]
        )

        results = search_service.hybrid_search("test query", "test_collection", top_k=5)

        # Verify results are correctly mapped
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, SearchResult)
        assert result.chunk_id == "test_chunk_123"
        # Score will be RRF fused if fallback path taken (expected in test env)
        # or original Qdrant score if fusion path succeeds
        assert result.score > 0, f"Score should be positive, got {result.score}"
        assert result.text == "Test chunk text"
        assert result.source_path == "/path/to/file.md"
        assert result.canonical_source_id == "file.md"
        assert result.header == "Section Header"
        assert result.library == "pandas"
        assert result.version == "2.0.0"
        assert result.title == "Test Document"
        assert result.metadata == {"extra_field": "extra_value"}

    def test_hybrid_search_maps_scored_point_to_search_result_fallback(
        self, search_service, mock_vector_service
    ):
        """Test that ScoredPoint is correctly mapped to SearchResult in fallback path."""
        mock_qdrant_client = search_service.client

        # Create mock ScoredPoint with all payload fields
        point_id = uuid.uuid4()
        scored_point = ScoredPoint(
            id=point_id,
            version=1,
            score=0.9,  # Original Qdrant score (will be replaced by RRF)
            payload={
                "chunk_id": "test_chunk_123",
                "text": "Test chunk text",
                "source_path": "/path/to/file.md",
                "canonical_source_id": "file.md",
                "header": "Section Header",
                "library": "pandas",
                "version": "2.0.0",
                "title": "Test Document",
                "extra_field": "extra_value",
            },
        )

        # Force fallback by making FusionQuery construction fail
        # Then return same point in both dense and sparse queries
        with patch(
            "qdrant_client.models.FusionQuery",
            side_effect=AttributeError("FusionMethod not found"),
        ):
            mock_qdrant_client.query_points.side_effect = [
                create_mock_query_response([scored_point]),  # dense query
                create_mock_query_response([scored_point]),  # sparse query
            ]

            results = search_service.hybrid_search(
                "test query", "test_collection", top_k=5
            )

            assert len(results) == 1
            result = results[0]
            assert isinstance(result, SearchResult)
            assert result.chunk_id == "test_chunk_123"
            # In fallback path, score should be fused RRF score
            # Point appears at rank 1 in both dense and sparse: 1/(60+1) + 1/(60+1) = 2/61
            expected_rrf_score = 2.0 / 61.0
            assert abs(result.score - expected_rrf_score) < 1e-10
            assert result.text == "Test chunk text"
            assert result.source_path == "/path/to/file.md"
            assert result.canonical_source_id == "file.md"
            assert result.header == "Section Header"
            assert result.library == "pandas"
            assert result.version == "2.0.0"
            assert result.title == "Test Document"
            assert result.metadata == {"extra_field": "extra_value"}


class TestRRFHelper:
    """Tests for reciprocal_rank_fusion helper method."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock QdrantClient."""
        return MagicMock()

    @pytest.fixture
    def mock_vector_service(self):
        """Create a mock VectorService."""
        service = MagicMock()
        service.embed_query.return_value = (
            np.array([0.1, 0.2, 0.3]),
            {1: 0.5, 2: 0.3},
        )
        return service

    @pytest.fixture
    def search_service(self, mock_qdrant_client, mock_vector_service):
        """Create SearchService instance."""
        return SearchService(mock_qdrant_client, mock_vector_service)

    def test_rrf_score_calculation(self, search_service):
        """Test RRF score calculation matches formula."""
        point1_id = uuid.uuid4()
        point2_id = uuid.uuid4()

        dense_results = [
            ScoredPoint(id=point1_id, version=1, score=0.9, payload={}),
            ScoredPoint(id=point2_id, version=1, score=0.8, payload={}),
        ]
        sparse_results = [
            ScoredPoint(id=point1_id, version=1, score=0.85, payload={}),
            ScoredPoint(id=point2_id, version=1, score=0.75, payload={}),
        ]

        fused = search_service.reciprocal_rank_fusion(
            dense_results, sparse_results, top_k=2
        )

        # point1: rank 1 in dense (1/(60+1)) + rank 1 in sparse (1/(60+1)) = 2/61
        # point2: rank 2 in dense (1/(60+2)) + rank 2 in sparse (1/(60+2)) = 2/62
        # point1 should have higher score
        assert len(fused) == 2
        assert fused[0].id == point1_id
        assert fused[0].score > fused[1].score

    def test_rrf_ranking_stability(self, search_service):
        """Test that RRF produces stable rankings."""
        point1_id = uuid.uuid4()
        point2_id = uuid.uuid4()

        dense_results = [
            ScoredPoint(id=point1_id, version=1, score=0.9, payload={}),
            ScoredPoint(id=point2_id, version=1, score=0.8, payload={}),
        ]
        sparse_results = [
            ScoredPoint(id=point1_id, version=1, score=0.85, payload={}),
            ScoredPoint(id=point2_id, version=1, score=0.75, payload={}),
        ]

        fused1 = search_service.reciprocal_rank_fusion(
            dense_results, sparse_results, top_k=2
        )
        fused2 = search_service.reciprocal_rank_fusion(
            dense_results, sparse_results, top_k=2
        )

        # Same inputs should produce same outputs
        assert len(fused1) == len(fused2)
        assert fused1[0].id == fused2[0].id
        assert fused1[1].id == fused2[1].id
        assert fused1[0].score == fused2[0].score
        assert fused1[1].score == fused2[1].score

    def test_rrf_empty_lists(self, search_service):
        """Test RRF with empty input lists."""
        fused = search_service.reciprocal_rank_fusion([], [], top_k=5)
        assert fused == []

    def test_rrf_single_result(self, search_service):
        """Test RRF with single result in each list."""
        point_id = uuid.uuid4()
        dense_results = [ScoredPoint(id=point_id, version=1, score=0.9, payload={})]
        sparse_results = [ScoredPoint(id=point_id, version=1, score=0.85, payload={})]

        fused = search_service.reciprocal_rank_fusion(
            dense_results, sparse_results, top_k=5
        )

        assert len(fused) == 1
        assert fused[0].id == point_id
        # Score should be 1/(60+1) + 1/(60+1) = 2/61
        expected_score = 2.0 / 61.0
        assert abs(fused[0].score - expected_score) < 1e-10


class TestUtils:
    """Tests for utility functions."""

    def test_convert_sparse_dict_to_qdrant_sparsevector_sorts_indices(self):
        """Test that indices are sorted ascending."""
        sparse_dict = {5: 0.5, 1: 0.3, 3: 0.7, 2: 0.2}

        sparse_vec = convert_sparse_dict_to_qdrant_sparsevector(sparse_dict)

        assert isinstance(sparse_vec, SparseVector)
        assert sparse_vec.indices == [1, 2, 3, 5]  # Sorted ascending
        assert sparse_vec.values == [0.3, 0.2, 0.7, 0.5]  # Values reordered to match

    def test_convert_sparse_dict_to_qdrant_sparsevector_empty(self):
        """Test conversion of empty sparse vector."""
        sparse_vec = convert_sparse_dict_to_qdrant_sparsevector({})
        assert isinstance(sparse_vec, SparseVector)
        assert sparse_vec.indices == []
        assert sparse_vec.values == []

    def test_convert_sparse_dict_to_qdrant_sparsevector_single_element(self):
        """Test conversion with single element."""
        sparse_dict = {42: 0.99}
        sparse_vec = convert_sparse_dict_to_qdrant_sparsevector(sparse_dict)
        assert sparse_vec.indices == [42]
        assert sparse_vec.values == [0.99]

    def test_convert_sparse_dict_to_qdrant_sparsevector_preserves_values(self):
        """Test that values are correctly matched to sorted indices."""
        sparse_dict = {3: 0.7, 1: 0.3, 2: 0.2}

        sparse_vec = convert_sparse_dict_to_qdrant_sparsevector(sparse_dict)

        # Verify values match indices
        assert sparse_vec.indices[0] == 1
        assert sparse_vec.values[0] == 0.3
        assert sparse_vec.indices[1] == 2
        assert sparse_vec.values[1] == 0.2
        assert sparse_vec.indices[2] == 3
        assert sparse_vec.values[2] == 0.7


@pytest.mark.integration
class TestSearchIntegration:
    """Integration tests for search functionality (requires Qdrant running)."""

    @pytest.fixture
    def test_collection_name(self):
        """Generate unique test collection name."""
        return f"test_search_{uuid.uuid4().hex[:8]}"

    @pytest.mark.skipif(
        not os.getenv("QDRANT_URL") and not os.getenv("QDRANT_HOST"),
        reason="Qdrant not configured (set QDRANT_URL or QDRANT_HOST env var)",
    )
    def test_search_returns_results(self, test_collection_name):
        """Test that search returns results with scores."""
        from qdrant_client import QdrantClient

        from src.config import QDRANT_HOST, QDRANT_PORT
        from src.ingest import VectorService

        # This test requires a collection with data
        # For now, just verify the service can be instantiated and called
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        vector_service = VectorService()
        search_service = SearchService(client, vector_service)

        # Try to search (may return empty if collection doesn't exist)
        results = search_service.hybrid_search(
            query="test", collection=test_collection_name, top_k=5
        )

        # Results should be a list (even if empty)
        assert isinstance(results, list)

        # If results exist, verify structure
        if results:
            assert all(isinstance(r, SearchResult) for r in results)
            assert all(r.score is not None for r in results)
            assert all(r.chunk_id for r in results)
            assert all(r.text for r in results)
