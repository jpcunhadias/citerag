"""Integration tests for Qdrant indexing (requires Qdrant running)."""

import uuid

import pytest
from qdrant_client.models import SparseVector

from src.models import DocumentChunk


@pytest.mark.integration
class TestQdrantIntegration:
    """Integration tests for Qdrant indexing (requires Qdrant running)."""

    @pytest.fixture
    def test_collection_name(self):
        """Generate unique test collection name."""
        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_qdrant_upsert_with_named_vectors(self, test_collection_name, tmp_path):
        """Test upserting chunks with named vectors to Qdrant."""
        from qdrant_client import QdrantClient

        from src.config import QDRANT_HOST, QDRANT_PORT
        from src.ingest import index_to_qdrant

        # Create test chunks with embeddings
        chunks = [
            DocumentChunk(
                chunk_id=uuid.uuid4().hex,
                text="Test chunk 1",
                dense_vector=[0.1, 0.2, 0.3],
                sparse_vector={1: 0.5, 3: 0.7, 2: 0.3},
                metadata={"source_path": "test.md", "title": "Test"},
            ),
            DocumentChunk(
                chunk_id=uuid.uuid4().hex,
                text="Test chunk 2",
                dense_vector=[0.4, 0.5, 0.6],
                sparse_vector={2: 0.4, 1: 0.6},
                metadata={"source_path": "test2.md", "title": "Test2"},
            ),
        ]

        # Index to Qdrant
        dense_dimension = 3
        index_to_qdrant(
            chunks,
            collection_name=test_collection_name,
            dense_dimension=dense_dimension,
        )

        # Verify collection was created
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        assert test_collection_name in collection_names

        # Verify points were upserted
        points = client.scroll(collection_name=test_collection_name, limit=10, with_vectors=True)[0]
        assert len(points) == 2

        # Verify named vectors structure
        point = points[0]
        assert "dense" in point.vector
        assert "sparse" in point.vector

        # Verify sparse vector is SparseVector object with sorted indices
        sparse_vec = point.vector["sparse"]
        assert isinstance(sparse_vec, SparseVector)
        assert sparse_vec.indices == sorted(sparse_vec.indices)  # Sorted ascending

        # Verify payload
        assert point.payload["text"] in ["Test chunk 1", "Test chunk 2"]
        assert "source_path" in point.payload

        # Cleanup: delete test collection
        client.delete_collection(collection_name=test_collection_name)
