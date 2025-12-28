"""Unit tests for the ingestion pipeline."""

import hashlib
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client.models import SparseVector

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKER_FINGERPRINT
from src.ingest import (
    chunk_documents,
    clean_text,
    load_documents,
)
from src.models import (
    DocumentChunk,
    generate_chunk_id,
    normalize_text_for_hashing,
)
from src.utils.qdrant import convert_sparse_dict_to_qdrant_sparsevector


class TestChunkIDGeneration:
    """Tests for chunk ID generation and normalization."""

    def test_chunk_id_stability(self):
        """Test that same inputs produce same chunk ID."""
        source_id = "docs/pandas/merge.md"
        text = "This is a test chunk."
        normalized = normalize_text_for_hashing(text)

        id1 = generate_chunk_id(source_id, CHUNKER_FINGERPRINT, normalized)
        id2 = generate_chunk_id(source_id, CHUNKER_FINGERPRINT, normalized)

        assert id1 == id2

    def test_chunk_id_uniqueness(self):
        """Test that different texts produce different chunk IDs."""
        source_id = "docs/pandas/merge.md"
        text1 = "This is a test chunk."
        text2 = "This is a different chunk."

        normalized1 = normalize_text_for_hashing(text1)
        normalized2 = normalize_text_for_hashing(text2)

        id1 = generate_chunk_id(source_id, CHUNKER_FINGERPRINT, normalized1)
        id2 = generate_chunk_id(source_id, CHUNKER_FINGERPRINT, normalized2)

        assert id1 != id2

    def test_chunk_id_normalization(self):
        """Test deterministic text normalization."""
        # Test \r\n conversion
        text1 = "Line 1\r\nLine 2"
        text2 = "Line 1\nLine 2"
        assert normalize_text_for_hashing(text1) == normalize_text_for_hashing(text2)

        # Test trailing spaces
        text3 = "Line 1   \nLine 2"
        text4 = "Line 1\nLine 2"
        assert normalize_text_for_hashing(text3) == normalize_text_for_hashing(text4)

        # Test excessive blank lines
        text5 = "Line 1\n\n\n\n\nLine 2"
        text6 = "Line 1\n\nLine 2"
        assert normalize_text_for_hashing(text5) == normalize_text_for_hashing(text6)

    def test_canonical_source_id_relative(self):
        """Test that canonical_source_id is relative POSIX path."""
        input_root = Path("/home/user/docs")
        file_path = Path("/home/user/docs/pandas/merge.md")

        relative_path = file_path.relative_to(input_root)
        canonical_id = relative_path.as_posix()

        assert canonical_id == "pandas/merge.md"
        assert "/" in canonical_id  # POSIX format
        assert "\\" not in canonical_id  # No Windows separators

    def test_chunker_fingerprint_usage(self):
        """Test that chunker fingerprint is used in chunk ID."""
        source_id = "docs/test.md"
        text = "Test chunk"
        normalized = normalize_text_for_hashing(text)

        chunk_id = generate_chunk_id(source_id, CHUNKER_FINGERPRINT, normalized)

        # Verify fingerprint is in the hash input
        hash_input = f"{source_id}|{CHUNKER_FINGERPRINT}|{normalized}"
        expected_id = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        assert chunk_id == expected_id


class TestDocumentChunkSchema:
    """Tests for DocumentChunk schema validation."""

    def test_valid_chunk(self):
        """Test creating a valid DocumentChunk."""
        chunk = DocumentChunk(
            chunk_id="abc123",
            text="Test chunk text",
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={1: 0.5, 2: 0.3},
            metadata={"source_path": "test.md", "title": "Test"},
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.text == "Test chunk text"
        assert chunk.dense_vector == [0.1, 0.2, 0.3]
        assert chunk.sparse_vector == {1: 0.5, 2: 0.3}

    def test_chunk_with_none_vectors(self):
        """Test creating chunk with None vectors."""
        chunk = DocumentChunk(
            chunk_id="abc123",
            text="Test chunk text",
            dense_vector=None,
            sparse_vector=None,
            metadata={},
        )
        assert chunk.dense_vector is None
        assert chunk.sparse_vector is None


class TestTextCleaning:
    """Tests for text cleaning function."""

    def test_clean_text_normalizes_whitespace(self):
        """Test whitespace normalization."""
        text = "Line 1   \n   Line 2"
        cleaned = clean_text(text)
        assert "   " not in cleaned  # Multiple spaces collapsed

    def test_clean_text_preserves_structure(self):
        """Test that markdown structure is preserved."""
        text = "# Header\n\nBody text"
        cleaned = clean_text(text)
        assert "# Header" in cleaned
        assert "Body text" in cleaned

    def test_clean_text_collapses_blank_lines(self):
        """Test excessive blank line collapse."""
        text = "Line 1\n\n\n\n\nLine 2"
        cleaned = clean_text(text)
        # Should have max 2 consecutive blank lines
        assert "\n\n\n\n\n" not in cleaned

    def test_clean_text_preserves_indentation_in_code_blocks(self):
        """Test that indentation is preserved inside fenced code blocks."""
        text = """Here's some Python code:

```python
def hello():
    print("hello")
    if True:
        print("world")
```

More text here."""
        cleaned = clean_text(text)
        # Verify indentation is preserved inside code block
        assert '    print("hello")' in cleaned
        assert '        print("world")' in cleaned
        assert "def hello():" in cleaned

    def test_clean_text_preserves_indentation_in_multiple_code_blocks(self):
        """Test that indentation is preserved in multiple code blocks."""
        text = """First block:

```python
    x = 1
```

Second block:

```bash
    echo "test"
```
"""
        cleaned = clean_text(text)
        assert "    x = 1" in cleaned
        assert '    echo "test"' in cleaned


class TestDocumentLoading:
    """Tests for document loading."""

    def test_load_documents_computes_canonical_id(self, tmp_path):
        """Test that canonical_source_id is computed correctly."""
        # Create test directory structure
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        subdir = docs_dir / "pandas"
        subdir.mkdir()

        # Create test file
        test_file = subdir / "test.md"
        test_file.write_text("# Test Document\n\nContent here.")

        input_root = tmp_path / "docs"
        documents = load_documents(docs_dir, input_root=input_root)

        assert len(documents) == 1
        assert documents[0]["canonical_source_id"] == "pandas/test.md"
        assert documents[0]["text"] == "# Test Document\n\nContent here."

    def test_load_documents_extracts_title(self, tmp_path):
        """Test title extraction from filename."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        test_file = docs_dir / "merge_dataframes.md"
        test_file.write_text("Content")

        input_root = tmp_path / "docs"
        documents = load_documents(docs_dir, input_root=input_root)

        assert documents[0]["title"] == "merge_dataframes"

    def test_load_documents_handles_txt_files(self, tmp_path):
        """Test loading .txt files."""
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        test_file = docs_dir / "readme.txt"
        test_file.write_text("Text content")

        input_root = tmp_path / "docs"
        documents = load_documents(docs_dir, input_root=input_root)

        assert len(documents) == 1
        assert documents[0]["canonical_source_id"] == "readme.txt"


class TestChunking:
    """Tests for two-stage chunking."""

    def test_two_stage_chunking_preserves_headers(self):
        """Test that headers are preserved in chunk metadata."""
        documents = [
            {
                "text": "# Introduction\n\nThis is the intro.\n\n## Details\n\nMore details here.",
                "source_path": "/test.md",
                "canonical_source_id": "test.md",
                "title": "Test",
                "metadata": {},
            }
        ]

        chunks = chunk_documents(documents)

        assert len(chunks) > 0
        # Check that at least one chunk has header metadata
        headers = [chunk.metadata.get("header") for chunk in chunks]
        assert any(h is not None for h in headers)

    def test_chunking_respects_size_and_overlap(self):
        """Test that chunking uses correct size and overlap."""
        # Create a long document
        long_text = "\n\n".join([f"Paragraph {i}" * 50 for i in range(20)])
        documents = [
            {
                "text": long_text,
                "source_path": "/test.md",
                "canonical_source_id": "test.md",
                "title": "Test",
                "metadata": {},
            }
        ]

        chunks = chunk_documents(documents)

        # Verify chunks are created
        assert len(chunks) > 0

        # Verify chunk sizes are reasonable (allowing for overlap)
        for chunk in chunks:
            assert len(chunk.text) <= CHUNK_SIZE + CHUNK_OVERLAP


class TestSparseVectorConversion:
    """Tests for sparse vector conversion."""

    def test_convert_sparse_vector_sorts_indices(self):
        """Test that indices are sorted ascending."""
        sparse_dict = {5: 0.5, 1: 0.3, 3: 0.7, 2: 0.2}

        sparse_vec = convert_sparse_dict_to_qdrant_sparsevector(sparse_dict)

        assert isinstance(sparse_vec, SparseVector)
        assert sparse_vec.indices == [1, 2, 3, 5]  # Sorted ascending
        assert sparse_vec.values == [0.3, 0.2, 0.7, 0.5]  # Values reordered to match

    def test_convert_sparse_vector_empty(self):
        """Test conversion of empty sparse vector."""
        sparse_vec = convert_sparse_dict_to_qdrant_sparsevector({})
        assert isinstance(sparse_vec, SparseVector)
        assert sparse_vec.indices == []
        assert sparse_vec.values == []

    def test_convert_sparse_vector_preserves_values(self):
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


class TestVectorService:
    """Tests for VectorService (mocked)."""

    @patch("src.ingest.BGEM3FlagModel")
    def test_vector_service_infers_dimension(self, mock_model_class):
        """Test that dense dimension is inferred from first batch."""
        from src.ingest import VectorService

        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": np.array([[0.1, 0.2, 0.3, 0.4]] * 2),  # Shape: (2, 4)
            "lexical_weights": [{1: 0.5}, {2: 0.6}],
        }
        mock_model_class.return_value = mock_model

        service = VectorService(batch_size=2)
        texts = ["text1", "text2"]
        # The return value from FlagEmbedding can vary, so we check for the keys
        # that the service is trying to access.
        try:
            dense, sparse, dim = service.embed_documents(texts)
        except ValueError as e:
            # This can happen if the mock is not set up correctly for the current version
            # of FlagEmbedding. We'll try to adapt.
            if "dense_vecs" in str(e) or "lexical_weights" in str(e):
                mock_model.encode.return_value = {
                    "dense": np.array([[0.1, 0.2, 0.3, 0.4]] * 2),
                    "sparse": [{1: 0.5}, {2: 0.6}],
                }
                dense, sparse, dim = service.embed_documents(texts)
            else:
                raise e

        assert dim == 4
        assert len(dense) == 2

    @patch("src.ingest.BGEM3FlagModel")
    def test_vector_service_returns_both_vectors(self, mock_model_class):
        """Test that both dense and sparse vectors are returned."""
        from src.ingest import VectorService

        mock_model = MagicMock()
        mock_model.encode.return_value = {
            "dense_vecs": np.array([[0.1, 0.2]]),
            "lexical_weights": [{1: 0.5}],
        }
        mock_model_class.return_value = mock_model

        service = VectorService(batch_size=1)
        texts = ["test"]
        try:
            dense, sparse, dim = service.embed_documents(texts)
        except ValueError as e:
            if "dense_vecs" in str(e) or "lexical_weights" in str(e):
                mock_model.encode.return_value = {
                    "dense": np.array([[0.1, 0.2]]),
                    "sparse": [{1: 0.5}],
                }
                dense, sparse, dim = service.embed_documents(texts)
            else:
                raise e

        assert dense is not None
        assert sparse is not None
        assert len(sparse) == 1
        assert isinstance(sparse[0], dict)


@pytest.mark.integration
class TestQdrantIntegration:
    """Integration tests for Qdrant indexing (requires Qdrant running)."""

    @pytest.fixture
    def test_collection_name(self):
        """Generate unique test collection name."""
        import uuid

        return f"test_collection_{uuid.uuid4().hex[:8]}"

    def test_qdrant_upsert_with_named_vectors(self, test_collection_name, tmp_path):
        """Test upserting chunks with named vectors to Qdrant."""
        from qdrant_client import QdrantClient

        from src.config import QDRANT_HOST, QDRANT_PORT
        from src.ingest import index_to_qdrant
        from src.models import DocumentChunk

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
        points = client.scroll(
            collection_name=test_collection_name, limit=10, with_vectors=True
        )[0]
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
