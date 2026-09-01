"""Document ingestion pipeline: load, chunk, embed, and index documents."""

import logging
import re
import uuid
from pathlib import Path

import numpy as np
from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKER_FINGERPRINT,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
)
from src.devices import get_device
from src.models import DocumentChunk, generate_chunk_id, normalize_text_for_hashing
from src.utils.qdrant import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    convert_sparse_dict_to_qdrant_sparsevector,
)

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text while preserving markdown structure.

    Args:
        text: Raw text content.

    Returns:
        Cleaned text with normalized whitespace.
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse multiple spaces on each line and strip ends,
    # but preserve indentation for lines inside fenced code blocks or lists.
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    in_code_block = False  # Track whether we're inside a fenced code block

    for line in lines:
        stripped_line = line.strip()

        # Check if this line starts or ends a fenced code block
        if stripped_line.startswith("```"):
            in_code_block = not in_code_block
            cleaned_lines.append(line)  # Preserve the fence line as-is
        elif in_code_block:
            # Inside code block: preserve line exactly as-is (including indentation)
            cleaned_lines.append(line)
        elif stripped_line.startswith(("* ", "- ")) or re.match(r"^\d+\. ", stripped_line):
            # List items: preserve as-is (including indentation for nested items)
            # Matches bullet lists (*, -) and numbered lists (1., 2., 10., etc.)
            cleaned_lines.append(line)
        else:
            # Outside code blocks: normalize whitespace
            cleaned_lines.append(re.sub(r" +", " ", line).strip())

    text_cleaned_spaces = "\n".join(cleaned_lines)

    # Collapse excessive blank lines (max 2 consecutive newlines -> 1 blank line)
    return re.sub(r"\n{3,}", "\n\n", text_cleaned_spaces)


def load_documents(
    docs_path: Path,
    input_root: Path,
    library: str | None = None,
    version: str | None = None,
) -> list[dict]:
    """
    Load documents from a directory.

    Args:
        docs_path: Path to directory containing markdown/text files.
        input_root: Root path for computing canonical_source_id (relative paths).
        library: Optional library name for metadata.
        version: Optional library version for metadata.

    Returns:
        List of document dictionaries with 'text', 'source_path', 'canonical_source_id',
        'title', and 'metadata' keys.
    """
    logger.info(f"Loading documents from {docs_path}")
    documents: list[dict] = []

    if not docs_path.exists():
        logger.warning(f"Path does not exist: {docs_path}")
        return documents

    # Resolve absolute paths
    docs_path = docs_path.resolve()
    input_root = input_root.resolve()

    # Find all .md and .txt files recursively
    for file_path in docs_path.rglob("*.md"):
        try:
            content = file_path.read_text(encoding="utf-8")
            # Extract title from filename or first markdown header
            title = file_path.stem
            # Try to extract from first header
            first_line = content.split("\n")[0] if content else ""
            if first_line.startswith("#"):
                title = first_line.lstrip("#").strip()

            # Compute canonical_source_id: relative POSIX path from input root
            try:
                relative_path = file_path.relative_to(input_root)
                canonical_source_id = relative_path.as_posix()
            except ValueError:
                # If file is not under input_root, use filename
                logger.warning(
                    f"File {file_path} is not under input root {input_root}, using filename"
                )
                canonical_source_id = file_path.name

            documents.append(
                {
                    "text": content,
                    "source_path": str(file_path),
                    "canonical_source_id": canonical_source_id,
                    "title": title,
                    "metadata": {
                        "library": library,
                        "version": version,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")

    # Also load .txt files
    for file_path in docs_path.rglob("*.txt"):
        try:
            content = file_path.read_text(encoding="utf-8")
            title = file_path.stem

            # Compute canonical_source_id
            try:
                relative_path = file_path.relative_to(input_root)
                canonical_source_id = relative_path.as_posix()
            except ValueError:
                logger.warning(
                    f"File {file_path} is not under input root {input_root}, using filename"
                )
                canonical_source_id = file_path.name

            documents.append(
                {
                    "text": content,
                    "source_path": str(file_path),
                    "canonical_source_id": canonical_source_id,
                    "title": title,
                    "metadata": {
                        "library": library,
                        "version": version,
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def chunk_documents(documents: list[dict]) -> list[DocumentChunk]:
    """
    Split documents into chunks while preserving structure using two-stage chunking.

    Args:
        documents: List of document dictionaries with 'text', 'source_path',
                   'canonical_source_id', 'title', and 'metadata' keys.

    Returns:
        List of DocumentChunk objects with chunk_id, text, and metadata.
    """
    logger.info("Chunking documents using two-stage chunking")
    chunks: list[DocumentChunk] = []

    # Stage 1: MarkdownHeaderTextSplitter
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
    )

    # Stage 2: RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )

    for doc in documents:
        text = doc["text"]
        canonical_source_id = doc["canonical_source_id"]
        title = doc.get("title", "")
        metadata = doc.get("metadata", {})

        # Stage 1: Split by headers
        try:
            header_splits = header_splitter.split_text(text)
        except Exception as e:
            logger.warning(f"Error in header splitting for {canonical_source_id}: {e}")
            # Fallback: treat entire document as one section
            header_splits = [Document(page_content=text, metadata={})]

        # Stage 2: Split each section further
        for header_split in header_splits:
            section_text = header_split.page_content
            header_metadata = header_split.metadata
            # Extract header from metadata
            header = header_metadata.get("Header 1") or header_metadata.get("Header 2") or None

            if not section_text.strip():
                continue

            # Split section into chunks
            try:
                text_chunks = text_splitter.split_text(section_text)
            except Exception as e:
                logger.warning(f"Error in text splitting for {canonical_source_id}: {e}")
                text_chunks = [section_text]

            # Create DocumentChunk for each text chunk
            for chunk_text in text_chunks:
                if not chunk_text.strip():
                    continue

                # Normalize text for chunk ID generation
                normalized_text = normalize_text_for_hashing(chunk_text)

                # Generate chunk ID
                chunk_id = generate_chunk_id(
                    canonical_source_id, CHUNKER_FINGERPRINT, normalized_text
                )

                # Create DocumentChunk
                chunk_metadata = {
                    "source_path": doc["source_path"],
                    "canonical_source_id": canonical_source_id,
                    "title": title,
                    "header": header,
                    **metadata,
                }

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    dense_vector=None,
                    sparse_vector=None,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


class VectorService:
    """Service for generating embeddings using BGEM3FlagModel."""

    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        """
        Initialize VectorService with BGEM3FlagModel.

        Args:
            batch_size: Batch size for embedding generation.
        """
        self.batch_size = batch_size
        device = get_device()
        use_fp16 = device.type == "cuda"
        logger.info(f"Initializing BGEM3FlagModel with use_fp16={use_fp16}")
        self.model = BGEM3FlagModel(model_name_or_path=EMBEDDING_MODEL_NAME, use_fp16=use_fp16)
        self._dense_dimension: int | None = None

    def embed_documents(
        self, texts: list[str], show_progress: bool = True
    ) -> tuple[np.ndarray, list[dict[int, float]], int]:
        """
        Generate dense and sparse embeddings for documents.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to display a progress bar (default True).

        Returns:
            Tuple of (dense_vectors, sparse_vectors, dense_dimension):
            - dense_vectors: numpy array of shape (n_texts, dense_dim)
            - sparse_vectors: list of dicts mapping token_id to weight
            - dense_dimension: inferred dimension from first batch
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}")

        all_dense: list[np.ndarray] = []
        all_sparse: list[dict[int, float]] = []

        batch_range = range(0, len(texts), self.batch_size)
        if show_progress:
            batch_range = tqdm(batch_range, desc="Embedding", unit="batch")

        # Process in batches
        for i in batch_range:
            batch_texts = texts[i : i + self.batch_size]
            logger.debug(f"Processing batch {i // self.batch_size + 1} ({len(batch_texts)} texts)")

            # Encode batch
            result = self.model.encode(
                batch_texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            # Extract dense and sparse vectors
            # FlagEmbedding encode() returns a tuple when multiple return types are requested
            # Format: (dense_vecs, sparse_vecs) or dict with keys
            if isinstance(result, tuple):
                # Tuple format: (dense, sparse) or (dense, sparse, colbert)
                batch_dense = result[0]
                batch_sparse = result[1] if len(result) > 1 else None
            elif isinstance(result, dict):
                # Dict format: FlagEmbedding returns {'dense_vecs': ..., 'lexical_weights': ...}
                # Check for key existence first (can't use 'or' with numpy arrays)
                if "dense_vecs" in result:
                    batch_dense = result["dense_vecs"]
                elif "dense" in result:
                    batch_dense = result["dense"]
                else:
                    batch_dense = None

                # Sparse vectors are returned as 'lexical_weights' in FlagEmbedding
                if "lexical_weights" in result:
                    batch_sparse = result["lexical_weights"]
                elif "sparse_vecs" in result:
                    batch_sparse = result["sparse_vecs"]
                elif "sparse" in result:
                    batch_sparse = result["sparse"]
                else:
                    batch_sparse = None

                # Log available keys for debugging if not found
                if batch_dense is None or batch_sparse is None:
                    logger.error(f"Available keys in result: {list(result.keys())}")
                    available_keys = list(result.keys())
                    raise ValueError(
                        f"Could not find dense/sparse vectors in result. "
                        f"Available keys: {available_keys}"
                    )
            else:
                logger.error(f"Unexpected return type from model.encode(): {type(result)}")
                raise ValueError(f"Unexpected return type from model.encode(): {type(result)}")

            if batch_dense is None:
                raise ValueError("Dense vectors not returned from model.encode()")
            if batch_sparse is None:
                raise ValueError("Sparse vectors not returned from model.encode()")

            # Check for empty lists (defensive check to prevent length mismatch)
            if isinstance(batch_sparse, list) and len(batch_sparse) == 0:
                raise ValueError(
                    "Sparse vectors list is empty, which would cause length "
                    "mismatch with dense vectors"
                )

            # Infer dimension from first batch
            if self._dense_dimension is None and batch_dense.size > 0:
                self._dense_dimension = batch_dense.shape[1]
                logger.info(f"Inferred dense vector dimension: {self._dense_dimension}")

            all_dense.append(batch_dense)
            all_sparse.extend(batch_sparse)

        # Concatenate dense vectors
        dense_vectors = np.vstack(all_dense) if all_dense else np.array([])

        if self._dense_dimension is None:
            raise ValueError("Could not infer dense vector dimension")

        return dense_vectors, all_sparse, self._dense_dimension

    def embed_query(self, text: str) -> tuple[np.ndarray, dict[int, float]]:
        """
        Generate embeddings for a single query text.

        Args:
            text: Query text string.

        Returns:
            Tuple of (dense_vector, sparse_vector).
        """
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        # Handle different return formats (same as embed_documents)
        if isinstance(result, tuple):
            # Tuple format: (dense, sparse) or (dense, sparse, colbert)
            dense_vecs = result[0]
            sparse_vecs = result[1] if len(result) > 1 else None
        elif isinstance(result, dict):
            # Dict format: FlagEmbedding returns {'dense_vecs': ..., 'lexical_weights': ...}
            # Check for key existence first (can't use 'or' with numpy arrays)
            if "dense_vecs" in result:
                dense_vecs = result["dense_vecs"]
            elif "dense" in result:
                dense_vecs = result["dense"]
            else:
                dense_vecs = None

            # Sparse vectors are returned as 'lexical_weights' in FlagEmbedding
            if "lexical_weights" in result:
                sparse_vecs = result["lexical_weights"]
            elif "sparse_vecs" in result:
                sparse_vecs = result["sparse_vecs"]
            elif "sparse" in result:
                sparse_vecs = result["sparse"]
            else:
                sparse_vecs = None

            if dense_vecs is None or sparse_vecs is None:
                available_keys = list(result.keys())
                logger.error(f"Available keys in result: {available_keys}")
                raise ValueError(
                    f"Could not find dense/sparse vectors in result. "
                    f"Available keys: {available_keys}"
                )
        else:
            raise ValueError(f"Unexpected return type from model.encode(): {type(result)}")

        if dense_vecs is None or sparse_vecs is None:
            raise ValueError("Vectors not returned from model.encode()")

        # Check for empty lists (defensive check)
        if isinstance(sparse_vecs, list) and len(sparse_vecs) == 0:
            raise ValueError("Sparse vectors list is empty")

        dense_vec = dense_vecs[0] if isinstance(dense_vecs, (list, np.ndarray)) else dense_vecs
        sparse_vec = sparse_vecs[0] if isinstance(sparse_vecs, list) else sparse_vecs

        return dense_vec, sparse_vec


def chunk_id_to_point_id(chunk_id: str) -> uuid.UUID:
    """
    Convert a chunk_id (hex string) to a Qdrant-compatible point ID.

    Qdrant requires point IDs to be UUIDs or unsigned integers, so we parse
    the first 32 hex characters (128 bits) of the chunk_id hash directly as
    a UUID. This is deterministic: the same chunk_id always maps to the same
    point ID, which is what makes upsert/delete idempotent.

    Args:
        chunk_id: Hex string chunk ID (sha256 hexdigest or prefix of one).

    Returns:
        UUID derived from the first 32 hex characters of chunk_id.
    """
    return uuid.UUID(hex=chunk_id[:32])


def get_existing_chunk_ids_by_source(
    client: QdrantClient,
    collection_name: str,
    canonical_source_ids: set[str],
) -> dict[str, set[str]]:
    """
    Fetch chunk_ids already indexed in Qdrant for the given source files.

    Used to skip re-embedding chunks that are already indexed (same source,
    same text) and to detect stale chunks left behind when a file's content
    changes and its chunk boundaries shift.

    Args:
        client: Qdrant client.
        collection_name: Name of the Qdrant collection.
        canonical_source_ids: Source IDs to look up.

    Returns:
        Mapping of canonical_source_id to the set of chunk_ids currently
        indexed for it. Sources with nothing indexed map to an empty set.
    """
    existing: dict[str, set[str]] = {source_id: set() for source_id in canonical_source_ids}
    if not canonical_source_ids:
        return existing

    collections = client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        return existing

    scroll_filter = Filter(
        must=[
            FieldCondition(
                key="canonical_source_id",
                match=MatchAny(any=list(canonical_source_ids)),
            )
        ]
    )

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=["chunk_id", "canonical_source_id"],
            with_vectors=False,
            limit=500,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            source_id = payload.get("canonical_source_id")
            chunk_id = payload.get("chunk_id")
            if source_id in existing and chunk_id:
                existing[source_id].add(chunk_id)
        if offset is None:
            break

    return existing


def get_chunk_ids_outside_sources(
    client: QdrantClient,
    collection_name: str,
    current_source_ids: set[str],
) -> set[str]:
    """
    Find chunk_ids indexed in Qdrant whose source file no longer exists.

    Scans the whole collection, so this is only cheap for small/medium
    collections; used by --prune-missing to garbage-collect files that were
    deleted from disk between ingestion runs.

    Args:
        client: Qdrant client.
        collection_name: Name of the Qdrant collection.
        current_source_ids: canonical_source_ids currently present on disk.

    Returns:
        Set of chunk_ids whose canonical_source_id is not in
        current_source_ids.
    """
    collections = client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        return set()

    stale: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=["chunk_id", "canonical_source_id"],
            with_vectors=False,
            limit=500,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            source_id = payload.get("canonical_source_id")
            chunk_id = payload.get("chunk_id")
            if chunk_id and source_id not in current_source_ids:
                stale.add(chunk_id)
        if offset is None:
            break

    return stale


def delete_chunks(client: QdrantClient, collection_name: str, chunk_ids: set[str]) -> None:
    """
    Delete points from Qdrant by their original chunk_id.

    Args:
        client: Qdrant client.
        collection_name: Name of the Qdrant collection.
        chunk_ids: chunk_ids (hex strings) to delete.
    """
    if not chunk_ids:
        return

    point_ids: list[int | str | uuid.UUID] = [chunk_id_to_point_id(cid) for cid in chunk_ids]
    logger.info(f"Deleting {len(point_ids)} stale point(s) from '{collection_name}'")
    client.delete(collection_name=collection_name, points_selector=point_ids)


def index_to_qdrant(
    chunks: list[DocumentChunk],
    collection_name: str,
    dense_dimension: int,
    show_progress: bool = True,
) -> None:
    """
    Upsert chunks with embeddings to Qdrant.

    Args:
        chunks: List of DocumentChunk objects with embeddings.
        collection_name: Name of the Qdrant collection.
        dense_dimension: Dimension of dense vectors (inferred from model).
        show_progress: Whether to display a progress bar (default True).
    """
    logger.info(f"Indexing {len(chunks)} chunks to Qdrant collection '{collection_name}'")

    if not chunks:
        logger.warning("No chunks to index")
        return

    # Initialize Qdrant client
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Check if collection exists, create if not
    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)

    if not collection_exists:
        logger.info(
            f"Creating collection '{collection_name}' with dense_dimension={dense_dimension}"
        )
        # For named vectors with sparse support, use vectors_config for dense
        # and sparse_vectors_config for sparse
        vectors_config = {
            DENSE_VECTOR_NAME: VectorParams(size=dense_dimension, distance=Distance.COSINE),
        }
        sparse_vectors_config = {
            SPARSE_VECTOR_NAME: SparseVectorParams(),
        }
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists")

    # Prepare points for upsert
    points: list[PointStruct] = []
    for chunk in chunks:
        if chunk.dense_vector is None or chunk.sparse_vector is None:
            logger.warning(f"Chunk {chunk.chunk_id} missing vectors, skipping")
            continue

        # Convert sparse vector to SparseVector object
        sparse_vec_obj = convert_sparse_dict_to_qdrant_sparsevector(chunk.sparse_vector)

        # Create point
        point = PointStruct(
            id=chunk_id_to_point_id(chunk.chunk_id),
            vector={
                DENSE_VECTOR_NAME: chunk.dense_vector,
                SPARSE_VECTOR_NAME: sparse_vec_obj,
            },
            payload={
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,  # Store original chunk_id in payload
                **chunk.metadata,
            },
        )
        points.append(point)

    # Batch upsert
    batch_size = 100
    upsert_range = range(0, len(points), batch_size)
    if show_progress:
        upsert_range = tqdm(upsert_range, desc="Indexing", unit="batch")
    for i in upsert_range:
        batch = points[i : i + batch_size]
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    logger.info(f"Successfully indexed {len(points)} chunks to Qdrant")


def ingest_documents(
    docs_path: Path,
    collection_name: str,
    library: str | None = None,
    version: str | None = None,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    show_progress: bool = True,
    prune_missing: bool = False,
) -> None:
    """
    Incremental ingestion pipeline: load, clean, chunk, diff, embed, and index.

    chunk_id is a content hash (source + chunker config + text), so unchanged
    chunks already indexed under this collection are skipped rather than
    re-embedded, and chunks left behind by an edited file's shifted chunk
    boundaries are deleted. Set prune_missing=True to also delete chunks for
    files that no longer exist under docs_path at all — off by default since
    it assumes this collection is populated exclusively from docs_path; a
    collection aggregated from multiple --input directories would have
    unrelated sources wrongly deleted.

    Args:
        docs_path: Path to directory containing documents (used as input root).
        collection_name: Name of Qdrant collection.
        library: Optional library name for metadata.
        version: Optional library version for metadata.
        batch_size: Batch size for embedding generation.
        show_progress: Whether to display progress bars for embedding and indexing (default True).
        prune_missing: Also delete chunks for files no longer present under docs_path.
    """
    logger.info(f"Starting ingestion pipeline for {docs_path}")
    input_root = docs_path.resolve()

    # Step 1: Load documents
    logger.info("Step 1: Loading documents")
    documents = load_documents(docs_path, input_root=input_root, library=library, version=version)

    if not documents:
        logger.warning("No documents found, exiting pipeline")
        return

    # Step 2: Clean text
    logger.info("Step 2: Cleaning text")
    for doc in documents:
        doc["text"] = clean_text(doc["text"])

    # Step 3: Two-stage chunking
    logger.info("Step 3: Chunking documents")
    chunks = chunk_documents(documents)

    if not chunks:
        logger.warning("No chunks created, exiting pipeline")
        return

    # Step 4: Diff against what's already indexed for these sources
    logger.info("Step 4: Checking for already-indexed chunks")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    current_source_ids = {doc["canonical_source_id"] for doc in documents}
    existing_by_source = get_existing_chunk_ids_by_source(
        client, collection_name, current_source_ids
    )
    existing_chunk_ids = {cid for ids in existing_by_source.values() for cid in ids}
    current_chunk_ids = {chunk.chunk_id for chunk in chunks}

    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_chunk_ids]
    stale_chunk_ids = existing_chunk_ids - current_chunk_ids

    if prune_missing:
        stale_chunk_ids |= get_chunk_ids_outside_sources(
            client, collection_name, current_source_ids
        )

    skipped = len(chunks) - len(new_chunks)
    if skipped:
        logger.info(f"Skipping {skipped} chunk(s) already indexed and unchanged")

    # Step 5: Generate embeddings for new/changed chunks only
    if new_chunks:
        logger.info(f"Step 5: Generating embeddings for {len(new_chunks)} new/changed chunk(s)")
        vector_service = VectorService(batch_size=batch_size)
        texts = [chunk.text for chunk in new_chunks]
        dense_vectors, sparse_vectors, dense_dimension = vector_service.embed_documents(
            texts, show_progress=show_progress
        )

        for i, chunk in enumerate(new_chunks):
            chunk.dense_vector = dense_vectors[i].tolist()
            chunk.sparse_vector = sparse_vectors[i]

        # Step 6: Index to Qdrant
        logger.info("Step 6: Indexing to Qdrant")
        index_to_qdrant(
            new_chunks,
            collection_name=collection_name,
            dense_dimension=dense_dimension,
            show_progress=show_progress,
        )
    else:
        logger.info("Nothing new to embed or index")

    # Step 7: Clean up stale chunks (edited files' old boundaries, and
    # removed files if prune_missing)
    if stale_chunk_ids:
        logger.info(f"Step 7: Removing {len(stale_chunk_ids)} stale chunk(s)")
        delete_chunks(client, collection_name, stale_chunk_ids)

    logger.info("Ingestion pipeline completed successfully")
