#!/usr/bin/env python3
"""Quick verification script for Qdrant collection."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient

from src.config import QDRANT_HOST, QDRANT_PORT


def verify_collection(collection_name: str) -> int:
    """
    Verify a Qdrant collection exists and has expected structure.

    Args:
        collection_name: Name of the collection to verify.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [col.name for col in collections]

    if collection_name not in collection_names:
        print(f"[ERROR] Collection '{collection_name}' does not exist")
        print(f"Available collections: {collection_names}")
        return 1

    print(f"[SUCCESS] Collection '{collection_name}' exists")

    # Get collection info
    collection_info = client.get_collection(collection_name)
    print(f"   Points count: {collection_info.points_count}")

    if collection_info.points_count == 0:
        print("[WARNING] Collection is empty")
        return 1

    # Check vectors config
    vectors_config = collection_info.config.params.vectors
    if isinstance(vectors_config, dict):
        print(f"   Named vectors: {list(vectors_config.keys())}")
        if "dense" in vectors_config:
            dense_config = vectors_config["dense"]
            print(f"   Dense vector size: {dense_config.size}")
            print(f"   Dense vector distance: {dense_config.distance}")
        # Note: sparse vectors config is now in a separate part of the collection config
        if collection_info.config.params.sparse_vectors:
             print("   Sparse vector: configured")
    else:
        print(f"   Single vector size: {vectors_config.size}")

    # Sample a point (with vectors to verify structure)
    points, _ = client.scroll(
        collection_name=collection_name, limit=1, with_vectors=True
    )
    if points:
        point = points[0]
        print(f"\n--- Sample Point (ID: {point.id}) ---")
        print(f"   Payload keys: {list(point.payload.keys())}")

        # Check required payload fields
        required_fields = ["text", "canonical_source_id"]
        missing_fields = [f for f in required_fields if f not in point.payload]
        if missing_fields:
            print(f"   [WARNING] Missing payload fields: {missing_fields}")
        else:
            print("   [SUCCESS] All required payload fields present")

        # Check vectors
        if isinstance(point.vector, dict):
            print(f"   Vector keys: {list(point.vector.keys())}")
            if "dense" in point.vector:
                dense_vec = point.vector["dense"]
                print(f"   [SUCCESS] Dense vector: {len(dense_vec)} dimensions")
            if "sparse" in point.vector:
                sparse_vec = point.vector["sparse"]
                if hasattr(sparse_vec, "indices"):
                    print(
                        f"   [SUCCESS] Sparse vector: {len(sparse_vec.indices)} non-zero values"
                    )
                else:
                    print("   [SUCCESS] Sparse vector: present")
        else:
            print("   [WARNING] Single vector (not named vectors)")

        # Show sample payload
        print("\n   Sample payload:")
        print(f"   - text: {point.payload.get('text', 'N/A')[:100]}...")
        print(
            f"   - canonical_source_id: {point.payload.get('canonical_source_id', 'N/A')}"
        )
        print(f"   - header: {point.payload.get('header', 'N/A')}")
        print(f"   - source_path: {point.payload.get('source_path', 'N/A')}")

    print("\nVerification complete.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_qdrant.py <collection_name>")
        sys.exit(1)

    collection_name = sys.argv[1]
    sys.exit(verify_collection(collection_name))
