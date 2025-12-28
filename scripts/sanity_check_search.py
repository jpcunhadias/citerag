#!/usr/bin/env python3
"""Sanity checks for search functionality."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.ingest import VectorService, index_to_qdrant
from src.search import SearchService
from src.utils.qdrant import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


def check_vector_names():
    """Check 1A: Vector names match ingestion."""
    print("=" * 60)
    print("Check 1A: Vector names match ingestion")
    print("=" * 60)
    print(f"DENSE_VECTOR_NAME = '{DENSE_VECTOR_NAME}'")
    print(f"SPARSE_VECTOR_NAME = '{SPARSE_VECTOR_NAME}'")

    # Check ingestion uses these names
    import inspect
    source = inspect.getsource(index_to_qdrant)
    if f"DENSE_VECTOR_NAME" in source:
        print("✅ Ingestion uses DENSE_VECTOR_NAME constant")
    else:
        print("⚠️  Check if ingestion uses DENSE_VECTOR_NAME constant")

    if f"SPARSE_VECTOR_NAME" in source:
        print("✅ Ingestion uses SPARSE_VECTOR_NAME constant")
    else:
        print("⚠️  Check if ingestion uses SPARSE_VECTOR_NAME constant")
    print()


def check_payload_keys():
    """Check 1B: Payload keys match exactly."""
    print("=" * 60)
    print("Check 1B: Payload keys match exactly")
    print("=" * 60)
    required_keys = [
        "chunk_id",
        "text",
        "canonical_source_id",
        "source_path",
        "header",
        "library",
        "version",
        "title",
    ]
    print("Required payload keys:")
    for key in required_keys:
        print(f"  - {key}")

    # Check search code extracts these
    from src.search import SearchService
    import inspect
    source = inspect.getsource(SearchService.hybrid_search)

    missing = []
    for key in required_keys:
        if f'"{key}"' in source or f"'{key}'" in source:
            print(f"✅ Search extracts '{key}'")
        else:
            missing.append(key)
            print(f"❌ Search missing '{key}'")

    if missing:
        print(f"\n⚠️  Missing keys in search extraction: {missing}")
    else:
        print("\n✅ All required keys are extracted")
    print()


def check_embed_query_shape():
    """Check 1C: embed_query shape and list conversion."""
    print("=" * 60)
    print("Check 1C: embed_query shape and list conversion")
    print("=" * 60)

    vector_service = VectorService()
    dense_vec, sparse_dict = vector_service.embed_query("test query")

    print(f"Dense vector type: {type(dense_vec)}")
    print(f"Dense vector shape: {dense_vec.shape if hasattr(dense_vec, 'shape') else 'N/A'}")
    print(f"Sparse dict type: {type(sparse_dict)}")
    print(f"Sparse dict keys (sample): {list(sparse_dict.keys())[:5] if sparse_dict else 'empty'}")

    # Check conversion
    import numpy as np
    if isinstance(dense_vec, np.ndarray):
        dense_list = dense_vec.tolist()
        print(f"✅ Dense vector converted to list: {type(dense_list)}")
        print(f"   List length: {len(dense_list)}")
    else:
        print("⚠️  Dense vector is not numpy array")

    from src.utils.qdrant import convert_sparse_dict_to_qdrant_sparsevector
    sparse_vector = convert_sparse_dict_to_qdrant_sparsevector(sparse_dict)
    print(f"✅ Sparse vector converted: {type(sparse_vector)}")
    print(f"   Indices sorted: {sparse_vector.indices == sorted(sparse_vector.indices)}")
    print()


def check_fusion_api():
    """Check 2: Fusion API correctness."""
    print("=" * 60)
    print("Check 2: Fusion API correctness")
    print("=" * 60)

    from src.search import SearchService
    import inspect
    source = inspect.getsource(SearchService.hybrid_search)

    # Check fusion API uses Prefetch + FusionQuery
    if "Prefetch" in source and "FusionQuery" in source:
        print("✅ Fusion API uses Prefetch with FusionQuery")
    else:
        print("❌ Fusion API does not appear to use Prefetch/FusionQuery")

    # Check with_payload
    payload_count = source.count("with_payload=True")
    print(f"✅ with_payload=True appears {payload_count} times")
    if payload_count >= 3:
        print("   (fusion query + 2 fallback queries)")
    else:
        print(f"   ⚠️  Expected 3, found {payload_count}")
    print()


def check_fallback_path():
    """Check 3: Fallback path correctness."""
    print("=" * 60)
    print("Check 3: Fallback path correctness (RRF in Python)")
    print("=" * 60)

    from src.search import SearchService
    import inspect

    # Check prefetch_limit/prefetch_k
    source = inspect.getsource(SearchService.hybrid_search)
    if (
        ("prefetch_limit = max(" in source or "prefetch_k = max(" in source)
        and "top_k *" in source
        and "min_prefetch_limit" in source
        and "prefetch_multiplier" in source
    ):
        print("✅ prefetch_limit/prefetch_k uses min_prefetch_limit and prefetch_multiplier (e.g. max(self.min_prefetch_limit, top_k * self.prefetch_multiplier))")
    else:
        print("❌ prefetch_limit/prefetch_k calculation not found or does not use min_prefetch_limit/prefetch_multiplier")

    # Check rrf_k
    service = SearchService(QdrantClient(), VectorService())
    if service.rrf_k == 60:
        print(f"✅ rrf_k = {service.rrf_k}")
    else:
        print(f"❌ rrf_k = {service.rrf_k} (expected 60)")

    # Check RRF formula
    rrf_source = inspect.getsource(service.reciprocal_rank_fusion)
    if "1.0 / (self.rrf_k + rank)" in rrf_source:
        print("✅ RRF formula: 1.0 / (rrf_k + rank)")
    else:
        print("❌ RRF formula not found")

    # Check score assignment
    if "score=fused_score" in rrf_source or "score=fused_rrf_score" in rrf_source:
        print("✅ Uses fused RRF score (not original Qdrant score)")
    else:
        print("⚠️  Check score assignment in RRF")
    print()


def check_filters():
    """Check 4: Filters strictness and correctness."""
    print("=" * 60)
    print("Check 4: Filters strictness and correctness")
    print("=" * 60)

    service = SearchService(QdrantClient(), VectorService())

    # Test invalid key
    try:
        service.hybrid_search("test", "test_collection", top_k=5, filters={"invalid": "key"})
        print("❌ Invalid filter key not rejected")
    except ValueError as e:
        print(f"✅ Invalid filter key rejected: {e}")

    # Check allowed keys
    import inspect
    source = inspect.getsource(service.hybrid_search)
    if 'allowed_keys = {"library", "version"}' in source:
        print("✅ Filter keys limited to library and version")
    else:
        print("⚠️  Check filter key validation")
    print()


def check_cli_routing():
    """Check 5: CLI routing."""
    print("=" * 60)
    print("Check 5: CLI routing")
    print("=" * 60)

    import inspect
    with open("app.py") as f:
        app_source = f.read()

    if 'sys.argv[1] in ("ingest", "search")' in app_source:
        print("✅ app.py routes both ingest and search commands")
    else:
        print("⚠️  Check app.py routing")

    # Check CLI has search command
    from src.cli import search_command
    print("✅ search_command function exists")
    print()


def main():
    """Run all sanity checks."""
    print("\n" + "=" * 60)
    print("SANITY CHECKS FOR SEARCH FUNCTIONALITY")
    print("=" * 60 + "\n")

    check_vector_names()
    check_payload_keys()
    check_embed_query_shape()
    check_fusion_api()
    check_fallback_path()
    check_filters()
    check_cli_routing()

    print("=" * 60)
    print("Sanity checks complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

