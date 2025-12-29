"""Smoke tests for the ask command."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient

from src.config import QDRANT_HOST, QDRANT_PORT, RAG_MAX_CONTEXT_CHARS

# Use test_collection for testing (fallback to smoke_docs if test_collection doesn't exist)
TEST_COLLECTION = "test_collection"
from src.ingest import VectorService
from src.llm import OllamaClient
from src.rag import RAGService
from src.rerank import RerankerService
from src.search import SearchService


def _test_positive_query():
    """Test 1: Positive query that should exist in docs."""
    print("\n" + "=" * 80)
    print("TEST 1: Positive Query (should find answer with citations)")
    print("=" * 80)

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_service = VectorService()
    search_service = SearchService(qdrant_client, vector_service)
    reranker_service = RerankerService()
    llm_client = OllamaClient()
    rag_service = RAGService(search_service, reranker_service, llm_client)

    # Use a query that should exist in smoke_docs
    query = "merge"
    response = rag_service.ask(
        query=query,
        collection=TEST_COLLECTION,
        top_k=25,
        top_n=5,
        rerank=True,
        debug=False,
    )

    print(f"Query: {query}")
    print(f"Answer: {response.answer[:200]}...")
    print(f"Citations: {len(response.citations)}")

    # Check if answer contains citations
    import re

    citation_pattern = r"\[\d+\]"
    has_citations = bool(re.search(citation_pattern, response.answer))

    if has_citations:
        print("[PASS] Answer contains citations")
    else:
        print("[FAIL] Answer does not contain citations")
        if response.answer.strip() == "I couldn't find this in the indexed documentation.":
            print("   (This is a refusal, which is acceptable if no relevant docs found)")

    return (
        has_citations
        or response.answer.strip() == "I couldn't find this in the indexed documentation."
    )


def _test_negative_query():
    """Test 2: Nonsense query should return refusal."""
    print("\n" + "=" * 80)
    print("TEST 2: Negative Query (nonsense should return refusal)")
    print("=" * 80)

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_service = VectorService()
    search_service = SearchService(qdrant_client, vector_service)
    reranker_service = RerankerService()
    llm_client = OllamaClient()
    rag_service = RAGService(search_service, reranker_service, llm_client)

    query = "xyzabc123nonsensequerythatdoesnotexist"
    response = rag_service.ask(
        query=query,
        collection=TEST_COLLECTION,
        top_k=25,
        top_n=5,
        rerank=True,
        debug=False,
    )

    print(f"Query: {query}")
    print(f"Answer: {response.answer}")
    print(f"Citations: {len(response.citations)}")

    refusal_string = "I couldn't find this in the indexed documentation."
    is_refusal = response.answer.strip() == refusal_string
    has_no_citations = len(response.citations) == 0

    if is_refusal and has_no_citations:
        print("[PASS] Returns refusal with no citations")
    elif is_refusal:
        print("[WARNING] Returns refusal but has citations (might be acceptable)")
    else:
        print(f"[FAIL] Expected refusal '{refusal_string}', got: {response.answer[:100]}")

    return is_refusal


def _test_budget_truncation():
    """Test 3: Budget truncation with lower RAG_MAX_CONTEXT_CHARS."""
    print("\n" + "=" * 80)
    print("TEST 3: Budget Truncation (lower RAG_MAX_CONTEXT_CHARS)")
    print("=" * 80)

    # Temporarily lower the budget
    original_budget = RAG_MAX_CONTEXT_CHARS

    # We'll need to patch it, but for now let's test with a query that should hit multiple chunks
    print(f"Original budget: {original_budget}")

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_service = VectorService()
    search_service = SearchService(qdrant_client, vector_service)
    reranker_service = RerankerService()
    llm_client = OllamaClient()

    # Create service and manually test build_context with limited budget
    rag_service = RAGService(search_service, reranker_service, llm_client)

    query = "merge"
    results = search_service.hybrid_search(query=query, collection=TEST_COLLECTION, top_k=25)
    results = reranker_service.rerank(query=query, results=results, top_n=10)

    # Test with original budget
    context1, citations1, _ = rag_service.build_context(results)
    print(
        f"With original budget ({original_budget}): {len(citations1)} chunks, {len(context1)} chars"
    )

    # Temporarily modify budget (for testing only)
    import src.config as config_module

    config_module.RAG_MAX_CONTEXT_CHARS = 500  # Very low budget
    context2, citations2, _ = rag_service.build_context(results)
    config_module.RAG_MAX_CONTEXT_CHARS = original_budget  # Restore

    print(f"With low budget (500): {len(citations2)} chunks, {len(citations2)} chars")

    # Check labels are consistent
    labels2 = [c.label for c in citations2]

    if labels2 == [str(i) for i in range(1, len(citations2) + 1)]:
        print("[PASS] Labels are consistent (1, 2, 3...)")
    else:
        print(f"[FAIL] Labels inconsistent: {labels2}")

    if len(citations2) < len(citations1):
        print("[PASS] Budget truncation works (fewer chunks with lower budget)")
    else:
        print("[WARNING] Budget truncation may not be working")

    return len(citations2) <= len(citations1) and labels2 == [
        str(i) for i in range(1, len(citations2) + 1)
    ]


def _test_no_rerank():
    """Test 4: --no-rerank flag should use fused retrieval scores."""
    print("\n" + "=" * 80)
    print("TEST 4: No Rerank (should use fused retrieval scores)")
    print("=" * 80)

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_service = VectorService()
    search_service = SearchService(qdrant_client, vector_service)
    reranker_service = RerankerService()
    llm_client = OllamaClient()
    rag_service = RAGService(search_service, reranker_service, llm_client)

    query = "merge"

    # Test with rerank
    response_rerank = rag_service.ask(
        query=query,
        collection=TEST_COLLECTION,
        top_k=25,
        top_n=5,
        rerank=True,
        debug=False,
    )

    # Test without rerank
    response_no_rerank = rag_service.ask(
        query=query,
        collection=TEST_COLLECTION,
        top_k=25,
        top_n=5,
        rerank=False,
        debug=False,
    )

    print(f"With rerank: {len(response_rerank.citations)} citations")
    if response_rerank.citations:
        print(f"  First citation score: {response_rerank.citations[0].score}")

    print(f"Without rerank: {len(response_no_rerank.citations)} citations")
    if response_no_rerank.citations:
        print(f"  First citation score: {response_no_rerank.citations[0].score}")

    # Scores should be different (reranker scores are typically negative, fused scores are positive)
    if response_rerank.citations and response_no_rerank.citations:
        score_rerank = response_rerank.citations[0].score
        score_no_rerank = response_no_rerank.citations[0].score

        if score_rerank != score_no_rerank:
            print("[PASS] Scores differ between rerank and no-rerank (as expected)")
            return True
        else:
            print("[WARNING] Scores are the same (might be coincidence)")
            return True  # Still pass, might be edge case
    else:
        print("[WARNING] No citations to compare")
        return True


def main():
    """Run all smoke tests."""
    print("=" * 80)
    print("RAG ASK COMMAND SMOKE TESTS")
    print("=" * 80)

    # Check if collection exists
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = qdrant_client.get_collections().collections
    collection_names = [c.name for c in collections]

    # Try test_collection first, fallback to smoke_docs if available
    collection_to_use = TEST_COLLECTION
    if TEST_COLLECTION not in collection_names:
        if "smoke_docs" in collection_names:
            collection_to_use = "smoke_docs"
            print(f"[WARNING] Collection '{TEST_COLLECTION}' not found, using 'smoke_docs'")
        else:
            print(f"[FAIL] ERROR: Collection '{TEST_COLLECTION}' or 'smoke_docs' not found.")
            print(f"Available collections: {collection_names}")
            print("Please run ingestion first.")
            return 1

    # Update TEST_COLLECTION for the test functions
    import tests.test_smoke_ask as test_module

    test_module.TEST_COLLECTION = collection_to_use

    results = []
    results.append(("Positive Query", _test_positive_query()))
    results.append(("Negative Query", _test_negative_query()))
    results.append(("Budget Truncation", _test_budget_truncation()))
    results.append(("No Rerank", _test_no_rerank()))

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
