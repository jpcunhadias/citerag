"""End-to-end tests for the RAG API."""

import re
import sys
from pathlib import Path

# Add project root for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api_client import APIClient, APIClientError  # noqa: E402

# Use test_collection for testing (fallback to smoke_docs if test_collection doesn't exist)
TEST_COLLECTION = "test_collection"


def _test_positive_query(api_client: APIClient):
    """Test 1: Positive query that should exist in docs."""
    print("\n" + "=" * 80)
    print("TEST 1: Positive Query (should find answer with citations)")
    print("=" * 80)

    query = "merge"
    try:
        response = api_client.ask(
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

        citation_pattern = r"[\d+]"
        has_citations = bool(re.search(citation_pattern, response.answer))

        if has_citations:
            print("[PASS] Answer contains citations")
        else:
            print("[FAIL] Answer does not contain citations")
            if "couldn't find" in response.answer:
                print("   (This is a refusal, which is acceptable if no relevant docs found)")

        return has_citations or "couldn't find" in response.answer

    except APIClientError as e:
        print(f"[FAIL] API Client error: {e}")
        return False


def _test_negative_query(api_client: APIClient):
    """Test 2: Nonsense query should return refusal."""
    print("\n" + "=" * 80)
    print("TEST 2: Negative Query (nonsense should return refusal)")
    print("=" * 80)

    query = "xyzabc123nonsensequerythatdoesnotexist"
    try:
        response = api_client.ask(
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
        is_refusal = refusal_string in response.answer
        has_no_citations = len(response.citations) == 0

        if is_refusal and has_no_citations:
            print("[PASS] Returns refusal with no citations")
        else:
            print(f"[FAIL] Expected refusal, got: {response.answer[:100]}")

        return is_refusal and has_no_citations

    except APIClientError as e:
        print(f"[FAIL] API Client error: {e}")
        return False


def main():
    """Run all e2e tests."""
    print("=" * 80)
    print("RAG API END-TO-END TESTS")
    print("=" * 80)

    api_client = APIClient()

    try:
        collections = api_client.get_collections()
        # Try test_collection first, fallback to smoke_docs if available
        collection_to_use = TEST_COLLECTION
        if TEST_COLLECTION not in collections:
            if "smoke_docs" in collections:
                collection_to_use = "smoke_docs"
                print(f"[WARNING] Collection '{TEST_COLLECTION}' not found, using 'smoke_docs'")
            else:
                print(f"[FAIL] ERROR: Collection '{TEST_COLLECTION}' or 'smoke_docs' not found.")
                print(f"Available collections: {collections}")
                print("Please run ingestion first.")
                return 1

        # Update TEST_COLLECTION for the test functions
        import tests.integration.test_e2e_api as test_module

        test_module.TEST_COLLECTION = collection_to_use
    except APIClientError as e:
        print(f"[FAIL] ERROR: Could not connect to API to get collections: {e}")
        return 1

    results = []
    results.append(("Positive Query", _test_positive_query(api_client)))
    results.append(("Negative Query", _test_negative_query(api_client)))

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
