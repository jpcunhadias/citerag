#!/usr/bin/env python3
"""Integration test for FastAPI backend and Streamlit client."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from src.api_client import APIClient, APIClientError
from src.config import API_BASE_URL


def wait_for_api(max_retries=10, delay=2):
    """Wait for API to be available."""
    print(f"Waiting for API at {API_BASE_URL}...")
    for i in range(max_retries):
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{API_BASE_URL}/health")
                if response.status_code == 200:
                    print("[PASS] API is available")
                    return True
        except Exception:
            if i < max_retries - 1:
                print(f"  Retry {i+1}/{max_retries}...")
                time.sleep(delay)
            else:
                print(f"[FAIL] API not available after {max_retries} retries")
                print(f"\n[WARNING] Start FastAPI server:")
                print(f"   uvicorn api.main:app --reload")
                return False
    return False


def _test_health_endpoint():
    """Test /health endpoint."""
    print("\n" + "=" * 80)
    print("TEST 1: Health Check Endpoint")
    print("=" * 80)

    url = f"{API_BASE_URL}/health"
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            print(f"Status: {data['status']}")
            print(f"Services: {data['services']}")

            if data['status'] == 'healthy':
                print("[PASS] All services healthy")
                return True
            else:
                print(f"[WARNING] Services unhealthy: {data['services']}")
                return False
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def _test_collections_endpoint():
    """Test /collections endpoint."""
    print("\n" + "=" * 80)
    print("TEST 2: Collections Endpoint")
    print("=" * 80)

    url = f"{API_BASE_URL}/collections"
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            collections = data.get('collections', [])
            print(f"Found {len(collections)} collections: {collections}")

            if collections:
                print("[PASS] Collections endpoint working")
                return True, collections[0]
            else:
                print("[WARNING] No collections found (ingest documents first)")
                return True, None  # Still pass, just no collections
    except Exception as e:
        print(f"[FAIL] {e}")
        return False, None


def _test_search_endpoint(collection_name: str):
    """Test /api/search endpoint."""
    print("\n" + "=" * 80)
    print("TEST 3: Search Endpoint")
    print("=" * 80)

    url = f"{API_BASE_URL}/api/search"
    payload = {
        "query": "merge",
        "collection": collection_name,
        "top_k": 5
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            results = data.get('results', [])
            print(f"Found {len(results)} results")

            if results:
                print(f"First result: {results[0].get('text', '')[:100]}...")
                print(f"Score: {results[0].get('score', 0):.4f}")
                print("[PASS] Search endpoint working")
                return True
            else:
                print("[WARNING] No results returned (collection may be empty)")
                return True  # Still pass, might be empty collection
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def _test_ask_endpoint(collection_name: str):
    """Test /api/ask endpoint."""
    print("\n" + "=" * 80)
    print("TEST 4: Ask Endpoint (RAG Pipeline)")
    print("=" * 80)

    url = f"{API_BASE_URL}/api/ask"
    payload = {
        "query": "What is merge?",
        "collection": collection_name,
        "top_k": 10,
        "top_n": 3,
        "rerank": True,
        "debug": False
    }

    try:
        print("Sending request (this may take 30-60 seconds)...")
        with httpx.Client(timeout=300.0) as client:  # RAG can take time
            response = client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()

            answer = data.get('answer', '')
            citations = data.get('citations', [])

            print(f"Answer: {answer[:200]}...")
            print(f"Citations: {len(citations)}")

            if answer and not answer.startswith("I couldn't find"):
                print("[PASS] Ask endpoint working")
                return True
            else:
                print("[WARNING] Got refusal message (might be expected if no relevant docs)")
                return True  # Still pass, refusal is valid
    except Exception as e:
        print(f"[FAIL] {e}")
        return False


def _test_api_client():
    """Test Streamlit's APIClient."""
    print("\n" + "=" * 80)
    print("TEST 5: Streamlit APIClient")
    print("=" * 80)

    try:
        client = APIClient(base_url=API_BASE_URL)
        collections = client.get_collections()
        print(f"[PASS] APIClient working, found {len(collections)} collections")
        return True, collections[0] if collections else None
    except APIClientError as e:
        print(f"[FAIL] {e}")
        return False, None


def _test_api_client_search(client: APIClient, collection_name: str):
    """Test APIClient search method."""
    print("\n" + "=" * 80)
    print("TEST 6: APIClient Search Method")
    print("=" * 80)

    try:
        results = client.search(
            query="merge",
            collection=collection_name,
            top_k=5
        )
        print(f"[PASS] APIClient search working, found {len(results)} results")
        if results:
            print(f"   First result score: {results[0].score:.4f}")
        return True
    except APIClientError as e:
        print(f"[FAIL] {e}")
        return False


def _test_api_client_ask(client: APIClient, collection_name: str):
    """Test APIClient ask method."""
    print("\n" + "=" * 80)
    print("TEST 7: APIClient Ask Method")
    print("=" * 80)

    try:
        print("Sending request (this may take 30-60 seconds)...")
        response = client.ask(
            query="What is merge?",
            collection=collection_name,
            top_k=10,
            top_n=3,
            rerank=True,
            debug=False
        )
        print(f"[PASS] APIClient ask working")
        print(f"   Answer length: {len(response.answer)} chars")
        print(f"   Citations: {len(response.citations)}")
        if response.citations:
            print(f"   First citation: {response.citations[0].label} - {response.citations[0].header or 'N/A'}")
        return True
    except APIClientError as e:
        print(f"[FAIL] {e}")
        return False


def main():
    """Run all integration tests."""
    print("=" * 80)
    print("FASTAPI + STREAMLIT CLIENT INTEGRATION TESTS")
    print("=" * 80)
    print(f"Testing API at: {API_BASE_URL}")

    # Wait for API
    if not wait_for_api():
        return 1

    results = []

    # Test 1: Health
    results.append(("Health Check", _test_health_endpoint()))

    # Test 2: Collections
    success, collection_name = _test_collections_endpoint()
    results.append(("Collections", success))

    # Test 3-4: Direct API calls (require collection)
    if collection_name:
        results.append(("API Search", _test_search_endpoint(collection_name)))
        results.append(("API Ask", _test_ask_endpoint(collection_name)))
    else:
        print("\n[WARNING] Skipping API Search/Ask tests (no collections available)")
        results.append(("API Search", False))
        results.append(("API Ask", False))

    # Test 5-7: Streamlit APIClient
    success, client_collection = _test_api_client()
    results.append(("APIClient Init", success))

    if success and client_collection:
        client = APIClient(base_url=API_BASE_URL)
        results.append(("APIClient Search", _test_api_client_search(client, client_collection)))
        results.append(("APIClient Ask", _test_api_client_ask(client, client_collection)))
    else:
        print("\n[WARNING] Skipping APIClient Search/Ask tests (no collections)")
        results.append(("APIClient Search", False))
        results.append(("APIClient Ask", False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARNING] Some tests failed or were skipped")
        print("   (This is OK if you haven't ingested documents yet)")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

