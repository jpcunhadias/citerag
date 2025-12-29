#!/usr/bin/env python3
"""Test script to validate Docker Compose setup."""

import sys
from typing import Optional

import requests


def test_api_health(base_url: str = "http://localhost:8000") -> bool:
    """Test API health endpoint."""
    print("Testing API health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        # Health endpoint can return 200 (healthy) or 503 (unhealthy)
        # Both are valid responses, we just check the JSON content
        data = response.json()
        print(f"  [PASS] API is responding")
        print(f"  Status: {data['status']}")
        print(f"  Services: {data['services']}")
        return data["status"] == "healthy"
    except Exception as e:
        print(f"  [FAIL] API health check failed: {e}")
        return False


def test_api_collections(base_url: str = "http://localhost:8000") -> bool:
    """Test API collections endpoint."""
    print("\nTesting API collections endpoint...")
    try:
        response = requests.get(f"{base_url}/collections", timeout=10)
        response.raise_for_status()
        data = response.json()
        collections = data.get("collections", [])
        print(f"  [PASS] Collections endpoint working")
        print(f"  Found {len(collections)} collections: {collections}")
        return True
    except Exception as e:
        print(f"  [FAIL] Collections endpoint failed: {e}")
        return False


def test_api_search(base_url: str = "http://localhost:8000", collection: Optional[str] = None) -> bool:
    """Test API search endpoint."""
    print("\nTesting API search endpoint...")
    if not collection:
        print("  [WARNING] Skipping search test (no collection specified)")
        return True

    try:
        payload = {
            "query": "test query",
            "collection": collection,
            "top_k": 3,
        }
        response = requests.post(f"{base_url}/api/search", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        print(f"  [PASS] Search endpoint working")
        print(f"  Found {len(results)} results")
        return True
    except Exception as e:
        print(f"  [FAIL] Search endpoint failed: {e}")
        return False


def test_api_ask(base_url: str = "http://localhost:8000", collection: Optional[str] = None) -> bool:
    """Test API ask endpoint."""
    print("\nTesting API ask endpoint...")
    if not collection:
        print("  [WARNING] Skipping ask test (no collection specified)")
        return True

    try:
        payload = {
            "query": "What is this documentation about?",
            "collection": collection,
            "top_k": 5,
            "top_n": 3,
            "rerank": True,
            "debug": False,
        }
        print(f"  Sending request (this may take a while)...")
        response = requests.post(f"{base_url}/api/ask", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        print(f"  [PASS] Ask endpoint working")
        print(f"  Answer length: {len(answer)} characters")
        print(f"  Citations: {len(citations)}")
        if answer:
            print(f"  Answer preview: {answer[:100]}...")
        return True
    except Exception as e:
        print(f"  [FAIL] Ask endpoint failed: {e}")
        return False


def test_streamlit(base_url: str = "http://localhost:8501") -> bool:
    """Test Streamlit UI accessibility."""
    print("\nTesting Streamlit UI...")
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        print(f"  [PASS] Streamlit UI is accessible")
        return True
    except Exception as e:
        print(f"  [FAIL] Streamlit UI not accessible: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("=" * 60)
    print("Docker Compose Setup Test")
    print("=" * 60)

    # Get collection name from command line or use default
    collection: Optional[str] = None
    if len(sys.argv) > 1:
        collection = sys.argv[1]
    else:
        # Try to get first collection from API
        try:
            response = requests.get("http://localhost:8000/collections", timeout=10)
            if response.status_code == 200:
                collections = response.json().get("collections", [])
                if collections:
                    collection = collections[0]
                    print(f"Using collection: {collection}\n")
        except Exception:
            # If the collections endpoint is unavailable or returns invalid data,
            # proceed without a default collection so collection-based tests are skipped.
            pass

    results = []

    # Test API health
    results.append(("API Health", test_api_health()))

    # Test collections
    results.append(("API Collections", test_api_collections()))

    # Test search (if collection available)
    if collection:
        results.append(("API Search", test_api_search(collection=collection)))

    # Test ask (if collection available)
    if collection:
        results.append(("API Ask", test_api_ask(collection=collection)))

    # Test Streamlit
    results.append(("Streamlit UI", test_streamlit()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name:20s}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

