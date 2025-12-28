# scripts/verify_services.py
from src.llm import OllamaClient
from src.rerank import RerankerService
from src.models import SearchResult
import sys

def test_services():
    print("--- 1. Testing LLM Connection ---")
    try:
        client = OllamaClient()
        # Should be fast if model is loaded
        resp = client.generate("Reply with one word: Online")
        print(f"✅ LLM Response: {resp}")
    except Exception as e:
        print(f"❌ LLM Failed: {e}")
        sys.exit(1)

    print("\n--- 2. Testing Reranker (GPU Load) ---")
    try:
        reranker = RerankerService()
        docs = [
            SearchResult(chunk_id="1", score=0.1, text="Apple", source_path="a", canonical_source_id="a"),
            SearchResult(chunk_id="2", score=0.1, text="Car", source_path="b", canonical_source_id="b")
        ]
        # Query matches "Apple", so doc 1 should jump to top with higher score
        ranked = reranker.rerank("fruit", docs, top_n=1)

        top_doc = ranked[0]
        print(f"✅ Reranker Top Result: {top_doc.text}")
        print(f"   New Score: {top_doc.score:.4f}")

        if top_doc.text != "Apple":
            print("⚠️ Logic Warning: Expected 'Apple' to win.")
    except Exception as e:
        print(f"❌ Reranker Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_services()

