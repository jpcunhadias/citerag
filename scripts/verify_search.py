#!/usr/bin/env python3
"""Verify search functionality."""

import json
import subprocess
import sys
from typing import Any


def run_search_command(query: str, collection: str, limit: int) -> tuple[int, str]:
    """Run search command and return exit code and output."""
    cmd = [
        "python",
        "app.py",
        "search",
        query,
        "--collection",
        collection,
        "--limit",
        str(limit),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def verify_json_format(output: str) -> tuple[bool, list[dict[str, Any]]]:
    """Verify output is valid JSON and return parsed results."""
    try:
        # Extract JSON from output (may have log lines before/after)
        lines = output.strip().split("\n")
        json_start = None
        json_end = None

        for i, line in enumerate(lines):
            if line.strip().startswith("["):
                json_start = i
            if json_start is not None and line.strip().endswith("]"):
                json_end = i + 1
                break

        if json_start is None or json_end is None:
            return False, []

        json_str = "\n".join(lines[json_start:json_end])
        results = json.loads(json_str)
        return True, results
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[FAIL] JSON parsing failed: {e}")
        return False, []


def verify_ordering(results: list[dict[str, Any]]) -> bool:
    """Verify results are sorted by score descending."""
    if len(results) < 2:
        return True  # Single result is trivially sorted

    scores = [r["score"] for r in results]
    is_descending = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    if not is_descending:
        print(f"[FAIL] Scores not in descending order: {scores}")
    return is_descending


def verify_relevance(results: list[dict[str, Any]], query: str) -> bool:
    """Verify top result is relevant to query."""
    if not results:
        print("[FAIL] No results returned")
        return False

    top_text = results[0]["text"].lower()

    # Check for merge/join/concat keywords
    merge_keywords = ["merge", "join", "concat", "combine", "dataframe"]
    has_relevant_keyword = any(keyword in top_text for keyword in merge_keywords)

    if not has_relevant_keyword:
        print(f"[FAIL] Top result doesn't mention merge/join/concat: {top_text[:100]}...")
        return False

    return True


def verify_scores(results: list[dict[str, Any]]) -> bool:
    """Verify scores are present and positive."""
    if not results:
        return False

    for i, result in enumerate(results):
        if "score" not in result:
            print(f"[FAIL] Result {i} missing score field")
            return False

        score = result["score"]
        if not isinstance(score, (int, float)):
            print(f"[FAIL] Result {i} score is not numeric: {type(score)}")
            return False

        if score <= 0:
            print(f"[FAIL] Result {i} has non-positive score: {score}")
            return False

    print(f"[PASS] All scores are positive (range: {min(r['score'] for r in results):.6f} to {max(r['score'] for r in results):.6f})")
    return True


def main():
    """Run search verification."""
    query = "how to merge dataframes"
    collection = "pandas_docs"
    limit = 3

    print("=" * 60)
    print("Search Verification")
    print("=" * 60)
    print(f"Query: '{query}'")
    print(f"Collection: {collection}")
    print(f"Limit: {limit}")
    print()

    # Criterion 1: Execution
    print("[1] Execution: Command runs without stack traces...")
    exit_code, output = run_search_command(query, collection, limit)
    if exit_code != 0:
        print(f"[FAIL] Command failed with exit code {exit_code}")
        print("Output:")
        print(output)
        return 1
    print("[PASS] Command executed successfully")
    print()

    # Criterion 2: Format
    print("[2] Format: Output is valid JSON...")
    is_valid_json, results = verify_json_format(output)
    if not is_valid_json:
        print("[FAIL] Output is not valid JSON")
        print("Output:")
        print(output[-500:])  # Last 500 chars
        return 1
    print(f"[PASS] Output is valid JSON ({len(results)} results)")
    print()

    # Criterion 3: Ordering
    print("[3] Ordering: Results sorted by score descending...")
    if not verify_ordering(results):
        return 1
    scores = [r["score"] for r in results]
    print(f"[PASS] Results sorted correctly: {scores}")
    print()

    # Criterion 4: Relevance
    print("[4] Relevance: Top result mentions merge/join/concat...")
    if not verify_relevance(results, query):
        return 1
    print(f"[PASS] Top result is relevant: '{results[0]['text'][:80]}...'")
    print()

    # Criterion 5: Scores
    print("[5] Scores: Scores present and positive...")
    if not verify_scores(results):
        return 1
    print()

    print("=" * 60)
    print("[SUCCESS] ALL CRITERIA PASSED!")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  - Executed successfully")
    print(f"  - Valid JSON output")
    print(f"  - Results sorted by score (descending)")
    print(f"  - Top result is relevant to query")
    print(f"  - All scores are positive")

    return 0


if __name__ == "__main__":
    sys.exit(main())

