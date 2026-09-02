#!/usr/bin/env python3
"""Run the golden-set eval harness against a live RAG stack.

Requires a running Qdrant + Ollama, with the golden set's collections
already ingested. See eval/README.md for setup.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient  # noqa: E402

from src.config import QDRANT_HOST, QDRANT_PORT  # noqa: E402
from src.eval import build_report, load_golden_set, score_case  # noqa: E402
from src.ingest import VectorService  # noqa: E402
from src.llm import OllamaClient  # noqa: E402
from src.rag import RAGService  # noqa: E402
from src.rerank import RerankerService  # noqa: E402
from src.search import SearchService  # noqa: E402

DEFAULT_GOLDEN_SET = Path(__file__).parent.parent / "eval" / "golden_set.yaml"
DEFAULT_REPORT_OUT = Path(__file__).parent.parent / "eval" / "results" / "latest.json"


def main() -> int:
    """Run the golden set against a live RAG pipeline and report results."""
    parser = argparse.ArgumentParser(description="Run the RAG eval harness")
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument(
        "--fail-under",
        type=float,
        default=0.8,
        help="Exit non-zero if pass rate falls below this fraction (default: 0.8)",
    )
    parser.add_argument(
        "--hyde",
        action="store_true",
        help="Retrieve using a generated hypothetical answer (HyDE) instead of the raw query",
    )
    args = parser.parse_args()

    cases = load_golden_set(args.golden_set)
    mode = "HyDE" if args.hyde else "baseline"
    print(f"Loaded {len(cases)} golden-set cases from {args.golden_set} ({mode} retrieval)\n")

    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    vector_service = VectorService()
    search_service = SearchService(qdrant_client, vector_service)
    reranker_service = RerankerService()
    llm_client = OllamaClient()
    rag_service = RAGService(
        search_service=search_service,
        reranker_service=reranker_service,
        llm_client=llm_client,
    )

    results = []
    for case in cases:
        print(f"[{case.id}] {case.query!r} (collection={case.collection})...", end=" ", flush=True)
        response = rag_service.ask(query=case.query, collection=case.collection, use_hyde=args.hyde)
        result = score_case(case, response)
        results.append(result)
        print("PASS" if result.passed else f"FAIL: {'; '.join(result.reasons)}")

    report = build_report(results)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report.model_dump_json(indent=2))

    print(f"\n{report.passed}/{report.total} passed ({report.pass_rate:.0%})")
    print(f"Report written to {args.report_out}")

    return 0 if report.pass_rate >= args.fail_under else 1


if __name__ == "__main__":
    sys.exit(main())
