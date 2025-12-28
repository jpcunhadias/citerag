"""CLI entrypoint for the RAG documentation search system."""

import argparse
import json
import logging
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import EMBEDDING_BATCH_SIZE, QDRANT_HOST, QDRANT_PORT
from src.ingest import VectorService, ingest_documents
from src.search import SearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_command(args: argparse.Namespace) -> int:
    """
    Handle the ingest command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1

    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_path}")
        return 1

    try:
        ingest_documents(
            docs_path=input_path,
            collection_name=args.collection,
            library=args.library,
            version=args.version,
            batch_size=args.batch_size,
        )
        logger.info("Ingestion completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return 1


def search_command(args: argparse.Namespace) -> int:
    """
    Handle the search command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Initialize Qdrant client and VectorService
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        vector_service = VectorService()
        search_service = SearchService(qdrant_client, vector_service)

        # Build filters dict (only include provided keys)
        filters = {}
        if args.library:
            filters["library"] = args.library
        if args.version:
            filters["version"] = args.version

        # Perform search
        results = search_service.hybrid_search(
            query=args.query,
            collection=args.collection,
            top_k=args.limit,
            filters=filters if filters else None,
        )

        # Format output as JSON
        results_json = [result.model_dump() for result in results]
        print(json.dumps(results_json, indent=2))

        logger.info(f"Search completed successfully, returned {len(results)} results")
        return 0
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="RAG Documentation Search System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into Qdrant")
    ingest_parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to directory containing documents (.md/.txt files)",
    )
    ingest_parser.add_argument(
        "--collection",
        required=True,
        type=str,
        help="Name of Qdrant collection",
    )
    ingest_parser.add_argument(
        "--library",
        type=str,
        default=None,
        help="Library name for metadata (optional)",
    )
    ingest_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Library version for metadata (optional)",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help=f"Batch size for embedding generation (default: {EMBEDDING_BATCH_SIZE})",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents in Qdrant")
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query text",
    )
    search_parser.add_argument(
        "--collection",
        required=True,
        type=str,
        help="Name of Qdrant collection",
    )
    search_parser.add_argument(
        "--limit",
        "--top-k",
        dest="limit",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    search_parser.add_argument(
        "--library",
        type=str,
        default=None,
        help="Filter by library name (optional)",
    )
    search_parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Filter by version (optional)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "ingest":
        return ingest_command(args)
    elif args.command == "search":
        return search_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

