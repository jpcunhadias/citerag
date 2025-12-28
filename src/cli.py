"""CLI entrypoint for the RAG documentation search system."""

import argparse
import logging
import sys
from pathlib import Path

from src.config import EMBEDDING_BATCH_SIZE
from src.ingest import ingest_documents

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

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "ingest":
        return ingest_command(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

