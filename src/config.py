"""Configuration settings for the RAG documentation search system."""

import os
from pathlib import Path
from typing import Literal

# Model Configuration
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

# Qdrant Configuration (can be overridden by environment variables)
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION_NAME: str = "docs_collection"

# Ollama Configuration (can be overridden by environment variables)
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "llama3")

# API Configuration (can be overridden by environment variables)
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")

# Paths
BASE_DOCS_DIR: Path = Path("data/raw")
PROCESSED_DOCS_DIR: Path = Path("data/processed")

# Device Configuration
DeviceType = Literal["cuda", "cpu"]
DEFAULT_DEVICE: DeviceType = "cuda"

# Chunking Configuration
CHUNK_SIZE: int = 1024  # characters
CHUNK_OVERLAP: int = 100  # characters
CHUNKER_FINGERPRINT: str = "md:v1|headers=#,##|size=1024|overlap=100"
EMBEDDING_BATCH_SIZE: int = 32

# Search Configuration
HYBRID_SEARCH_TOP_K: int = 25
RERANK_TOP_K: int = 5

# RAG Configuration
RAG_MAX_CONTEXT_CHARS: int = 12000
RAG_REFUSAL_MESSAGE: str = "I couldn't find this in the indexed documentation."

