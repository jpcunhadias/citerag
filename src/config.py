"""Configuration settings for the RAG documentation search system."""

from pathlib import Path
from typing import Literal

# Model Configuration
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"

# Qdrant Configuration
QDRANT_HOST: str = "localhost"
QDRANT_PORT: int = 6333
QDRANT_COLLECTION_NAME: str = "docs_collection"

# Ollama Configuration
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL_NAME: str = "llama3:8b-instruct"

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

