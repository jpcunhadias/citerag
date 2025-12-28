# rag-ds-docsearcher

A self-hosted, offline-first documentation search engine powered by a Retrieval-Augmented Generation (RAG) pipeline. This system indexes local technical documentation and provides a simple web interface for asking questions and finding relevant information.

## Features

*   **Offline-First:** Runs entirely on local hardware without needing external services.
*   **Hybrid Search:** Utilizes both dense and sparse vectors (`BGE-M3`) for state-of-the-art retrieval that understands both semantic meaning and keyword importance.
*   **Re-ranking:** Employs a re-ranker model (`BGE-Reranker-v2-M3`) to improve the relevance of search results before they are passed to the language model.
*   **Question Answering:** Uses a local LLM (`Llama-3-8B` via Ollama) to generate direct answers based on the retrieved documentation.
*   **Simple UI:** A clean Streamlit web application for querying the document collection.
*   **CLI for Ingestion:** A command-line interface to easily ingest new documentation into the search index.

## Tech Stack

*   **Vector DB:** Qdrant (running in Docker)
*   **Application Framework:** Streamlit
*   **Embedding Model:** `BAAI/bge-m3` (via `FlagEmbedding`)
*   **Reranker Model:** `BAAI/bge-reranker-v2-m3`
*   **LLM:** Ollama with `Llama-3-8B-Instruct`

## Architecture

The system is composed of two main pipelines:

1.  **Ingestion Pipeline:** A multi-step process that prepares and indexes documents.
    `CLI -> Loader -> Cleaner -> HeaderSplitter -> RecursiveSplitter -> VectorService -> Qdrant`
    This pipeline loads `.md` and `.txt` files, cleans the text, splits it into deterministic chunks, generates dense and sparse embeddings, and finally indexes them into the Qdrant vector database.

2.  **Search Pipeline:** The retrieval and generation process that answers user queries.
    `Query -> Hybrid Search -> Re-ranking -> LLM Synthesis -> Answer`
    A user query is used to perform a hybrid search in Qdrant. The results are re-ranked for relevance, and the top results are used as context for the LLM to generate a final answer.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Docker and Docker Compose
*   [Ollama](https://ollama.com/) installed and running.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd rag-ds-docsearcher
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Pull the required LLM model:**
    ```bash
    ollama pull llama3:8b-instruct
    ```

4.  **Start the Qdrant vector database:**
    ```bash
    docker-compose up -d
    ```
    This will start a Qdrant container and persist data in a local Docker volume named `qdrant_storage`.

## Usage

### 1. Ingesting Documents

Place the documentation files you want to index into a directory (e.g., `data/raw/my-docs`). Then, run the ingestion command.

```bash
python3 app.py ingest --input <path_to_your_docs> --collection <your_collection_name>
```

**Arguments:**
*   `--input`: (Required) Path to the directory containing your `.md` or `.txt` documentation files.
*   `--collection`: (Required) A unique name for the Qdrant collection where the documents will be stored.
*   `--library` / `--version`: (Optional) Metadata tags to associate with the ingested documents.

**Example:**
```bash
python3 app.py ingest --input data/raw/pandas-docs --collection pandas_v2 --library pandas
```

### 2. Verifying Ingestion (Optional)

You can use the provided verification script to inspect the contents of your Qdrant collection.

```bash
python3 scripts/verify_qdrant.py <your_collection_name>
```

### 3. Running the Search Application

Once your documents have been ingested, you can start the Streamlit web application to begin searching.

```bash
streamlit run app.py
```

This will open the search interface in your web browser.
