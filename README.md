# RAG DS DocSearcher

A self-hosted, offline-first documentation search engine powered by a Retrieval-Augmented Generation (RAG) pipeline. This system indexes local technical documentation and provides a simple web interface for asking questions and finding relevant information.

## High-Level Architecture

```mermaid
graph TD
    %% --- User Layer ---
    subgraph Interface ["User Interface & Entrypoints"]
        User([User])
        Streamlit["Streamlit UI (src/ui.py)"]
        CLI["CLI Command (app.py ask)"]
    end

    %% --- API Layer ---
    subgraph API ["FastAPI Backend"]
        FastAPI["FastAPI Server (api/main.py)"]
        Routes["API Routes (/ask, /search, /collections)"]
    end

    %% --- Orchestration Layer ---
    subgraph Brain ["The Brain (Orchestrator)"]
        RAG["RAGService (src/rag.py)"]
        ContextBuilder["Context Builder (Budgeting & Citations)"]
    end

    %% --- Core Services Layer ---
    subgraph Services ["Core Services"]
        SearchSvc["SearchService (src/search.py)"]
        RerankSvc["RerankerService (src/rerank.py)"]
        LLMSvc["OllamaClient (src/llm.py)"]
        VectorSvc["VectorService (src/ingest.py)"]
    end

    %% --- Infrastructure Layer ---
    subgraph Infra ["Infrastructure & Hardware"]
        Qdrant["Qdrant DB (Docker:6333)"]
        OllamaProc["Ollama Process (HTTP:11434)"]
        FileSystem["Local Docs (.md / .txt)"]

        subgraph Hardware ["RTX 2080 Ti (11GB VRAM)"]
            GPU_Embed["BGE-M3 (Embeddings)"]
            GPU_Rerank["BGE-Reranker-v2-m3 (Cross-Encoder)"]
            GPU_LLM["Llama-3-8B (Inference)"]
        end
    end

    %% --- Connections: Application Flow ---
    User --> Streamlit
    User --> CLI

    Streamlit -->|HTTP| FastAPI
    CLI -->|Direct| RAG

    FastAPI --> Routes
    Routes -->|"1: Ask query"| RAG

    %% --- Connections: RAG Pipeline ---
    RAG -->|"2: Get candidates"| SearchSvc
    SearchSvc -->|"Embed query"| VectorSvc
    VectorSvc -.->|"Inference"| GPU_Embed
    SearchSvc <-->|"Hybrid search"| Qdrant

    RAG -->|"3: Rescore candidates"| RerankSvc
    RerankSvc -.->|"Inference"| GPU_Rerank

    RAG -->|"4: Format & add citations"| ContextBuilder
    ContextBuilder -->|"Prompt with context"| RAG

    RAG -->|"5: Generate answer"| LLMSvc
    LLMSvc <-->|"POST /api/generate"| OllamaProc
    OllamaProc -.->|"Inference"| GPU_LLM

    %% --- Connections: Ingestion (Background) ---
    FileSystem -.->|"Load & split"| VectorSvc
    VectorSvc -.->|"Upsert vectors"| Qdrant

    %% --- Styling ---
    classDef ui fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef api fill:#fff9c4,stroke:#f57f17,stroke-width:2px;
    classDef logic fill:#fff3e0,stroke:#ff6f00,stroke-width:2px;
    classDef service fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef infra fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef hardware fill:#424242,stroke:#000,stroke-width:2px,color:#fff;

    class Streamlit,CLI ui;
    class FastAPI,Routes api;
    class RAG,ContextBuilder logic;
    class SearchSvc,RerankSvc,LLMSvc,VectorSvc service;
    class Qdrant,OllamaProc,FileSystem infra;
    class GPU_Embed,GPU_Rerank,GPU_LLM hardware;
```

## Features

- **Offline-First:** Runs entirely on local hardware. No data leaves your machine.
- **State-of-the-Art RAG:** Implements a sophisticated RAG pipeline using cutting-edge models.
- **Hybrid Search:** Utilizes `BGE-M3` for dense and sparse embeddings, ensuring both semantic and keyword understanding.
- **Cross-Encoder Re-ranking:** Employs `BGE-Reranker-v2-M3` to refine search results for maximum relevance.
- **LLM Synthesis:** Uses `Llama-3-8B` via Ollama to generate accurate, context-aware answers.
- **Dual Interfaces:** Interact via a clean Streamlit web UI or a powerful command-line interface.
- **Citation-Aware:** All generated answers include citations, allowing you to verify the source of the information.

## Tech Stack

- **Vector DB:** Qdrant (running in Docker)
- **Backend API:** FastAPI with Uvicorn
- **Frontend:** Streamlit
- **Embedding Model:** `BAAI/bge-m3` (via `FlagEmbedding`)
- **Reranker Model:** `BAAI/bge-reranker-v2-m3`
- **LLM:** Ollama with `Llama-3-8B-Instruct`
- **Core Libraries:** `langchain`, `sentence-transformers`, `torch`, `qdrant-client`, `fastapi`, `httpx`

## Architecture

The system is composed of two main pipelines:

1.  **Ingestion Pipeline:**
    `CLI -> Loader -> Cleaner -> HeaderSplitter -> RecursiveSplitter -> VectorService -> Qdrant`
    This pipeline loads `.md` and `.txt` files, cleans the text, splits it into deterministic chunks, generates dense and sparse embeddings, and indexes them into the Qdrant vector database.

2.  **Search Pipeline:**
    `Query -> Hybrid Search -> Re-ranking -> LLM Synthesis -> Answer`
    A user query triggers a hybrid search in Qdrant. The results are re-ranked for relevance, and the top results are used as context for the LLM to generate a final, cited answer.

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- [Ollama](https://ollama.com/) installed and running.

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/run-llama/rag-ds-docsearcher.git
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

4.  **Start services with Docker Compose:**

    **Option A: Full stack (recommended for production)**
    ```bash
    docker-compose up -d
    ```
    This starts Qdrant, FastAPI backend, and Streamlit UI. Access:
    - Streamlit UI: http://localhost:8501
    - FastAPI API: http://localhost:8000
    - API Docs: http://localhost:8000/docs

    **Option B: Qdrant only (for local development)**
    ```bash
    docker-compose up -d qdrant
    ```
    Then run FastAPI and Streamlit locally (see Usage section below).

    **Important Notes:**
    - Models are kept on the host filesystem (`~/.cache`) and mounted into containers to avoid re-downloading
    - **Ollama Configuration**: Ollama must be configured to listen on all interfaces (not just localhost) for containers to access it:
      ```bash
      # Set OLLAMA_HOST environment variable before starting Ollama
      export OLLAMA_HOST=0.0.0.0:11434
      ollama serve
      ```
      Or add to your shell profile: `echo 'export OLLAMA_HOST=0.0.0.0:11434' >> ~/.bashrc`
    - The `HOME` environment variable must be set for model cache mounting to work
    - To rebuild containers after code changes: `docker-compose build && docker-compose up -d`

## Usage

This project provides two interfaces: a web app and a CLI.

### 1. Ingesting Documents (CLI)

Before you can search, you must index your documentation.

1.  Place your `.md` or `.txt` files in a directory (e.g., `data/raw/my-docs`).
2.  Run the ingestion command:
    ```bash
    python3 app.py ingest --input <path_to_your_docs> --collection <your_collection_name>
    ```
    - `--input`: Path to the directory containing your documents.
    - `--collection`: A unique name for the Qdrant collection.
    - `--library`, `--version`: (Optional) Metadata tags.

    **Example:**
    ```bash
    # Ingest documentation for the pandas library
    python3 app.py ingest \
      --input data/raw/pandas-docs \
      --collection pandas_v2 \
      --library pandas
    ```

### 2. Asking Questions (Web UI)

The easiest way to interact with your documents is through the Streamlit web application.

**Using Docker Compose (recommended):**
```bash
docker-compose up -d
```
Then open http://localhost:8501 in your browser.

**Using local development:**
1.  **Start the FastAPI backend:**
    ```bash
    uvicorn api.main:app --reload
    ```
    This will start the API server at `http://localhost:8000`. The `--reload` flag enables hot-reload during development.

2.  **Start the Streamlit UI** (in a separate terminal):
    ```bash
    streamlit run ui.py
    ```
3.  **Open your browser** to the displayed URL (usually `http://localhost:8501`).
4.  **Select your collection** from the sidebar and start asking questions.

**Note:** The Streamlit UI requires the FastAPI backend to be running. The CLI (`app.py ask`) works independently and does not require the backend.

### 3. Asking Questions (CLI)

For command-line enthusiasts, the `ask` command provides a direct way to get answers.

```bash
python3 app.py ask "<your_question>" --collection <your_collection_name>
```

**Example:**
```bash
python3 app.py ask "how do i merge two dataframes?" --collection pandas_v2
```

This will output a detailed answer along with the sources used to generate it.

## Testing

Run the automated test script to validate your setup:

```bash
python3 tests/integration/test_docker_setup_integration.py [collection_name]
```

This will test:
- API health endpoint
- Collections endpoint
- Search endpoint (if collection provided)
- Ask/RAG endpoint (if collection provided)
- Streamlit UI accessibility

See `tests/integration/test_docker_setup_integration.py` for more details.
