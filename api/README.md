# FastAPI Backend API

REST API backend for the RAG documentation search system.

## Endpoints

### Health Check

**GET** `/health`

Check the health status of the API and its dependencies (Qdrant, Ollama).

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "qdrant": "healthy",
    "ollama": "healthy"
  }
}
```

### Collections

**GET** `/collections`

Get list of available Qdrant collections.

**Response:**
```json
{
  "collections": ["pandas_docs", "numpy_docs"]
}
```

### Ask (RAG Pipeline)

**POST** `/api/ask`

Execute the full RAG pipeline: search, rerank, build context, and generate answer.

**Request Body:**
```json
{
  "query": "How do I merge two DataFrames?",
  "collection": "pandas_docs",
  "top_k": 25,
  "top_n": 5,
  "rerank": true,
  "debug": false
}
```

**Response:**
```json
{
  "answer": "You can merge two DataFrames using the `merge()` method...",
  "citations": [
    {
      "label": "1",
      "chunk_id": "...",
      "canonical_source_id": "...",
      "source_path": "...",
      "header": "DataFrame.merge",
      "title": null,
      "score": 0.95
    }
  ],
  "context_used": null,
  "used_chunk_ids": ["..."]
}
```

**Note:** `context_used` is only included if `debug: true` in the request.

### Ask Stream (RAG Pipeline with Streaming)

**POST** `/api/ask/stream`

Execute the RAG pipeline and stream LLM tokens as NDJSON. Same request body as `/api/ask` (except `debug` is ignored). Each line is JSON: `{"type": "token", "content": "..."}` or `{"type": "done", "citations": [...], "used_chunk_ids": [...]}`.

### Search

**POST** `/api/search`

Perform hybrid search on documents (without RAG generation).

**Request Body:**
```json
{
  "query": "DataFrame merge",
  "collection": "pandas_docs",
  "top_k": 5,
  "filters": {
    "library": "pandas",
    "version": "2.0"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "...",
      "score": 0.92,
      "text": "...",
      "source_path": "...",
      "canonical_source_id": "...",
      "header": "DataFrame.merge",
      "library": "pandas",
      "version": "2.0",
      "title": null,
      "metadata": {}
    }
  ]
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Configuration

### CORS (Cross-Origin Resource Sharing)

The API uses CORS middleware to control which origins can make requests to it. This is configured via the `CORS_ORIGINS` environment variable.

**Default (Development):**
```bash
CORS_ORIGINS=http://localhost:8501
```

**Production:**
Set to specific origins (comma-separated list):
```bash
CORS_ORIGINS=https://app.example.com,https://admin.example.com
```

**Important:** Never use wildcard origins (`*`) in production as this allows any website to make requests to your API, which is a security risk.

To configure CORS origins when running with Docker Compose, edit the `docker-compose.yml` file and update the `CORS_ORIGINS` environment variable under the `api` service.

## Service Initialization

Services (SearchService, RerankerService, OllamaClient) are initialized lazily on first request and cached as singletons for subsequent requests. This ensures efficient resource usage while maintaining fast response times.

## Error Handling

- **400 Bad Request**: Invalid request parameters
- **500 Internal Server Error**: RAG pipeline or search failure
- **503 Service Unavailable**: Backend services (Qdrant, Ollama) unavailable

Error responses include a `detail` field with a human-readable error message.

