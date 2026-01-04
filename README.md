# Retrieval-Augmented Generation (RAG) Project

## Overview
- End-to-end RAG pipeline for ingesting documents, building a vector index, and serving grounded responses.
- Components: data ingestion, embeddings store, retriever, generator, and service layer.
- Notebook folder intentionally ignored; core workflow runs via CLI/API.

## Project Structure
- `src/` – application code (ingestion, indexing, retrieval, generation, APIs).
- `configs/` – configuration files (model/backends, paths, parameters).
- `data/` – input documents (raw) and processed artifacts.
- `models/` – local model weights or adapters (if used).
- `scripts/` – helper/automation scripts (ingest, index, maintenance).
- `tests/` – automated tests.
- `docker/` – container assets (Dockerfile, compose, entrypoints).
- `logs/` – runtime logs.
- `notebook/` – development notebooks (ignored for deployment).

## Prerequisites
- Python 3.9+ (or project-defined runtime)
- Recommended: virtualenv/conda
- Optional: Docker & Docker Compose
- Access to embedding and LLM backends (API keys or local models)
- GPU optional; CPU supported (may be slower)

## Setup
1) Clone and enter repo:
   ```bash
   git clone <repo-url> e:\RAG
   cd e:\RAG
   ```
2) Create environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
3) Configure environment variables (example):
   ```bash
   copy .env.example .env
   ```
   - `EMBEDDING_MODEL`, `LLM_MODEL` or provider keys
   - `VECTOR_STORE_URL` or local path
   - `DOCS_PATH`, `INDEX_PATH`, `LOG_LEVEL`

## Data Ingestion & Indexing
- Place raw docs in `data/raw/`.
- Run ingestion (example):
  ```bash
  python -m scripts.ingest --input data/raw --output data/processed
  ```
- Build/update index:
  ```bash
  python -m scripts.index --input data/processed --index indexes/default
  ```

## Running the Service
- Start API (example FastAPI/Flask):
  ```bash
  uvicorn src.api.app:app --host 0.0.0.0 --port 8000
  ```
- Query endpoint (example):
  ```bash
  curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"question\": \"...\"}"
  ```

## CLI Usage
- Example query via CLI:
  ```bash
  python -m src.cli.query --question "What is RAG?" --top-k 5
  ```

## Configuration
- Default configs in `configs/`.
- Override via `--config` flag or environment variables.
- Logging via `LOG_LEVEL` and `logs/` output.

## Testing
- Run tests:
  ```bash
  pytest
  ```
- Lint/format (if configured):
  ```bash
  ruff . && black .
  ```

## Deployment
- Docker (example):
  ```bash
  docker compose up --build
  ```
- Production notes: set persistent storage for indexes, secure API keys, enable auth/rate limiting.

## Troubleshooting
- Missing embeddings/LLM keys: ensure `.env` is loaded.
- Slow responses: lower context window or top-k; enable GPU if available.
- Index not found: rebuild via `scripts.index`.

## Contributing
- Fork/branch, open PR with tests and lint passing.
- Follow code style defined in linters/formatters.

## License
- Specify project license in `LICENSE` (add if missing).
