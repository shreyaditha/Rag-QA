# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) document question-and-answering system. Ingests PDF, plain text, and Markdown files, indexes them into vector storage, and generates cited responses grounded exclusively in the retrieved content.

---

## Architecture Overview

The system runs entirely locally across containerized application services and a local Ollama instance. The frontend interacts with the FastAPI backend over REST for document management and Server-Sent Events (SSE) for streaming query responses. The backend handles structural document parsing, chunking with sentence-boundary preservation, vector embeddings via Ollama, cosine similarity search in PostgreSQL via `pgvector`, and answer synthesis using a local LLM.

```
+-----------------------------------------------------------------------------------+
| Host Machine                                                                      |
|                                                                                   |
|  +------------------+         REST / SSE          +----------------------------+  |
|  | Streamlit UI     | --------------------------> | FastAPI Backend            |  |
|  | (Port 8501)      | <-------------------------- | (Port 8000)                |  |
|  +------------------+                             +----------------------------+  |
|                                                     |                        |    |
|                                        SQL / pgvector                        |    |
|                                                     v                        |    |
|                                           +-------------------+              |    |
|                                           | PostgreSQL 16     |              |    |
|                                           | + pgvector        |              |    |
|                                           | (Port 5432)       |              |    |
|                                           +-------------------+              |    |
|                                                                              |    |
|                                            HTTP (host.docker.internal:11434) |    |
|                                                                              v    |
|                                           +------------------------------------+  |
|                                           | Ollama Service                     |  |
|                                           | - nomic-embed-text (Embeddings)    |  |
|                                           | - llama3.1:8b (Answer Generation)  |  |
|                                           +------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

All embeddings and generative inference execute on the local machine via Ollama. No data is transmitted to external cloud APIs, resulting in zero per-query API costs and complete data privacy.

---

## Tech Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Backend Framework | FastAPI | 0.115.5 | REST endpoints, dependency injection, and SSE streaming |
| ASGI Server | Uvicorn | 0.32.1 | Asynchronous server execution |
| Database | PostgreSQL + pgvector | 16 (Image: `pgvector/pgvector:pg16`) | Relational metadata storage and vector indexing |
| Database Driver | AsyncPG / Psycopg2-binary | 0.30.0 / 2.9.10 | Async runtime DB access / Sync DB access for migrations |
| ORM | SQLAlchemy (asyncio) | 2.0.36 | Async database mapping and query construction |
| Migrations | Alembic | 1.14.0 | Schema management and extension initialization |
| Vector Embeddings | nomic-embed-text | Ollama (768 dimensions) | Dense semantic vector representations |
| Generation LLM | llama3.1:8b | Ollama | Context-grounded answer synthesis and citation |
| Document Parsing | PyPDF | 5.1.0 | PDF text extraction per page |
| Tokenization | Tiktoken / NLTK | 0.8.0 / 3.9.1 | Exact token counting (`cl100k_base`) and sentence boundary detection |
| Configuration | Pydantic Settings | 2.6.1 | Environment variable parsing and validation |
| Frontend | Streamlit | 1.40.1 | Chat interface, document manager, and citation inspection |
| HTTP Client | HTTPX | 0.28.0 | Async communication with Ollama endpoints |
| Testing | Pytest / Pytest-AsyncIO | 8.3.3 / 0.24.0 | Unit test suite for chunking and parsing pipelines |
| Containerization | Docker / Docker Compose | Compose file v2 syntax | Multi-container orchestration (Postgres, Backend, Frontend) |

---

## Key Design Decisions

### 1. Local Models via Ollama vs. Cloud APIs
- **Cost**: Embedding and LLM generation run locally without incurring recurring token billing from cloud providers.
- **Data Privacy**: Proprietary documents and user queries remain strictly on the host system.
- **Reliability**: Eliminates external network latency, cloud rate limits, and third-party service outages.

### 2. Structural Parsing and Sentence-Boundary Chunking
Blind character or fixed-token splitting breaks sentences mid-thought and splits related paragraphs across arbitrary boundaries. The ingestion pipeline (`backend/app/services/ingestion.py`) implements a three-tier chunking strategy:
- **Format-Specific Structural Extraction**:
  - **PDF**: Processed page by page via `pypdf.PdfReader`, attaching page number metadata to each chunk.
  - **Markdown**: Parsed using regular expressions targeting ATX headings (`#` through `######`), extracting hierarchical section titles as metadata.
  - **Plain Text**: Segmented by double newline (`\n\n`) paragraph boundaries.
- **Sentence-Aware Sliding Window**:
  - Sections smaller than `max_chunk_tokens` (default: 500 tokens, counted via Tiktoken `cl100k_base`) are preserved as single chunks.
  - Oversized sections are tokenized into individual sentences via NLTK `sent_tokenize`.
  - Sentences are accumulated into a window until the token limit is reached. The window is emitted as a chunk, and the algorithm backtracks by `chunk_overlap_ratio` (default: 0.12, ~60 tokens) of complete sentences. This prevents mid-sentence cuts and preserves contextual continuity across chunk boundaries.

### 3. Similarity Threshold Guard
- Default threshold: `0.3` cosine similarity (`1 - (embedding <=> query_vec)`).
- Before invoking the LLM, `backend/app/services/generation.py` evaluates the top-ranked retrieved chunk. If no chunk meets or exceeds the similarity threshold, generation is bypassed entirely.
- The endpoint immediately returns: `"I could not find relevant information in the provided documents to answer your question."`
- This prevents the model from attempting to answer out of general pre-trained weights or hallucinating answers when no supporting evidence exists in the uploaded records.

### 4. PostgreSQL + pgvector vs. Dedicated Vector Databases
- **Single Source of Truth**: Document metadata (filenames, chunk indexes, character offsets, section headings, page numbers) and vector embeddings (`vector(768)`) reside in the same transactional database.
- **Relational Integrity**: Foreign key constraints with `ON DELETE CASCADE` guarantee that deleting a document cleans up all associated chunks, embeddings, and index entries atomically.
- **HNSW Indexing**: An HNSW index (`m = 16, ef_construction = 64`, `vector_cosine_ops`) is created via Alembic migrations, providing approximate nearest-neighbor retrieval without needing an external vector sync pipeline or secondary database service.

### 5. Migration and Database Ordering
The `pgvector` extension must exist prior to creating tables containing `vector` column definitions. The initial migration (`backend/alembic/versions/001_init.py`) executes `CREATE EXTENSION IF NOT EXISTS vector` as its first operation before creating the `documents` and `chunks` tables. The Docker Compose file enforces a health check on Postgres before starting the backend, where `alembic upgrade head` executes automatically on boot.

---

## Setup Instructions

### Prerequisites
1. **Docker Desktop** installed and running.
2. **Ollama** installed on the host machine.
3. Download the required embedding and generation models in Ollama:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3.1:8b
   ```
4. Verify models are available:
   ```bash
   ollama list
   ```

### Installation

1. Navigate to the project directory:
   ```bash
   cd c:\Users\lenov\OneDrive\shreya\Projects\Rag_QA
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   *(Note: API keys in `.env` can remain blank because all inference runs locally through Ollama).*

3. Build and launch the container stack:
   ```bash
   docker-compose down -v
   docker-compose up --build
   ```

### Access Endpoints

- **Frontend Chat UI**: [http://localhost:8501](http://localhost:8501)
- **Backend Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Usage

1. Open [http://localhost:8501](http://localhost:8501) in a web browser.
2. In the sidebar under **Document Manager**, select a `.pdf`, `.txt`, or `.md` file and click **Ingest Document**.
3. Once ingested, the document appears in the **Available Documents** list with its total chunk count.
4. Enter a question in the main chat prompt (e.g., `"What are the key terms in Section 2?"`).
5. The assistant streams the answer with inline citations (`[Chunk 1]`, `[Chunk 2]`).
6. Expand the **source citation(s)** drawer below the response to review the retrieved text snippets, source filename, section heading, page number, and similarity score.
7. To restrict retrieval to specific documents, select their corresponding checkboxes in the sidebar before querying.

---

## Project Structure

```
Rag_QA/
|-- .env.example
|-- .env
|-- docker-compose.yml
|-- README.md
|-- backend/
|   |-- Dockerfile
|   |-- requirements.txt
|   |-- alembic.ini
|   |-- alembic/
|   |   |-- env.py
|   |   |-- __init__.py
|   |   `-- versions/
|   |       `-- 001_init.py
|   |-- app/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- database.py
|   |   |-- main.py
|   |   |-- models.py
|   |   |-- schemas.py
|   |   |-- routers/
|   |   |   |-- __init__.py
|   |   |   |-- chat.py
|   |   |   `-- documents.py
|   |   `-- services/
|   |       |-- __init__.py
|   |       |-- generation.py
|   |       |-- ingestion.py
|   |       `-- retrieval.py
|   `-- tests/
|       |-- __init__.py
|       `-- test_chunking.py
`-- frontend/
    |-- Dockerfile
    |-- requirements.txt
    |-- app.py
    `-- .streamlit/
        `-- config.toml
```

---

## Known Limitations and Scope

This system represents a Phase 1 core RAG implementation. Deliberate boundaries in the current architecture include:

- **Vector-Only Retrieval**: Search relies exclusively on cosine similarity over dense embeddings. Exact keyword matches (e.g., alphanumeric serial codes, specific function signatures, or rare terms) do not yet benefit from sparse BM25/full-text search ranking.
- **No Second-Stage Reranking**: Chunks are retrieved and passed directly to the generator based on bi-encoder similarity scores without cross-encoder reranking.
- **Stateless Chat Context**: Queries are processed independently without conversational memory or multi-turn context tracking.
- **Single Document Namespace**: Documents are ingested into a shared table. Multi-tenant partitioning and user-scoped collection boundaries are not yet implemented.
- **Single Model Profile**: Ingestion and generation configurations are global and set via environment variables.

---

## Testing

The test suite validates the ingestion, text extraction, sentence segmentation, and chunk boundary algorithms.

### Running Unit Tests

Run tests using pytest inside the backend environment:

```bash
cd backend
pytest tests/test_chunking.py -v
```

### Test Coverage

The test suite (`backend/tests/test_chunking.py`) covers:
1. `test_overlap_ratio_within_bounds`: Confirms adjacent chunk overlap falls within configured bounds (5% to 20%) on large sections.
2. `test_chunks_end_at_sentence_boundary`: Ensures no chunk terminates mid-sentence by verifying terminal punctuation.
3. `test_section_heading_inherited`: Verifies that chunks generated from structured sections inherit parent heading metadata.
4. `test_short_document_one_chunk`: Verifies that texts shorter than the token threshold produce exactly one chunk with accurate character offsets.
5. `test_empty_section_produces_no_chunks`: Checks that empty or whitespace-only sections are skipped.
6. `test_chunk_indices_are_sequential`: Validates 0-indexed sequential ordering across all generated chunks.
7. `test_markdown_parser_extracts_headings`: Tests ATX heading extraction across `#`, `##`, and `###` levels.
8. `test_plain_text_paragraph_split`: Tests paragraph separation on double-newline boundaries.
9. `test_token_count_basic`: Validates token estimation behavior on empty and populated strings.
