# DocSense — RAG-Powered Document Q&A using Endee Vector Database

DocSense is a production-grade Retrieval-Augmented Generation (RAG) application
that lets users upload documents (PDF, DOCX, TXT) and ask natural language questions.
It uses **Endee** as the vector database for semantic retrieval, 
**sentence-transformers** for embeddings, and **Groq LLaMA 3** for answer generation.

**Document Ingestion Flow:**
```
File Upload ──► Text Extraction (PyPDF2 / python-docx)
    ──► Dynamic Chunking (size adapts to doc length)
    ──► Batch Embedding (sentence-transformers)
    ──► Upsert into Endee (id, vector, metadata)
    ──► Registry update (ingested_files.json)
```

---

## How Endee is Used

| Operation | Details |
|---|---|
| **Index** | `docsense` — 384-dim, cosine similarity, INT8 precision |
| **Upsert** | Each document chunk stored with `id`, `vector`, `meta` (text, source, chunk_idx) |
| **Query** | Question embedding used to retrieve top-K similar chunks via HNSW ANN search |
| **Scaled retrieval** | `top_k` scales dynamically with number of indexed files for fair coverage |
| **Source filtering** | Client-side filtering by filename for single-document Q&A |
| **Delete** | Individual chunk deletion by stored IDs when file is removed |

---

## Features

- **Multi-document support** — Upload up to 10 files simultaneously
- **Semantic search** — Vector similarity via Endee, not keyword matching
- **Dynamic chunking** — Chunk size adapts to document length (resumes vs 100-page reports)
- **Duplicate prevention** — Re-uploading same file auto-replaces old chunks
- **In-memory query cache** — MD5-keyed cache eliminates redundant LLM calls
- **Model pre-warming** — Embedding model loaded at startup, not on first request
- **Per-file filtering** — Ask questions scoped to a single document
- **File management** — View, filter, and delete indexed documents from UI
- **Cache invalidation** — Cache clears automatically when new document is ingested

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Database | [Endee](https://github.com/endee-io/endee) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) |
| LLM | Groq API (`llama-3.1-8b-instant`) |
| Backend | Python Flask |
| Frontend | HTML, Tailwind CSS, Vanilla JS |
| Infrastructure | Docker (Endee server) |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Docker Desktop (running)
- Groq API key — free at https://console.groq.com

### 1. Star and Fork the Endee Repository
```bash
# Star: https://github.com/endee-io/endee
# Fork to your GitHub account, then clone your fork:
git clone https://github.com/YOUR_USERNAME/endee.git
cd endee
```

### 2. Start Endee Vector Database
```bash
cd docSense
docker compose up -d

# Verify:
docker ps
# Should show endee-server with status: healthy
```

### 3. Install Python Dependencies
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Create .env file:
echo GROQ_API_KEY=your_groq_api_key_here > .env
```

### 5. Run the Application
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Usage

1. **Upload** — Click the upload zone, select one or more PDF/DOCX/TXT files
2. **Index** — Click "Upload & Index" — files are chunked, embedded, and stored in Endee
3. **Ask** — Type any natural language question and press Enter or click Ask
4. **Filter** — Click "Filter" on any file to scope questions to that document only
5. **Delete** — Remove any file and all its indexed vectors from Endee

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/health` | Health check — Flask + Endee status |
| `GET` | `/files` | List all indexed files with chunk counts |
| `POST` | `/ingest` | Upload and index a document |
| `POST` | `/ask` | Ask a question (RAG pipeline) |
| `POST` | `/delete-file` | Delete a file and its vectors from Endee |

---

## Project Structure
```
docSense/
├── app.py              # Flask API — all routes
├── endee_client.py     # Endee SDK wrapper — index, upsert, search
├── embedder.py         # Sentence transformer embedding
├── ingestion.py        # Document extraction, chunking, registry
├── rag.py              # RAG pipeline with query caching
├── templates/
│   └── index.html      # Single-page web UI
├── docker-compose.yml  # Endee server setup
├── requirements.txt
├── .env.example
└── ingested_files.json # Auto-generated file registry
```

---

## Performance Optimizations

- **Startup pre-warming** — Embedding model loaded once at startup, eliminating cold-start latency on first request
- **In-memory LRU cache** — MD5-hashed query cache (max 50 entries) returns repeated questions instantly without re-embedding or LLM calls
- **Batch embedding** — All chunks of a document are embedded in a single model call
- **Dynamic chunk sizing** — Smaller chunks for large documents improve retrieval precision; larger chunks for short documents preserve context
- **Scaled top-k retrieval** — Retrieval count scales with indexed file count ensuring all documents get fair representation

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | Groq API key for LLaMA 3 | Yes |