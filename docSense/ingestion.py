import io
import uuid
import json
import os
import PyPDF2
from docx import Document
from embedder import embed
from endee_client import upsert_chunks, ensure_index, client, INDEX_NAME

# ── File Registry ─────────────────────────────────────────────────────────────
# Tracks which files have been ingested and how many chunks each has.
# Stored as a simple JSON file so it persists across restarts.
REGISTRY_PATH = "ingested_files.json"


def load_registry() -> dict:
    """Loads the ingestion registry from disk."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    """Saves the ingestion registry to disk."""
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def is_already_ingested(filename: str) -> bool:
    """Returns True if this file has already been indexed."""
    registry = load_registry()
    return filename in registry


def get_ingested_files() -> list:
    """Returns list of all ingested filenames."""
    return list(load_registry().keys())


# ── Text Extraction ───────────────────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extracts raw text from PDF, DOCX, or TXT files."""

    if filename.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text.strip()

    elif filename.lower().endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        return " ".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif filename.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    return ""


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size=1200, overlap=150) -> list:
    """
    Splits text into overlapping chunks.
    Larger chunk_size = more context per chunk.
    overlap = shared characters between consecutive chunks
    so meaning is not lost at boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Main Ingestion ────────────────────────────────────────────────────────────

def ingest_document(file_bytes: bytes, filename: str) -> dict:
    """
    Production-grade ingestion pipeline:
    1. Check registry — if file exists, delete old chunks first
    2. Extract text from file
    3. Split into chunks
    4. Embed each chunk
    5. Store in Endee
    6. Update registry
    Returns a result dict with status and chunk count.
    """
    ensure_index()
    registry = load_registry()

    # ── Duplicate Prevention ──────────────────────────────────────────────────
    if filename in registry:
        print(f"[Ingestion] '{filename}' already exists — removing old chunks first.")
        # Delete old vectors by their stored IDs
        old_ids = registry[filename].get("chunk_ids", [])
        if old_ids:
            try:
                index = client.get_index(INDEX_NAME)
                for chunk_id in old_ids:
                    try:
                        index.delete_vector(chunk_id)
                    except:
                        pass  # Already deleted or not found
                print(f"[Ingestion] Removed {len(old_ids)} old chunks for '{filename}'.")
            except Exception as e:
                print(f"[Ingestion] Warning during cleanup: {e}")

    # ── Extract ───────────────────────────────────────────────────────────────
    text = extract_text(file_bytes, filename)
    if not text:
        return {"success": False, "error": "Could not extract text from file", "chunks": 0}

    # Dynamic chunk 
    doc_length = len(text)
    if doc_length < 5000:
        chunk_size = 1200
        overlap = 150
    elif doc_length < 20000:
        chunk_size = 1000
        overlap = 150
    elif doc_length < 100000:
        chunk_size = 800
        overlap = 100
    else:
        chunk_size = 600
        overlap = 80

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"[Ingestion] Doc length: {doc_length} chars → chunk_size={chunk_size}, chunks={len(chunks)}")
    if not chunks:
        return {"success": False, "error": "No content found after chunking", "chunks": 0}

    # Embed
    vectors = embed(chunks)

    # Build items
    items = []
    chunk_ids = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{filename}-chunk-{i}-{uuid.uuid4().hex[:8]}"
        chunk_ids.append(chunk_id)
        items.append({
            "id": chunk_id,
            "vector": vector,
            "meta": {
                "text": chunk,
                "source": filename,
                "chunk_idx": i
            }
        })

    # Upsert into Endee
    upsert_chunks(items)

    # Update registry
    registry[filename] = {
        "chunk_count": len(items),
        "chunk_ids": chunk_ids
    }
    save_registry(registry)

    print(f"[Ingestion] ✅ '{filename}' indexed with {len(items)} chunks.")
    return {"success": True, "chunks": len(items), "filename": filename}