import io
import uuid
import PyPDF2
from docx import Document
from embedder import embed
from endee_client import upsert_chunks, ensure_index


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
        return " ".join([para.text for para in doc.paragraphs if para.text.strip()])

    elif filename.lower().endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    else:
        return ""


def chunk_text(text: str, chunk_size=500, overlap=50) -> list:
    """
    Splits text into overlapping chunks.
    overlap=50 means consecutive chunks share 50 characters
    so context is not lost at boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 30:  # skip tiny/empty chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def ingest_document(file_bytes: bytes, filename: str) -> int:
    """
    Full pipeline:
    1. Extract text from file
    2. Split into chunks
    3. Embed each chunk
    4. Store in Endee with metadata
    Returns number of chunks stored.
    """
    # Make sure index exists
    ensure_index()

    # Step 1: Extract
    text = extract_text(file_bytes, filename)
    if not text:
        return 0

    # Step 2: Chunk
    chunks = chunk_text(text)
    if not chunks:
        return 0

    # Step 3: Embed all chunks at once (faster than one by one)
    vectors = embed(chunks)

    # Step 4: Build items and upsert into Endee
    items = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        items.append({
            "id": f"{filename}-chunk-{i}-{uuid.uuid4().hex[:8]}",
            "vector": vector,
            "meta": {
                "text": chunk,
                "source": filename,
                "chunk_idx": i
            }
        })

    upsert_chunks(items)
    return len(items)