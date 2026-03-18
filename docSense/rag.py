import os
import hashlib
from groq import Groq
from dotenv import load_dotenv
from embedder import embed
from endee_client import search

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── In-Memory Query Cache ─────────────────────────────────────────────────────
# Stores recent question→answer pairs to avoid redundant LLM calls.
# Key: MD5 hash of question | Value: full result dict
_query_cache: dict = {}
MAX_CACHE_SIZE = 50  # max number of cached answers


def _get_cache_key(question: str) -> str:
    """Creates a consistent hash key for a question."""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


def clear_cache():
    """Clears the query cache — called when new document is ingested."""
    _query_cache.clear()
    print("[Cache] Query cache cleared.")


def answer_question(question: str, source_filter: str = None) -> dict:
    """
    Full RAG pipeline with query caching:
    1. Check cache — return instantly if already answered
    2. Embed the question
    3. Search Endee for relevant chunks
    4. Feed chunks as context to LLM
    5. Cache and return answer
    """

    # Cache Check
    cache_key = _get_cache_key(question)
    if cache_key in _query_cache:
        print(f"[Cache] HIT for: '{question[:50]}'")
        cached = _query_cache[cache_key].copy()
        cached["cached"] = True
        return cached

    print(f"[Cache] MISS for: '{question[:50]}' — running RAG pipeline")

    #Embed question
    query_vector = embed([question])[0]

    #Retrieve relevant chunks from Endee
    results = search(query_vector, top_k=6, source_filter=source_filter)

    if not results:
        return {
            "answer": "No relevant documents found. Please upload a document first.",
            "sources": [],
            "cached": False
        }

    # Build context
    context_parts = []
    sources = []

    for r in results:
        if isinstance(r, dict):
            meta = r.get("meta", {})
            similarity = round(r.get("similarity", 0), 4)
        else:
            meta = getattr(r, "meta", {}) or {}
            similarity = round(getattr(r, "similarity", 0), 4)

        text = meta.get("text", "")
        source = meta.get("source", "unknown")

        if text:
            context_parts.append(f"[Source: {source}]\n{text}")
            sources.append({
                "source": source,
                "similarity": similarity,
                "chunk": text[:150] + "..."
            })

    context = "\n\n---\n\n".join(context_parts)

    # Call Groq LLM
    prompt = f"""You are a helpful assistant that answers questions based on provided documents.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.3
    )

    result = {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
        "cached": False
    }

    # Store in cache
    if len(_query_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entry when cache is full
        oldest_key = next(iter(_query_cache))
        del _query_cache[oldest_key]

    _query_cache[cache_key] = result
    print(f"[Cache] Stored answer for: '{question[:50]}'")

    return result