import os
from groq import Groq
from dotenv import load_dotenv
from embedder import embed
from endee_client import search

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def answer_question(question: str) -> dict:
    """
    Full RAG pipeline:
    1. Embed the question
    2. Search Endee for relevant chunks
    3. Feed chunks as context to LLM
    4. Return answer + sources
    """

    # Step 1: Embed the question
    query_vector = embed([question])[0]

    # Step 2: Retrieve relevant chunks from Endee
    results = search(query_vector, top_k=5)

    if not results:
        return {
            "answer": "No relevant documents found. Please upload a document first.",
            "sources": []
        }

    # Step 3: Build context from retrieved chunks
    context_parts = []
    sources = []

    for r in results:
        # SDK returns objects, REST returns dicts — handle both
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
                "chunk": text[:150] + "..."  # preview
            })

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Call Groq LLM
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
        max_tokens=512,
        temperature=0.3
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources
    }