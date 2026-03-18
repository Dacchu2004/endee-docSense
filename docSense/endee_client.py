import requests
from endee import Endee, Precision
from endee.index import VectorItem

if not hasattr(VectorItem, "get"):
    VectorItem.get = lambda self, key, default=None: getattr(self, key, default)

INDEX_NAME = "docsense"
DIMENSION = 384

client = Endee()

def ensure_index():
    """Creates the index if it doesn't exist yet."""
    try:
        client.get_index(INDEX_NAME)
        print(f"[Endee] Index '{INDEX_NAME}' already exists.")
    except:
        client.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            space_type="cosine",
            precision=Precision.INT8
        )
        print(f"[Endee] Index '{INDEX_NAME}' created successfully.")

def upsert_chunks(chunks: list):
    """Inserts vectors into Endee using the SDK."""
    index = client.get_index(INDEX_NAME)
    index.upsert(chunks)
    print(f"[Endee] Upserted {len(chunks)} chunks.")

def search(query_vector: list, top_k=5, source_filter: str = None) -> list:
    """Searches Endee for similar vectors. Scales top_k by indexed file count."""
    index = client.get_index(INDEX_NAME)

    try:
        from ingestion import load_registry
        file_count = len(load_registry())
        scaled_top_k = max(top_k, min(file_count * 4, 20))
    except:
        scaled_top_k = top_k

    results = index.query(
        vector=query_vector,
        top_k=scaled_top_k,
        ef=128
    )

    #filtering
    if source_filter:
        results = [
            r for r in results
            if (r.get("meta", {}) if isinstance(r, dict)
                else getattr(r, "meta", {}) or {}).get("source") == source_filter
        ]

    return results


def list_sources() -> dict:
    """Returns index info."""
    try:
        index = client.get_index(INDEX_NAME)
        return index.describe()
    except:
        return {}