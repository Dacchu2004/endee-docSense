import requests
from endee import Endee, Precision
from endee.index import VectorItem

# ── Monkey-patch fix for Endee SDK bug ──────────────────────────────────────
# VectorItem is a Pydantic model — it has no .get() method, but the SDK
# internally calls v_item.get("filter", None) which crashes. This adds it.
if not hasattr(VectorItem, "get"):
    VectorItem.get = lambda self, key, default=None: getattr(self, key, default)
# ─────────────────────────────────────────────────────────────────────────────

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


def search(query_vector: list, top_k=5) -> list:
    """Searches Endee for similar vectors."""
    index = client.get_index(INDEX_NAME)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        ef=128
    )
    return results


def list_sources() -> dict:
    """Returns index info."""
    try:
        index = client.get_index(INDEX_NAME)
        return index.describe()
    except:
        return {}