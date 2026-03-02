import os
import uuid
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PointIdsList

load_dotenv()

HF_API_KEY  = os.getenv("HF_API_KEY")
QDRANT_URL  = os.getenv("QDRANT_URL")
QDRANT_KEY  = os.getenv("QDRANT_API_KEY")

HF_EMBEDDING_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

COLLECTION_NAME = "long_term_memory"
VECTOR_SIZE     = 384

# ── Connect to Qdrant ─────────────────────────────────────────────────────
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)


def _ensure_collection():
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"[VECTOR_STORE] Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"[VECTOR_STORE] Collection '{COLLECTION_NAME}' already exists.")

_ensure_collection()


# ── Embedding ─────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": [text]}

    response = requests.post(HF_EMBEDDING_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")

    result = response.json()
    # Handle both nested and flat response formats
    embedding = result[0] if isinstance(result[0], list) else result
    return [float(x) for x in embedding]


# ── ADD ───────────────────────────────────────────────────────────────────
def add_memory(text: str) -> str:
    memory_id = str(uuid.uuid4())
    embedding  = get_embedding(text)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=memory_id, vector=embedding, payload={"text": text})]
    )
    print(f"[ADD] '{text}' → ID: {memory_id[:8]}...")
    return memory_id


# ── SEARCH ────────────────────────────────────────────────────────────────
def search_similar_memories(text: str, top_s: int = 5) -> list:
    """
    Returns top-s similar memories with score >= 0.50.
    Lower threshold so the LLM can see contradicting memories too.
    The LLM — not the threshold — decides if they're related enough.
    """
    count = client.count(collection_name=COLLECTION_NAME).count
    if count == 0:
        return []

    embedding = get_embedding(text)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=min(top_s, count)
    )

    memories = []
    for result in results.points:
        if result.score >= 0.50:   # Low threshold — let the LLM judge relevance
            memories.append({
                "id":    result.id,
                "text":  result.payload["text"],
                "score": result.score
            })

    return memories


# ── UPDATE ────────────────────────────────────────────────────────────────
def update_memory(memory_id: str, new_text: str):
    new_embedding = get_embedding(new_text)
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=memory_id, vector=new_embedding, payload={"text": new_text})]
    )
    print(f"[UPDATE] ID {memory_id[:8]}... → '{new_text}'")


# ── DELETE ────────────────────────────────────────────────────────────────
def delete_memory(memory_id: str):
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[memory_id])
    )
    print(f"[DELETE] ID {memory_id[:8]}... removed.")


# ── GET ALL ───────────────────────────────────────────────────────────────
def get_all_memories() -> list:
    count = client.count(collection_name=COLLECTION_NAME).count
    if count == 0:
        return []

    results, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=200,
        with_payload=True,
        with_vectors=False
    )
    return [{"id": p.id, "text": p.payload["text"]} for p in results]


# ── CLEAR ALL ─────────────────────────────────────────────────────────────
def clear_all_memories():
    client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print("[VECTOR_STORE] All memories cleared.")