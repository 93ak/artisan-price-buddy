"""
market_index_service.py

Embeds the clean market dataset with SentenceTransformer and stores it in
its own ChromaDB collection — "market_index" — in its own persistent
directory (data/market/chroma_market/), completely separate from wherever
your existing pricing RAG's Chroma store lives. Nothing here touches
rag/rag_service.py or its collection.

ASSUMPTION: embedding model is all-MiniLM-L6-v2 (fast, 384-dim, good default
for short product text). Change EMBED_MODEL_NAME if you want this to match
whatever your pricing RAG already uses.
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market"
CHROMA_DIR = DATA_DIR / "chroma_market"
COLLECTION_NAME = "market_index"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_client = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def _embedding_text(record: dict) -> str:
    parts = [
        record.get("title", ""),
        record.get("category", ""),
        ", ".join(record.get("materials", [])),
        ", ".join(record.get("keywords", [])),
        (record.get("description", "") or "")[:300],
    ]
    return " | ".join(p for p in parts if p)


def _to_metadata(record: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool — lists get flattened
    to comma-joined strings and parsed back out in _from_metadata."""
    return {
        "title": record.get("title", ""),
        "description": record.get("description", "") or "",
        "price": float(record.get("price", 0) or 0),
        "seller": record.get("seller", ""),
        "category": record.get("category", ""),
        "materials": ", ".join(record.get("materials", [])),
        "keywords": ", ".join(record.get("keywords", [])),
        "source": record.get("source", "Yuukke Marketplace"),
    }


def _from_metadata(record_id: str, metadata: dict) -> dict:
    return {
        "id": record_id,
        "title": metadata.get("title", ""),
        "description": metadata.get("description", ""),
        "price": metadata.get("price", 0.0),
        "seller": metadata.get("seller", ""),
        "category": metadata.get("category", ""),
        "materials": [m.strip() for m in metadata.get("materials", "").split(",") if m.strip()],
        "keywords": [k.strip() for k in metadata.get("keywords", "").split(",") if k.strip()],
        "source": metadata.get("source", "Yuukke Marketplace"),
    }


def build_market_index(dataset: list[dict], batch_size: int = 200) -> None:
    """Rebuilds market_index from scratch with the given dataset."""
    client = _get_client()
    model = _get_model()

    # Drop and recreate so a rebuild never mixes stale + fresh records
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        texts = [_embedding_text(r) for r in batch]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()

        collection.add(
            ids=[r["id"] for r in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[_to_metadata(r) for r in batch],
        )
        print(f"  indexed {min(i + batch_size, len(dataset))}/{len(dataset)}")

    print(f"market_index built: {collection.count()} records -> {CHROMA_DIR}")


def index_exists() -> bool:
    client = _get_client()
    try:
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count() > 0
    except Exception:
        return False


def search_market(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over market_index. Returns dataset records + _score (0-1, higher = closer)."""
    client = _get_client()
    model = _get_model()

    collection = client.get_collection(COLLECTION_NAME)
    q_embedding = model.encode([query], convert_to_numpy=True).tolist()

    result = collection.query(query_embeddings=q_embedding, n_results=top_k)

    records = []
    ids = result.get("ids", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for record_id, metadata, distance in zip(ids, metadatas, distances):
        record = _from_metadata(record_id, metadata)
        # Chroma's default space is L2 on normalized embeddings for most sentence-transformer
        # models in practice reads as "smaller distance = closer" — flip to a 0-1-ish score.
        record["_score"] = 1 / (1 + distance)
        records.append(record)

    return records