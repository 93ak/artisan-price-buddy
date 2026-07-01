"""
RAG service using ChromaDB + sentence-transformers (local, CPU, no Ollama needed).
Model: all-MiniLM-L6-v2 (~90MB, downloads once on first run)
"""
import json
import logging
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rag.dataset import ARTISAN_PRODUCTS

logger = logging.getLogger("uvicorn.error")

CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "artisan_products_v2"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_client = None
_collection = None
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("RAG: Loading sentence-transformers model (first run may download ~90MB)...")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
        logger.info("RAG: Embedder ready.")
    return _embedder


def embed(texts: list[str]) -> list[list[float]]:
    """Synchronous local embedding — fast enough to not need async."""
    model = get_embedder()
    return model.encode(texts, show_progress_bar=False).tolist()


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None  # we supply our own
        )
    return _collection


def _reset_collection():
    """
    Drop and recreate the collection. Used when the persisted on-disk index
    is corrupted or incompatible with the installed chromadb version
    (symptom: cryptic errors like "object of type 'int' has no len()" from
    collection.count()/query()). Caller is responsible for re-indexing after.
    """
    global _collection
    logger.warning("RAG: collection appears corrupted/incompatible — resetting it.")
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None
    )
    return _collection


def _product_to_text(p: dict) -> str:
    return (
        f"{p['description']}. "
        f"Category: {p['category']}. "
        f"Complexity: {p['complexity']}. "
        f"Experience: {p['experience_level']}. "
        f"Tags: {', '.join(p['tags'])}."
    )


async def index_dataset(force: bool = False) -> dict:
    collection = get_collection()
    try:
        existing = collection.count()
    except Exception as e:
        logger.warning(f"RAG: collection.count() failed ({e}) — resetting collection.")
        collection = _reset_collection()
        existing = 0

    if existing >= len(ARTISAN_PRODUCTS) and not force:
        logger.info(f"RAG: Already indexed {existing} products, skipping.")
        return {"status": "skipped", "count": existing}

    logger.info(f"RAG: Indexing {len(ARTISAN_PRODUCTS)} products with {EMBED_MODEL_NAME}...")
    texts = [_product_to_text(p) for p in ARTISAN_PRODUCTS]
    ids = [p["id"] for p in ARTISAN_PRODUCTS]
    metadatas = [{
        "category": p["category"],
        "material_cost": p["material_cost"],
        "labor_hours": float(p["labor_hours"]),
        "complexity": p["complexity"],
        "experience_level": p["experience_level"],
        "selling_price": p["selling_price"],
        "tags": json.dumps(p["tags"]),
        "description": p["description"],
    } for p in ARTISAN_PRODUCTS]

    embeddings = embed(texts)

    if existing > 0:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info(f"RAG: Indexed {len(ARTISAN_PRODUCTS)} products.")
    return {"status": "indexed", "count": len(ARTISAN_PRODUCTS), "model": EMBED_MODEL_NAME}


async def retrieve_similar(query: str, n_results: int = 5,
                           category_filter: str = None) -> list[dict]:
    collection = get_collection()
    try:
        count = collection.count()
    except Exception as e:
        logger.warning(f"RAG: collection.count() failed ({e}) — resetting and re-indexing.")
        await index_dataset(force=True)
        collection = get_collection()
        count = collection.count()

    if count == 0:
        await index_dataset()
        collection = get_collection()
        count = collection.count()

    n = min(n_results, count)
    if n <= 0:
        return []
    where = {"category": category_filter} if category_filter else None
    query_embedding = embed([query])

    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.warning(f"RAG: query failed ({e}) — resetting, re-indexing, retrying once.")
        await index_dataset(force=True)
        collection = get_collection()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

    retrieved = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        similarity = round(1 - results["distances"][0][i], 3)
        retrieved.append({
            "id": results["ids"][0][i],
            "description": meta["description"],
            "category": meta["category"],
            "material_cost": meta["material_cost"],
            "labor_hours": meta["labor_hours"],
            "complexity": meta["complexity"],
            "experience_level": meta["experience_level"],
            "selling_price": meta["selling_price"],
            "tags": json.loads(meta["tags"]),
            "similarity_score": similarity,
        })
    return retrieved


def format_for_prompt(retrieved: list[dict]) -> str:
    lines = ["Similar products from knowledge base:"]
    for i, p in enumerate(retrieved, 1):
        lines.append(
            f"{i}. {p['description']} | "
            f"Materials: ₹{p['material_cost']} | "
            f"Labor: {p['labor_hours']}hrs | "
            f"Sold at: ₹{p['selling_price']}"
        )
    return "\n".join(lines)


async def get_index_status() -> dict:
    try:
        c = get_collection()
        return {
            "indexed": c.count(),
            "total": len(ARTISAN_PRODUCTS),
            "ready": c.count() > 0,
            "embed_model": EMBED_MODEL_NAME
        }
    except Exception as e:
        return {"indexed": 0, "total": len(ARTISAN_PRODUCTS), "ready": False, "error": str(e)}