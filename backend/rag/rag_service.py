"""
RAG service using ChromaDB + nomic-embed-text via Ollama.
nomic-embed-text must be pulled: ollama pull nomic-embed-text
"""
import httpx
import json
import logging
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from rag.dataset import ARTISAN_PRODUCTS

logger = logging.getLogger("uvicorn.error")

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "artisan_products_v1"

_client = None
_collection = None

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
        # embedding_function=None because we supply our own via Ollama
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None
        )
    return _collection

async def embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama nomic-embed-text."""
    embeddings = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for text in texts:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text}
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
    return embeddings

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
    existing = collection.count()

    if existing >= len(ARTISAN_PRODUCTS) and not force:
        logger.info(f"RAG: Already indexed {existing} products, skipping.")
        return {"status": "skipped", "count": existing}

    logger.info(f"RAG: Embedding {len(ARTISAN_PRODUCTS)} products with {EMBED_MODEL}...")
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

    embeddings = await embed(texts)

    if existing > 0:
        try:
            collection.delete(ids=ids)
        except Exception:
            pass

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    logger.info(f"RAG: Indexed {len(ARTISAN_PRODUCTS)} products.")
    return {"status": "indexed", "count": len(ARTISAN_PRODUCTS), "model": EMBED_MODEL}

async def retrieve_similar(query: str, n_results: int = 5, category_filter: str = None) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        await index_dataset()

    n = min(n_results, collection.count())
    where = {"category": category_filter} if category_filter else None
    query_embedding = await embed([query])

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
    lines = ["Similar products from our knowledge base:"]
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
        return {"indexed": c.count(), "total": len(ARTISAN_PRODUCTS), "ready": c.count() > 0}
    except Exception as e:
        return {"indexed": 0, "total": len(ARTISAN_PRODUCTS), "ready": False, "error": str(e)}
