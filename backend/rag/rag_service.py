"""
RAG service using Postgres + pgvector (persists across Render deploys,
unlike the old Chroma on-disk store) + sentence-transformers (local, CPU).
Model: all-MiniLM-L6-v2 (~90MB, downloads once on first run)
"""
import json
import logging
import os
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from rag.dataset import ARTISAN_PRODUCTS

logger = logging.getLogger("uvicorn.error")

DATABASE_URL = os.environ["DATABASE_URL"]
TABLE_NAME = "artisan_products"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

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
    return get_embedder().encode(texts, show_progress_bar=False).tolist()


def get_conn():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)  # lets us pass python lists straight in as `vector`
    return conn


def _ensure_table():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                embedding vector({EMBED_DIM}) NOT NULL,
                category TEXT,
                material_cost REAL,
                labor_hours REAL,
                complexity TEXT,
                experience_level TEXT,
                selling_price REAL,
                tags TEXT,
                description TEXT
            );
        """)
    conn.commit()
    conn.close()


def _product_to_text(p: dict) -> str:
    return (
        f"{p['description']}. "
        f"Category: {p['category']}. "
        f"Complexity: {p['complexity']}. "
        f"Experience: {p['experience_level']}. "
        f"Tags: {', '.join(p['tags'])}."
    )


async def index_dataset(force: bool = False) -> dict:
    _ensure_table()
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
        existing = cur.fetchone()["n"]

    if existing >= len(ARTISAN_PRODUCTS) and not force:
        conn.close()
        logger.info(f"RAG: Already indexed {existing} products, skipping.")
        return {"status": "skipped", "count": existing}

    logger.info(f"RAG: Indexing {len(ARTISAN_PRODUCTS)} products with {EMBED_MODEL_NAME}...")
    texts = [_product_to_text(p) for p in ARTISAN_PRODUCTS]
    embeddings = embed(texts)

    with conn.cursor() as cur:
        for p, text, emb in zip(ARTISAN_PRODUCTS, texts, embeddings):
            cur.execute(
                f"""
                INSERT INTO {TABLE_NAME}
                    (id, document, embedding, category, material_cost, labor_hours,
                     complexity, experience_level, selling_price, tags, description)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document = EXCLUDED.document,
                    embedding = EXCLUDED.embedding,
                    category = EXCLUDED.category,
                    material_cost = EXCLUDED.material_cost,
                    labor_hours = EXCLUDED.labor_hours,
                    complexity = EXCLUDED.complexity,
                    experience_level = EXCLUDED.experience_level,
                    selling_price = EXCLUDED.selling_price,
                    tags = EXCLUDED.tags,
                    description = EXCLUDED.description
                """,
                (p["id"], text, emb, p["category"], p["material_cost"], float(p["labor_hours"]),
                 p["complexity"], p["experience_level"], p["selling_price"],
                 json.dumps(p["tags"]), p["description"])
            )
    conn.commit()
    conn.close()
    logger.info(f"RAG: Indexed {len(ARTISAN_PRODUCTS)} products.")
    return {"status": "indexed", "count": len(ARTISAN_PRODUCTS), "model": EMBED_MODEL_NAME}


async def retrieve_similar(query: str, n_results: int = 5,
                           category_filter: str = None) -> list[dict]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
        count = cur.fetchone()["n"]
    conn.close()

    if count == 0:
        await index_dataset()
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
            count = cur.fetchone()["n"]
        conn.close()

    n = min(n_results, count)
    if n <= 0:
        return []

    q_embedding = embed([query])[0]

    sql = f"""
        SELECT id, description, category, material_cost, labor_hours, complexity,
            experience_level, selling_price, tags,
            embedding <=> %s::vector AS distance
        FROM {TABLE_NAME}
    """
    params = [q_embedding]
    if category_filter:
        sql += " WHERE category = %s"
        params.append(category_filter)
    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params += [q_embedding, n]

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()

    retrieved = []
    for r in rows:
        retrieved.append({
            "id": r["id"],
            "description": r["description"],
            "category": r["category"],
            "material_cost": r["material_cost"],
            "labor_hours": r["labor_hours"],
            "complexity": r["complexity"],
            "experience_level": r["experience_level"],
            "selling_price": r["selling_price"],
            "tags": json.loads(r["tags"]),
            "similarity_score": round(1 - r["distance"], 3),
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
        _ensure_table()
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
            n = cur.fetchone()["n"]
        conn.close()
        return {
            "indexed": n,
            "total": len(ARTISAN_PRODUCTS),
            "ready": n > 0,
            "embed_model": EMBED_MODEL_NAME
        }
    except Exception as e:
        return {"indexed": 0, "total": len(ARTISAN_PRODUCTS), "ready": False, "error": str(e)}