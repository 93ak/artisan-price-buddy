"""
market_index_service.py

Embeds the clean market dataset with SentenceTransformer and stores it in
Postgres (pgvector) — table "market_index" — separate from artisan_products.
Nothing here touches rag/rag_service.py or its table.

ASSUMPTION: embedding model is all-MiniLM-L6-v2 (fast, 384-dim, good default
for short product text) — same as rag_service.py, so both tables share a
dimension. If you ever change one, change both.
"""

import json
import os

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ["DATABASE_URL"]
TABLE_NAME = "market_index"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def get_conn():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    register_vector(conn)
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
                title TEXT,
                description TEXT,
                price REAL,
                seller TEXT,
                category TEXT,
                materials TEXT,
                keywords TEXT,
                source TEXT
            );
        """)
    conn.commit()
    conn.close()


def _embedding_text(record: dict) -> str:
    parts = [
        record.get("title", ""),
        record.get("category", ""),
        ", ".join(record.get("materials", [])),
        ", ".join(record.get("keywords", [])),
        (record.get("description", "") or "")[:300],
    ]
    return " | ".join(p for p in parts if p)


def build_market_index(dataset: list[dict], batch_size: int = 200) -> None:
    """Rebuilds market_index from scratch with the given dataset."""
    _ensure_table()
    model = _get_model()
    conn = get_conn()

    # Wipe and rebuild so a rebuild never mixes stale + fresh records
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {TABLE_NAME}")
    conn.commit()

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        texts = [_embedding_text(r) for r in batch]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()

        with conn.cursor() as cur:
            for r, text, emb in zip(batch, texts, embeddings):
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME}
                        (id, document, embedding, title, description, price, seller,
                         category, materials, keywords, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        document = EXCLUDED.document,
                        embedding = EXCLUDED.embedding,
                        title = EXCLUDED.title,
                        description = EXCLUDED.description,
                        price = EXCLUDED.price,
                        seller = EXCLUDED.seller,
                        category = EXCLUDED.category,
                        materials = EXCLUDED.materials,
                        keywords = EXCLUDED.keywords,
                        source = EXCLUDED.source
                    """,
                    (
                        r["id"], text, emb, r.get("title", ""), r.get("description", "") or "",
                        float(r.get("price", 0) or 0), r.get("seller", ""), r.get("category", ""),
                        json.dumps(r.get("materials", [])), json.dumps(r.get("keywords", [])),
                        r.get("source", "Yuukke Marketplace"),
                    )
                )
        conn.commit()
        print(f"  indexed {min(i + batch_size, len(dataset))}/{len(dataset)}")

    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
        count = cur.fetchone()["n"]
    conn.close()
    print(f"market_index built: {count} records -> Postgres ({TABLE_NAME})")


def index_exists() -> bool:
    try:
        _ensure_table()
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
            count = cur.fetchone()["n"]
        conn.close()
        return count > 0
    except Exception:
        return False


def search_market(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search over market_index. Returns dataset records + _score (0-1, higher = closer)."""
    model = _get_model()
    q_embedding = model.encode([query], convert_to_numpy=True).tolist()[0]

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, title, description, price, seller, category, materials, keywords, source,
                embedding <=> %s::vector AS distance
            FROM {TABLE_NAME}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (q_embedding, q_embedding, top_k)
        )
        rows = cur.fetchall()
    conn.close()

    records = []
    for r in rows:
        records.append({
            "id": r["id"],
            "title": r["title"],
            "description": r["description"],
            "price": r["price"],
            "seller": r["seller"],
            "category": r["category"],
            "materials": json.loads(r["materials"]) if r["materials"] else [],
            "keywords": json.loads(r["keywords"]) if r["keywords"] else [],
            "source": r["source"],
            # cosine distance -> a 0-1-ish "closer is higher" score,
            # same shape as the old Chroma L2 conversion so retrieval.py
            # (MIN_RELEVANCE_SCORE = 0.35) doesn't need to change.
            "_score": 1 / (1 + r["distance"]),
        })
    return records