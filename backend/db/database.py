import json
import os
from datetime import datetime, timedelta

import psycopg
from psycopg.rows import dict_row

# Render injects DATABASE_URL automatically when the Postgres and web service
# are in the same project. Locally, set it in your .env to the "External
# Database URL" shown on the Render Postgres dashboard.
DATABASE_URL = os.environ["DATABASE_URL"]

# The fields the LLM is allowed to ask follow-up questions about.
# Order matters: this is the priority order questions are asked in.
# marketplace_platform is intentionally last.
# "quality" merges what used to be two separate fields (work_quality +
# uniqueness) into a single question — they're asked together now.
REQUIRED_PROFILE_FIELDS = [
    "experience_level",
    "labor_hours",
    "packaging_quality",
    "quality",
    "marketplace_platform",
]


def get_conn():
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_db():
    """
    Idempotent — safe to call on every startup. Schema itself should already
    exist (run schema.sql once against the DB), this just guarantees it if
    you're spinning up a fresh instance.
    """
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                product_type TEXT,
                summary TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                product_type TEXT,
                costs TEXT,
                floor_price REAL,
                tiers TEXT,
                confidence REAL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS market_cache (
                id SERIAL PRIMARY KEY,
                query TEXT UNIQUE NOT NULL,
                result TEXT NOT NULL,
                cached_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_profile (
                session_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL
            );
        """)
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    conn = get_conn()
    with conn.cursor() as cur:
        # Session must exist BEFORE the message insert — unlike sqlite,
        # Postgres actually enforces the FK on messages.session_id.
        cur.execute(
            "INSERT INTO sessions (id, created_at) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (session_id, datetime.utcnow().isoformat())
        )
        cur.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
            (session_id, role, content, datetime.utcnow().isoformat())
        )
    conn.commit()
    conn.close()


def get_session_messages(session_id: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = %s ORDER BY id",
            (session_id,)
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_sessions():
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT s.id, s.created_at, s.product_type, s.summary, COUNT(m.id) as msg_count "
            "FROM sessions s LEFT JOIN messages m ON s.id = m.session_id "
            "GROUP BY s.id ORDER BY s.created_at DESC LIMIT 20"
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_session_meta(session_id: str, product_type: str, summary: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sessions SET product_type = %s, summary = %s WHERE id = %s",
            (product_type, summary, session_id)
        )
    conn.commit()
    conn.close()


def save_analysis(session_id: str, data: dict):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO analyses (session_id, product_type, costs, floor_price, tiers, confidence, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                session_id,
                data.get("product_type"),
                json.dumps(data.get("costs", {})),
                data.get("floor_price"),
                json.dumps(data.get("tiers", {})),
                data.get("confidence"),
                datetime.utcnow().isoformat()
            )
        )
    conn.commit()
    conn.close()


def get_market_cache(query: str):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT result, cached_at FROM market_cache WHERE query = %s", (query,)
        )
        row = cur.fetchone()
    conn.close()
    if row:
        cached_at = datetime.fromisoformat(row["cached_at"])
        if datetime.utcnow() - cached_at < timedelta(hours=6):
            return json.loads(row["result"])
    return None


def set_market_cache(query: str, result: dict):
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO market_cache (query, result, cached_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (query) DO UPDATE SET result = EXCLUDED.result, cached_at = EXCLUDED.cached_at",
            (query, json.dumps(result), datetime.utcnow().isoformat())
        )
    conn.commit()
    conn.close()


# ── Cross-session context ──────────────────────────────────────────────────────

def get_recent_analyses(limit: int = 3) -> list[dict]:
    """
    Fetch the last N analyses across ALL sessions.
    Used to build cross-session context for the LLM.
    """
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT product_type, costs, floor_price, tiers, confidence, created_at "
            "FROM analyses ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def build_cross_session_context(limit: int = 3) -> str:
    """
    Build a compact summary of recent past analyses to inject as context.
    Helps the LLM be consistent and refer to prior products if relevant.
    """
    analyses = get_recent_analyses(limit)
    if not analyses:
        return ""

    lines = ["User's recent pricing history (for context only, do not repeat unless relevant):"]
    for a in analyses:
        try:
            costs = json.loads(a["costs"]) if a["costs"] else {}
            tiers = json.loads(a["tiers"]) if a["tiers"] else {}
            std_price = tiers.get("standard", {}).get("price", "?")
            mat = costs.get("materials", "?")
            lab = costs.get("labor", "?")
            lines.append(
                f"- {a['product_type'] or 'Unknown'}: "
                f"materials ₹{mat}, labor ₹{lab}, "
                f"floor ₹{a['floor_price']}, recommended ₹{std_price} "
                f"({a['created_at'][:10]})"
            )
        except Exception:
            continue

    return "\n".join(lines)


# ── Session profile ──────────────────────────────────────────────────────────
# A per-session "fact sheet" of everything we already know about the product
# being priced. This is the source of truth for "is this field already known?"
# — used instead of trusting the LLM to remember/reason about the
# conversation history on every turn.
#
# Profile keys we track:
#   product_name, product_type, material_cost, quantity   (free-form facts)
#   packaging_quality, work_quality, experience_level,
#   uniqueness, marketplace_platform                      (the 5 "star" fields,
#                                                            value = star int 1-5,
#                                                            or the string "given"
#                                                            if known from free text
#                                                            without an exact star)

def get_profile(session_id: str) -> dict:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT profile FROM session_profile WHERE session_id = %s", (session_id,)
        )
        row = cur.fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row["profile"])
        except Exception:
            return {}
    return {}


def update_profile(session_id: str, updates: dict) -> dict:
    """
    Merge non-empty values into the session's stored profile.
    Returns the resulting full profile.
    """
    if not updates:
        return get_profile(session_id)

    current = get_profile(session_id)
    for k, v in updates.items():
        if v is None or v == "":
            continue
        current[k] = v

    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO session_profile (session_id, profile, updated_at) VALUES (%s, %s, %s) "
            "ON CONFLICT (session_id) DO UPDATE SET profile = EXCLUDED.profile, updated_at = EXCLUDED.updated_at",
            (session_id, json.dumps(current), datetime.utcnow().isoformat())
        )
    conn.commit()
    conn.close()
    return current


def get_answered_fields(session_id: str) -> set:
    """All field names already recorded for this session."""
    return set(get_profile(session_id).keys())