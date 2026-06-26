import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent.parent / "price_buddy.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            product_type TEXT,
            summary TEXT
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            product_type TEXT,
            costs TEXT,
            floor_price REAL,
            tiers TEXT,
            confidence REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS market_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE NOT NULL,
            result TEXT NOT NULL,
            cached_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
        (session_id, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_session_messages(session_id: str):
    conn = get_conn()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_all_sessions():
    conn = get_conn()
    rows = conn.execute(
        "SELECT s.id, s.created_at, s.product_type, s.summary, COUNT(m.id) as msg_count "
        "FROM sessions s LEFT JOIN messages m ON s.id = m.session_id "
        "GROUP BY s.id ORDER BY s.created_at DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def update_session_meta(session_id: str, product_type: str, summary: str):
    conn = get_conn()
    conn.execute(
        "UPDATE sessions SET product_type = ?, summary = ? WHERE id = ?",
        (product_type, summary, session_id)
    )
    conn.commit()
    conn.close()

def save_analysis(session_id: str, data: dict):
    conn = get_conn()
    conn.execute(
        "INSERT INTO analyses (session_id, product_type, costs, floor_price, tiers, confidence, created_at) VALUES (?,?,?,?,?,?,?)",
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
    row = conn.execute(
        "SELECT result, cached_at FROM market_cache WHERE query = ?", (query,)
    ).fetchone()
    conn.close()
    if row:
        from datetime import timedelta
        cached_at = datetime.fromisoformat(row["cached_at"])
        if datetime.utcnow() - cached_at < timedelta(hours=6):
            return json.loads(row["result"])
    return None

def set_market_cache(query: str, result: dict):
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO market_cache (query, result, cached_at) VALUES (?, ?, ?)",
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
    rows = conn.execute(
        "SELECT product_type, costs, floor_price, tiers, confidence, created_at "
        "FROM analyses ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
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
