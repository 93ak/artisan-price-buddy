"""not being used, use llm_service connected to groq"""
import httpx
import json
import re
from typing import Optional

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.1:8b"

BASE_SYSTEM = """You are a pricing mentor for Indian artisans. Reply ONLY with JSON. No expressions, no math, only final computed integers.

Labor rates: beginner=80/hr, intermediate=150/hr(default), experienced=250/hr, expert=400/hr.
Craft time DEFAULTS (use ONLY if user did not state hours): crochet teddy=5hr, clay diya=20min/piece, tote bag=45min, macrame=4hr, embroidery=5hr, resin jewelry=1hr, knit=6hr, candle=1hr, soap=1hr, leather wallet=4hr.

CRITICAL RULES:
- If the user states hours explicitly (e.g. "10 hours", "took me 3 hrs"), USE THAT NUMBER. Never override it with a default.
- If the user states material cost explicitly, USE THAT NUMBER as-is. Never multiply it.
- marketplace_fee = round((materials+labor+packaging+transport) * 0.08)
- waste_allowance = round(materials * 0.10)
- All JSON values must be plain integers. NEVER write expressions like "250*10".

{rag_context}

Output exactly this JSON:
{{"chat_reply":"3-4 sentences. Confirm what hours and rate you used for labor. Show the math: X hrs × ₹Y/hr = ₹Z. Mention floor price. Warn clearly if user's planned price is below floor.","product_type":"name","confidence":0.8,"labor_reasoning":"X hrs (user-stated or inferred) × ₹Y/hr (skill level) = ₹Z","costs":{{"materials":0,"labor":0,"packaging":15,"transport":10,"marketplace_fee":0,"waste_allowance":0}},"floor_price":0,"tiers":{{"basic":{{"price":0,"label":"Basic","note":""}},"standard":{{"price":0,"label":"Standard","note":""}},"premium":{{"price":0,"label":"Premium","note":""}}}},"user_planned_price":null,"missing_info":[],"market_search_query":"2-3 keywords","rag_influence":"one sentence"}}"""


def build_system(rag_context: str = "") -> str:
    ctx = f"\n{rag_context}\n" if rag_context else ""
    return BASE_SYSTEM.format(rag_context=ctx)


def sanitize_json(raw: str) -> str:
    """Evaluate simple arithmetic expressions inside JSON values before parsing."""
    # Match patterns like "key": 120 * 10 or "key": 120*10
    def eval_expr(m):
        try:
            val = eval(m.group(1), {"__builtins__": {}})
            return f": {round(val)}"
        except Exception:
            return m.group(0)
    return re.sub(r':\s*([\d\s\*\+\-\/\.]+(?:\d))', eval_expr, raw)


async def call_ollama(messages: list, system: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 700}
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            chunks = []
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        chunks.append(token)
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
            return "".join(chunks)


def extract_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"```json|```", "", raw).strip()
    raw = sanitize_json(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Find outermost JSON object
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(sanitize_json(match.group()))
        except json.JSONDecodeError:
            pass
    return None


def build_reasoning_reply(parsed: dict) -> str:
    """
    Build a rich reasoning reply in Python from the structured data.
    Used to enrich or replace a weak LLM chat_reply.
    """
    costs = parsed.get("costs", {})
    floor = parsed.get("floor_price", 0)
    product = parsed.get("product_type", "your product")
    labor_reasoning = parsed.get("labor_reasoning", "")
    conf = int(parsed.get("confidence", 0.5) * 100)
    planned = parsed.get("user_planned_price")
    standard_price = parsed.get("tiers", {}).get("standard", {}).get("price", 0)

    lines = []

    # Confidence + product
    lines.append(f"I've analyzed your {product} with {conf}% confidence.")

    # Labor reasoning
    if labor_reasoning:
        lines.append(f"For labor: {labor_reasoning}.")

    # Cost breakdown summary
    mat = costs.get("materials", 0)
    lab = costs.get("labor", 0)
    pkg = costs.get("packaging", 0)
    fee = costs.get("marketplace_fee", 0)
    lines.append(
        f"Your costs break down as: ₹{mat} materials + ₹{lab} labor + ₹{pkg} packaging + ₹{fee} marketplace fee, "
        f"giving a floor price of ₹{floor} — the minimum you must charge to break even."
    )

    # Planned price warning
    if planned:
        diff = planned - floor
        if diff < 0:
            lines.append(
                f"⚠️ You mentioned selling at ₹{planned}, but that's ₹{abs(diff)} below your floor price — "
                f"you would lose money on every sale. I recommend at least ₹{standard_price}."
            )
        else:
            lines.append(
                f"Your planned price of ₹{planned} is ₹{diff} above floor price — that's a healthy margin."
            )

    return " ".join(lines)


async def analyze_product(user_message: str, history: list = [], rag_context: str = "") -> dict:
    system = build_system(rag_context)
    messages = history + [{"role": "user", "content": user_message}]
    raw = await call_ollama(messages, system)
    parsed = extract_json(raw)

    if not parsed:
        return {
            "error": "Could not parse AI response",
            "raw": raw[:500],
            "chat_reply": "I had trouble processing that. Could you tell me: what you make, material cost, and time taken?"
        }

    # Always recalculate in Python — never trust LLM math
    costs = parsed.get("costs", {})
    floor = sum(v for v in costs.values() if isinstance(v, (int, float)))
    parsed["floor_price"] = round(floor)
    tiers = parsed.setdefault("tiers", {})
    tiers.setdefault("basic", {})["price"] = round(floor * 1.13)
    tiers.setdefault("standard", {})["price"] = round(floor * 1.35)
    tiers.setdefault("premium", {})["price"] = round(floor * 1.85)

    # Enrich chat_reply with Python-built reasoning
    # If LLM reply is short (< 100 chars), replace it entirely; otherwise append labor reasoning
    llm_reply = parsed.get("chat_reply", "")
    python_reply = build_reasoning_reply(parsed)
    if len(llm_reply) < 100:
        parsed["chat_reply"] = python_reply
    else:
        # LLM gave a good reply — append the breakdown summary if not already there
        if "floor price" not in llm_reply.lower():
            parsed["chat_reply"] = llm_reply + " " + python_reply

    return parsed


async def check_ollama_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False
