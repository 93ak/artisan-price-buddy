import os
import json
import re
import logging
from typing import Optional
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("uvicorn.error")

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.1-8b-instant"  # swap to "llama3-70b-8192" for better quality

BASE_SYSTEM = """You are a pricing mentor for Indian artisans. Reply ONLY with JSON. No expressions, no math, only final computed integers.

Labor rates: beginner=80/hr, intermediate=150/hr(default), experienced=250/hr, expert=400/hr.
Craft time DEFAULTS (use ONLY if user did not state hours): crochet teddy=5hr, clay diya=20min/piece, tote bag=45min, macrame=4hr, embroidery=5hr, resin jewelry=1hr, knit=6hr, candle=1hr, soap=1hr, leather wallet=4hr.

CRITICAL RULES — read carefully:
- If user states hours explicitly ("10 hours", "took me 3 hrs"), USE THAT EXACT NUMBER.
- If user states material cost, USE IT AS-IS. Never multiply by quantity.
- marketplace_fee = round((materials + labor + packaging + transport) * 0.08)
- waste_allowance = round(materials * 0.10)
- All JSON values must be plain integers. NEVER write expressions like "250*10".

Packaging inference (from user description):
- No packaging mentioned → ask in missing_info, use 0
- Poly bag / basic wrap → 8
- Kraft paper / simple box → 20
- Gift box with tissue/ribbon → 55
- Premium branded box → 110
- Custom described packaging → estimate from description

Transport inference:
- Local / hand delivery → 25
- City courier → 60
- Pan-India courier (default if unspecified) → 90
- International → 250

{rag_context}
{cross_session_context}

Output exactly this JSON with all keys present:
{{"chat_reply":"3-4 sentences. Show labor math explicitly: X hrs × ₹Y/hr = ₹Z. State packaging and transport assumptions. Mention floor price. Warn if planned price is below floor.","product_type":"short name","confidence":0.8,"labor_reasoning":"X hrs (stated/inferred) × ₹Y/hr (skill level) = ₹Z labor cost","costs":{{"materials":0,"labor":0,"packaging":0,"packaging_note":"what you assumed","transport":0,"transport_note":"what you assumed","marketplace_fee":0,"waste_allowance":0}},"floor_price":0,"tiers":{{"basic":{{"price":0,"label":"Basic","note":""}},"standard":{{"price":0,"label":"Standard","note":""}},"premium":{{"price":0,"label":"Premium","note":""}}}},"user_planned_price":null,"missing_info":["specific things that would improve accuracy"],"market_search_query":"2-3 keywords","rag_influence":"one sentence"}}"""

CONVERSATIONAL_SYSTEM = """You are a pricing mentor for Indian artisans. The user is asking a follow-up about their pricing.

If purely conversational (explaining a number, asking what something means, clarifying):
- Reply ONLY with: {{"chat_reply": "your explanation", "is_conversational": true}}

If the user provides NEW information that changes costs (new packaging, actual hours, material cost update):
- Reply with the full pricing JSON (same schema as before, all keys required).

Labor math: labor = hours × rate. Always state both numbers in your reply.
Packaging/transport: if user now describes them, infer the cost and include it."""


def build_system(rag_context: str = "", cross_session_context: str = "") -> str:
    rag = f"\n{rag_context}\n" if rag_context else ""
    ctx = f"\n{cross_session_context}\n" if cross_session_context else ""
    return BASE_SYSTEM.format(rag_context=rag, cross_session_context=ctx)


async def call_llm(messages: list, system: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        temperature=0.1,
        max_tokens=800,
    )
    return response.choices[0].message.content


def sanitize_json(raw: str) -> str:
    """Evaluate any arithmetic expressions that slipped into JSON values."""
    def eval_expr(m):
        try:
            val = eval(m.group(1), {"__builtins__": {}})
            return f": {round(val)}"
        except Exception:
            return m.group(0)
    return re.sub(r':\s*([\d\s\*\+\-\/\.]+(?:\d))', eval_expr, raw)


def extract_json(raw: str) -> Optional[dict]:
    raw = re.sub(r"```json|```", "", raw).strip()
    raw = sanitize_json(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(sanitize_json(match.group()))
        except json.JSONDecodeError:
            pass
    return None


def build_reasoning_reply(parsed: dict) -> str:
    costs = parsed.get("costs", {})
    floor = parsed.get("floor_price", 0)
    product = parsed.get("product_type", "your product")
    labor_reasoning = parsed.get("labor_reasoning", "")
    conf = int(parsed.get("confidence", 0.5) * 100)
    planned = parsed.get("user_planned_price")
    standard_price = parsed.get("tiers", {}).get("standard", {}).get("price", 0)

    lines = [f"I've analyzed your {product} with {conf}% confidence."]
    if labor_reasoning:
        lines.append(f"Labor: {labor_reasoning}.")

    pkg_note = costs.get("packaging_note", "")
    trn_note = costs.get("transport_note", "")
    pkg = costs.get("packaging", 0)
    trn = costs.get("transport", 0)
    mat = costs.get("materials", 0)
    lab = costs.get("labor", 0)
    fee = costs.get("marketplace_fee", 0)

    lines.append(
        f"Costs: ₹{mat} materials + ₹{lab} labor + ₹{pkg} packaging"
        f"{' (' + pkg_note + ')' if pkg_note else ''} + "
        f"₹{trn} transport{' (' + trn_note + ')' if trn_note else ''} + "
        f"₹{fee} marketplace fee = floor price ₹{floor}."
    )

    if planned:
        diff = planned - floor
        if diff < 0:
            lines.append(
                f"⚠️ Your planned price ₹{planned} is ₹{abs(diff)} below floor — "
                f"you'd lose money. Minimum recommended: ₹{standard_price}."
            )
        else:
            lines.append(f"Your planned price ₹{planned} is ₹{diff} above floor — healthy margin.")

    return " ".join(lines)


async def analyze_product(user_message: str, history: list = [],
                          rag_context: str = "", cross_session_context: str = "") -> dict:
    system = build_system(rag_context, cross_session_context)
    messages = history + [{"role": "user", "content": user_message}]
    raw = await call_llm(messages, system)
    parsed = extract_json(raw)

    if not parsed:
        return {
            "error": "Could not parse AI response",
            "raw": raw[:500],
            "chat_reply": "I had trouble processing that. Could you tell me: what you make, material cost, and how long it takes?"
        }

    # Always recalculate in Python — never trust LLM math
    costs = parsed.get("costs", {})
    numeric_costs = {k: v for k, v in costs.items() if isinstance(v, (int, float))}
    floor = sum(numeric_costs.values())
    parsed["floor_price"] = round(floor)
    tiers = parsed.setdefault("tiers", {})
    tiers.setdefault("basic", {})["price"] = round(floor * 1.13)
    tiers.setdefault("standard", {})["price"] = round(floor * 1.35)
    tiers.setdefault("premium", {})["price"] = round(floor * 1.85)

    # Enrich chat_reply
    llm_reply = parsed.get("chat_reply", "")
    python_reply = build_reasoning_reply(parsed)
    if len(llm_reply) < 100:
        parsed["chat_reply"] = python_reply
    elif "floor price" not in llm_reply.lower():
        parsed["chat_reply"] = llm_reply + " " + python_reply

    return parsed


async def check_llm_health() -> bool:
    try:
        await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        return True
    except Exception:
        return False
