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
MODEL = "qwen/qwen3-32b"


def strip_think(raw: str) -> str:
    """Qwen3 models emit <think>…</think> before their actual reply. Strip it."""
    return re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()

# ── Star rating → value mappings ──────────────────────────────────────────────
# Used to convert frontend star inputs into concrete values for re-analysis
STAR_MAPPINGS = {
    "packaging_quality": {
        1: {"value": 5,   "label": "No packaging (just the item)"},
        2: {"value": 10,  "label": "Basic poly bag or newspaper"},
        3: {"value": 25,  "label": "Kraft paper box or simple wrap"},
        4: {"value": 60,  "label": "Gift box with tissue and ribbon"},
        5: {"value": 120, "label": "Premium branded box with inserts"},
    },
    "work_quality": {
        1: {"value": 80,  "label": "Learning / practice pieces"},
        2: {"value": 120, "label": "Decent, minor imperfections"},
        3: {"value": 150, "label": "Good, market standard"},
        4: {"value": 220, "label": "Polished, professional finish"},
        5: {"value": 350, "label": "Expert, gallery or gift quality"},
    },
    "experience_level": {
        1: {"value": 80,  "label": "Just started (beginner, <1 yr)"},
        2: {"value": 120, "label": "Some experience (1-2 yrs)"},
        3: {"value": 150, "label": "Intermediate (2-4 yrs)"},
        4: {"value": 250, "label": "Experienced (4-8 yrs)"},
        5: {"value": 400, "label": "Expert / professional (8+ yrs)"},
    },
    "uniqueness": {
        1: {"value": 1.0,  "label": "Very common design"},
        2: {"value": 1.1,  "label": "Slight variation on common"},
        3: {"value": 1.2,  "label": "Some unique elements"},
        4: {"value": 1.35, "label": "Distinctive, hard to find"},
        5: {"value": 1.55, "label": "One-of-a-kind / signature work"},
    },
    "marketplace_platform": {
        1: {"value": 0,    "fee_pct": 0,    "label": "Selling offline / local (no fee)"},
        2: {"value": 0,    "fee_pct": 0,    "label": "Instagram / WhatsApp direct (no fee)"},
        3: {"value": 15,   "fee_pct": 15,   "label": "Meesho / Flipkart (~15% fee)"},
        4: {"value": 6.5,  "fee_pct": 6.5,  "label": "Etsy / Amazon (~6.5% fee)"},
        5: {"value": 0,    "fee_pct": 2,    "label": "Own website / boutique (~2% payment fee)"},
    },
}

# Question text + star labels used when WE synthesize a follow-up question
# ourselves (i.e. the profile says a field is missing, regardless of what
# the LLM did or didn't ask). Keep in sync with the wording in BASE_SYSTEM.
BASE_SYSTEM = """You are a pricing mentor for Indian artisans. Reply ONLY with JSON. No expressions, no math, only final computed integers.

Labor rates (hourly): 1-star=80, 2-star=120, 3-star=150(default), 4-star=250, 5-star=400.
Craft time DEFAULTS (use ONLY if user did not state hours): crochet teddy=5hr, clay diya=20min/piece, tote bag=45min, macrame=4hr, embroidery=5hr, resin jewelry=1hr, knit=6hr, candle=1hr, soap=1hr, leather wallet=4hr.

CRITICAL RULES:
- If user states hours explicitly, USE THAT EXACT NUMBER.
- If user states material cost, USE IT AS-IS. Never multiply by quantity.
- The prompt contains a LOCKED VALUES block — treat every value there as immutable. Never re-derive, re-estimate, or change them. They represent confirmed facts from prior turns.
- The prompt also contains USER FACTS lines. Later lines override earlier ones on the same field.
- marketplace_fee = round((materials + labor + packaging + transport) * 0.08)
- waste_allowance = round(materials * 0.10)
- All JSON values must be plain integers. NEVER write expressions.
- "user_planned_price" must be null UNLESS user explicitly states a selling price.
- chat_reply must be SHORT: max 3 sentences, no step-by-step math walkthrough. Never start sentences with "We will also".
- PREFER asking over assuming. If something would meaningfully change the price, ASK — don't guess. Set confidence low (<0.7) whenever you assumed a major cost factor.
- Include follow_up_questions whenever confidence < 0.85 OR a field that would materially affect price is missing.
- Ask up to 3 questions per turn. NEVER repeat a question already answered.

QUESTION HIERARCHY — strictly follow this order. Never ask a Tier N question if any Tier <N question is still unanswered:
  Tier 1 — Core inputs (ask these FIRST, nothing else matters without them):
    * labor_hours: how long does one unit take to make?
    * material_cost: if not already given
  Tier 2 — Product context (ask after Tier 1 is known):
    * Size, dimensions, weight — anything that directly affects labor time
    * Medium, material type, technique (e.g. art style, yarn type, wax type)
    * Product-specific details (e.g. number of colours for block print, kiln-fired for pottery)
  Tier 3 — Quality signals (ask after Tier 2):
    * work_quality, experience_level, uniqueness
  Tier 4 — Fulfillment (ask after Tier 3):
    * packaging_quality
  Tier 5 — Sales channel (ask LAST, only once all other tiers are done):
    * marketplace_platform

Packaging cost by star: 1=5, 2=10, 3=25, 4=60, 5=120.

{rag_context}
{cross_session_context}

Output exactly this JSON:
{{"chat_reply":"2-3 sentences. Show labor math: X hrs x ₹Y/hr = ₹Z. State assumptions. Mention floor price.","product_type":"short name","confidence":0.8,"labor_reasoning":"X hrs x ₹Y/hr = ₹Z","costs":{{"materials":0,"labor":0,"packaging":0,"packaging_note":"assumed or stated","transport":0,"transport_note":"assumed or stated","marketplace_fee":0,"waste_allowance":0}},"floor_price":0,"tiers":{{"basic":{{"price":0,"label":"Basic","note":""}},"standard":{{"price":0,"label":"Standard","note":""}},"premium":{{"price":0,"label":"Premium","note":""}}}},"user_planned_price":null,"missing_info":[],"market_search_query":"2-3 keywords","rag_influence":"one sentence","follow_up_questions":[{{"id":"unique_id","field":"field_name","question":"Short question text","type":"stars|number|choice|text","unit":"rs/hrs/cm/pieces or null","star_labels":["1-star label","2-star label","3-star label","4-star label","5-star label"],"options":["Option A","Option B","Option C"]}}]}}

follow_up_questions rules:
- Ask about ANYTHING genuinely missing that would improve pricing accuracy.
- Be contextual and smart: for a painting ask size and medium; for crochet ask yarn weight; for candles ask wax type and burn time; for jewellery ask metal or material and if handmade vs cast; for pottery ask if kiln-fired; etc.
- Use the right type for the question:
  * "stars" → for quality/experience/uniqueness scales (5 levels). Must include star_labels (5 items). Set options to [].
  * "number" → for measurable inputs like size, hours, quantity, cost. Set star_labels and options to [].
  * "choice" → for selecting from a fixed list of named options (e.g. art style, medium, occasion). Include options array (2-6 strings). Set star_labels to [].
  * "text" → for short open-ended answers where a fixed list doesn't fit. Set star_labels and options to [].
- The 5 standard fields (packaging_quality, work_quality, experience_level, uniqueness, marketplace_platform) are valid choices but NOT mandatory — only include them if they're genuinely missing AND would materially affect the price calculation.
- field name must be a short snake_case string describing what you're asking (e.g. "canvas_size", "art_style", "yarn_type", "occasion").
- If no questions needed, set follow_up_questions to []
- NEVER ask about the same field twice across conversation turns."""

CONVERSATIONAL_SYSTEM = """You are a pricing mentor for Indian artisans.

If purely conversational (explaining a number, asking what something means):
- Reply ONLY with: {{"chat_reply": "your explanation", "is_conversational": true}}

If user provides NEW cost info:
- Reply with the full pricing JSON (same schema, all keys required, follow_up_questions: []).

Labor math: hours × rate. Always show both."""


def build_system(rag_context: str = "", cross_session_context: str = "") -> str:
    rag = f"\n{rag_context}\n" if rag_context else ""
    ctx = f"\n{cross_session_context}\n" if cross_session_context else ""
    return BASE_SYSTEM.format(rag_context=rag, cross_session_context=ctx)


async def call_llm(messages: list, system: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "/no_think\n\n" + system}] + messages,
        temperature=0.1,
        max_tokens=1200,
    )
    return strip_think(response.choices[0].message.content)


def sanitize_json(raw: str) -> str:
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


def resolve_star_answers(answers: dict) -> str:
    """
    Convert star/number answers from the frontend into a plain text description
    that can be appended to the user message for re-analysis.
    e.g. {"packaging_quality": 3, "experience_level": 4}
    → "Packaging quality: 3 stars (Kraft paper box or simple wrap, ₹25).
       Experience level: 4 stars (Experienced 4-8 yrs, ₹250/hr rate)."
    """
    parts = []
    for field, value in answers.items():
        if field in STAR_MAPPINGS and isinstance(value, int):
            mapping = STAR_MAPPINGS[field].get(value, {})
            label = mapping.get("label", "")
            val = mapping.get("value", value)
            if field == "packaging_quality":
                parts.append(f"Packaging quality: {value} stars ({label}, packaging cost ₹{val})")
            elif field == "work_quality":
                parts.append(f"Work quality: {value} stars ({label}, labor rate ₹{val}/hr)")
            elif field == "experience_level":
                parts.append(f"Experience level: {value} stars ({label}, labor rate ₹{val}/hr)")
            elif field == "uniqueness":
                parts.append(f"Uniqueness: {value} stars ({label})")
            elif field == "marketplace_platform":
                fee_pct = mapping.get("fee_pct", 0)
                parts.append(f"Selling platform: {value} stars ({label}, marketplace fee = {fee_pct}% of subtotal)")
            else:
                parts.append(f"{field}: {value} stars ({label})")
        else:
            # Numeric answer
            parts.append(f"{field.replace('_', ' ')}: {value}")
    return ". ".join(parts) + "." if parts else ""




def reconcile_follow_up_questions(llm_questions: list, known_profile: dict) -> list:
    """
    Deduplication-only pass: drop any question whose field is already in the
    session profile (answered in a prior turn). No longer enforces a fixed
    field list — the LLM is free to ask contextual questions for any product.
    Caps at 2 questions.
    """
    if not isinstance(llm_questions, list):
        return []
    answered = set(known_profile.keys())
    kept = [
        q for q in llm_questions
        if isinstance(q, dict) and q.get("field") and q["field"] not in answered
    ]
    return kept[:3]


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


def infer_profile_updates_from_result(result: dict) -> dict:
    """
    After an analysis, pull out values the LLM used (assumed or stated) so
    they get locked into the session profile and never drift between turns.
    This is especially important for labor_hours — if the model assumed 5hrs
    on turn 1, that assumption must survive into turns 2, 3, etc. unchanged
    unless the user explicitly corrects it.
    """
    updates = {}
    if result.get("product_type"):
        updates["product_type"] = result["product_type"]

    costs = result.get("costs", {})

    # Lock labor hours derived from the result (labor ÷ rate).
    # We store it as a string like "5" so it shows up in LOCKED FACTS as plain text.
    labor = costs.get("labor")
    reasoning = result.get("labor_reasoning", "")
    if labor and reasoning:
        # Parse "X hrs x ₹Y/hr" from labor_reasoning
        m = re.match(r"([\d.]+)\s*hrs?", reasoning, re.IGNORECASE)
        if m:
            updates["labor_hours"] = m.group(1)
    elif labor:
        # Fallback: approximate from labor cost (use default 150/hr rate)
        approx_hrs = round(labor / 150, 1)
        if approx_hrs > 0:
            updates["labor_hours"] = str(approx_hrs)

    # Lock material cost
    mat = costs.get("materials")
    if mat:
        updates["material_cost"] = str(mat)

    # Lock packaging if stated
    if str(costs.get("packaging_note", "")).strip().lower() == "stated":
        updates["packaging_quality"] = "given"

    return updates


async def analyze_product(user_message: str, history: list = [],
                          rag_context: str = "", cross_session_context: str = "",
                          known_profile: Optional[dict] = None) -> dict:
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

    # Strip hallucinated planned price
    planned = parsed.get("user_planned_price")
    if planned is not None:
        price_keywords = re.search(
            r'(sell|selling|sold|price|charge|plan|want|going)\s*(at|for|to)?\s*[₹\d]|'
            r'[₹\d]\s*(rs|rupees?|inr)',
            user_message, re.IGNORECASE
        )
        if not price_keywords:
            parsed["user_planned_price"] = None

    # Recalculate in Python
    costs = parsed.get("costs", {})
    numeric_costs = {k: v for k, v in costs.items() if isinstance(v, (int, float))}
    floor = sum(numeric_costs.values())
    parsed["floor_price"] = round(floor)
    tiers = parsed.setdefault("tiers", {})
    tiers.setdefault("basic", {})["price"] = round(floor * 1.13)
    tiers.setdefault("standard", {})["price"] = round(floor * 1.35)
    tiers.setdefault("premium", {})["price"] = round(floor * 1.85)

    # ── Deterministic follow-up question reconciliation ────────────────────
    # Instead of trusting the LLM's own sense of "what's missing", check its
    # proposed questions against the session profile (the actual fact sheet
    # of what this session already knows) and correct accordingly.
    llm_questions = parsed.get("follow_up_questions", [])
    if known_profile is not None:
        parsed["follow_up_questions"] = reconcile_follow_up_questions(llm_questions, known_profile)
    else:
        parsed["follow_up_questions"] = llm_questions[:2] if isinstance(llm_questions, list) else []

    # Always use the deterministic, concise summary for pricing replies.
    # The LLM's own chat_reply is unreliable in length (small model rambles),
    # so we never surface it directly here — just build it from the numbers.
    parsed["chat_reply"] = build_reasoning_reply(parsed)

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