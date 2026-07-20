"""
Business Buddy — service layer
Insert this file at: backend/services/business_buddy_service.py

Mirrors the same Groq/Qwen calling pattern already used in price_buddy's
LLM service (AsyncGroq client, qwen/qwen3.6-27b, reasoning suppressed at
the API level) instead of raw requests + a guessed model id.
"""

import os
import json
import re
import logging
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("uvicorn.error")

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "qwen/qwen3.6-27b"  # same snapshot price_buddy uses — keep in sync


def strip_think(raw: str) -> str:
    """Safety net — reasoning_format='hidden' should already suppress this."""
    return re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()


SYSTEM_PROMPT = """You are Business Buddy, a market-intelligence assistant for Indian artisans
and small handmade-goods sellers. You are NOT a pricing tool and NOT a listing tool —
you reason using general knowledge only, never RAG, never web search, never invented
statistics.

Context: this platform is built for the Indian market. Always reason and answer in
that context.

Rules:
- Use qualitative scales only: "High" / "Medium" / "Low" for demand/competition/difficulty.
- Use words like "Popular", "Growing", "Seasonal", "Niche" instead of made-up numbers.
- Never fabricate specific sales figures, exact market sizes, or citations.
- ALL currency must be in Indian Rupees (₹), never dollars or any other currency.
  "suggested_price_range" must be a qualitative band that includes a ₹ figure or range,
  e.g. "Mid-range, ₹400-₹700" or "Budget, under ₹250" — never a currency-less label.
- "best_selling_seasons" should reflect Indian buying patterns and festivals where
  relevant (e.g. Diwali, wedding season, Raksha Bandhan, monsoon, New Year gifting)
  rather than generic Western seasons.
- "target_audience" and "marketing_ideas" should reflect Indian buyer context
  (e.g. Indian marketplaces, gifting culture, regional platforms) where relevant.
- Reply ONLY with JSON. No markdown, no code fences, no commentary before or after.
"""

JSON_SCHEMA_INSTRUCTIONS = """
Return JSON with exactly this shape. Keep it lean — one or two items max
anywhere a list is allowed, and one sentence max for any text field.

{
  "product_name": string,
  "product_category": string,
  "market_summary": string,               // ONE sentence, plain language
  "demand_level": "High" | "Medium" | "Low",
  "competition_level": "High" | "Medium" | "Low",
  "crafting_difficulty": "High" | "Medium" | "Low",
  "estimated_making_time": string,        // e.g. "2-4 hours"
  "best_selling_season": string,          // ONE season/period, e.g. "Winter Holidays"
  "target_audience": string[],            // MAX 2 short phrases
  "suggested_price_range": string,   // qualitative band WITH a ₹ figure, e.g. "Mid-range, ₹400-₹700"        // qualitative band, e.g. "Mid-range"
  "products_to_try_next": string[],       // MAX 2 short product ideas
  "top_tip": string,                      // ONE sentence, the single best piece of advice
  "top_marketing_idea": string,           // ONE sentence
  "top_risk": string,                     // ONE sentence, the single biggest risk
  "learn_more": { "term": string, "explanation": string }  // ONE business concept, one sentence
}
"""


def build_prompt(product_name: str, category: str = "", materials: str = "",
                  style: str = "", keywords: str = "") -> str:
    details = f"Product: {product_name}\n"
    if category:
        details += f"Category: {category}\n"
    if materials:
        details += f"Materials: {materials}\n"
    if style:
        details += f"Style: {style}\n"
    if keywords:
        details += f"Keywords: {keywords}\n"

    return (
        f"Analyze this handmade/artisan product for market intelligence purposes.\n\n"
        f"{details}\n"
        f"{JSON_SCHEMA_INSTRUCTIONS}"
    )


def extract_json(raw: str) -> dict | None:
    """
    Balanced-brace extraction: finds the first complete {...} object and
    ignores anything before/after it (handles stray prose, fences, or
    trailing text some Qwen responses tack on).
    """
    cleaned = re.sub(r"```json|```", "", raw).strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    end = None
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end is None:
        return None

    try:
        return json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None


async def call_llm(prompt: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=2600,
        reasoning_effort="none",   # ask it not to reason much at all
        reasoning_format="hidden", # strip any reasoning it does produce
    )
    return strip_think(response.choices[0].message.content)


async def get_business_insights(product_name: str, category: str = "", materials: str = "",
                                 style: str = "", keywords: str = "") -> dict:
    """Main entry point used by the router. Raises ValueError on bad model output."""
    prompt = build_prompt(product_name, category, materials, style, keywords)
    raw = await call_llm(prompt)

    parsed = extract_json(raw)
    if parsed is None:
        snippet = raw[:300].replace("\n", " ")
        logger.error(f"Business Buddy: could not parse model output: {snippet!r}")
        raise ValueError(f"Model did not return parseable JSON. Raw output started with: {snippet!r}")

    return parsed