"""
market_llm_service.py

Standalone LLM call used ONLY by the market-reference pipeline, to extract
category / materials / keywords from a Yuukke product listing.

Deliberately separate from services/llm_service.py — this is a bulk batch
job over ~1500 products with its own prompt, its own JSON contract, and
its own failure-tolerance needs (one bad extraction shouldn't kill the run).

Uses Groq + Qwen 3.6 27B in text-only, non-thinking, JSON mode — same
account already wired up for Design Buddy. Swap this file out if you'd
rather point the market pipeline at a different model/provider.
"""

import asyncio
import json
import os
import re
import html as html_lib
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "qwen/qwen3.6-27b"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

TAG_RE = re.compile(r"<[^>]+>")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """You are a product cataloguer for a handmade/artisan marketplace.
Given a product's name, description, and seller, extract structured metadata.

Respond with ONLY a single valid JSON object, no markdown fences, no commentary:

{
  "category": "string, 1-3 words, the single best product category",
  "materials": ["string", "..."],
  "keywords": ["string", "..."]
}

Rules:
- "materials": 0-5 items. Only materials you can reasonably infer from the text
  (e.g. "wood", "cotton", "millet flour"). Empty array if none are stated or implied.
- "keywords": 3-6 short search-relevant terms (product type, style, use-case), lowercase.
- "category" should be a general retail category, not the product name repeated.
"""


def _clean_details(raw_html: str) -> str:
    if not raw_html:
        return ""
    text = html_lib.unescape(raw_html)
    text = TAG_RE.sub("", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_OBJECT_RE.search(text)
        if match:
            return json.loads(match.group(0))
        raise


MAX_RETRIES = 5


async def extract_market_fields(client: httpx.AsyncClient, product: dict) -> dict | None:
    """
    product: one raw item from the Yuukke /getProducts response.
    Returns {"category": str, "materials": [str], "keywords": [str]} on success,
    or None if every retry was exhausted (caller should NOT treat this product
    as processed — it needs to be retried on a later run).
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    name = product.get("name", "")
    details = _clean_details(product.get("details", ""))
    seller = product.get("w_name", "")

    user_prompt = f"Name: {name}\nSeller: {seller}\nDescription: {details}"

    payload = {
        "model": MODEL_NAME,
        "reasoning_format": "hidden",
        "reasoning_effort": "none",
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(GROQ_URL, headers=headers, json=payload, timeout=60.0)

            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 2 * (attempt + 1)))
                print(f"  [rate limit] product {product.get('id')}: waiting {wait:.1f}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})")
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            return {
                "category": parsed.get("category", "Uncategorized"),
                "materials": parsed.get("materials", []),
                "keywords": parsed.get("keywords", []),
            }

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                await asyncio.sleep(2 * (attempt + 1))
                continue
            print(f"  [warn] product {product.get('id')} failed with {e.response.status_code}: {e}")
            return None
        except Exception as e:
            print(f"  [warn] product {product.get('id')} extraction error: {e}")
            return None

    print(f"  [warn] product {product.get('id')} exhausted all retries — will retry on next run")
    return None