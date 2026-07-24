"""
Connect Buddy — classification service
Insert this file at: backend/services/connect_buddy_service.py

Given a free-text question, decides whether it should surface a human
expert, a certification, a collaboration idea, or a community — and
extracts a few keywords to match against the hardcoded lists in
connect_buddy_data.py. Same AsyncGroq / qwen pattern as business_buddy_service.py.

No RAG, no vector DB — just one LLM call for classification, then plain
Python keyword-overlap scoring against the static lists.
"""

import os
import re
import json
import logging
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("uvicorn.error")

client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "qwen/qwen3.6-27b"  # keep in sync with business_buddy_service.py


def strip_think(raw: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()


SYSTEM_PROMPT = """You are the intent classifier for Connect Buddy, a feature inside an
Indian-artisan marketplace app that connects sellers to human experts, certifications,
collaborators, and local communities. You never answer the user's question yourself —
you only classify it so the app can show the right cards.

Rules:
- "intent" must be exactly one of: "expert", "certification", "collaboration", "community", "none".
  Use "none" if the question doesn't clearly benefit from any of these (e.g. small talk,
  a pure pricing question already handled elsewhere, or something too vague).
- "keywords" is a short list (max 6) of plain English keywords pulled from the question
  that describe the topic, craft, or need (e.g. ["candle", "packaging"], ["export", "customs"]).
- "reason" is ONE short sentence explaining why this intent fits, written for the end user
  (e.g. "Sounds like you could use certification guidance here.").
- Reply ONLY with JSON. No markdown, no commentary.
"""

JSON_SCHEMA_INSTRUCTIONS = """
Return JSON with exactly this shape:

{
  "intent": "expert" | "certification" | "collaboration" | "community" | "none",
  "keywords": string[],
  "reason": string
}
"""


def build_prompt(message: str) -> str:
    return f"User's question: \"{message}\"\n\n{JSON_SCHEMA_INSTRUCTIONS}"


def extract_json(raw: str) -> dict | None:
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
        temperature=0.2,
        max_tokens=400,
        reasoning_effort="none",
        reasoning_format="hidden",
    )
    return strip_think(response.choices[0].message.content)


def _fallback_keywords(message: str) -> list[str]:
    """If the model call fails entirely, just use the raw words as keywords
    so the router still has something to match against."""
    words = re.findall(r"[a-zA-Z]{3,}", message.lower())
    stop = {"the", "and", "for", "with", "you", "your", "are", "how", "what", "can",
            "need", "want", "help", "about", "this", "that", "any", "get", "sell", "selling"}
    return [w for w in words if w not in stop][:6]


async def classify_query(message: str) -> dict:
    """Main entry point used by the router. Never raises — falls back to a
    plain keyword-based guess if the model call or parsing fails, so the
    feature degrades gracefully rather than breaking chat."""
    try:
        raw = await call_llm(build_prompt(message))
        parsed = extract_json(raw)
        if parsed and parsed.get("intent") in {"expert", "certification", "collaboration", "community", "none"}:
            parsed.setdefault("keywords", [])
            parsed.setdefault("reason", "")
            return parsed
        logger.warning(f"Connect Buddy: unparseable classifier output: {raw[:200]!r}")
    except Exception as e:
        logger.warning(f"Connect Buddy: classifier call failed: {e}")

    return {"intent": "none", "keywords": _fallback_keywords(message), "reason": ""}