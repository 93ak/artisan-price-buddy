"""
qwen_service.py

Calls Qwen 3.6 27B on Groq to inspect a product photo and return
structured feedback for Design Buddy.

Groq is OpenAI-compatible: https://api.groq.com/openai/v1/chat/completions
Two Groq-specific features do most of the hard work for us:
  - reasoning_format="hidden"  -> Groq strips the <think>...</think> block
    server-side, so `content` only ever contains the final answer.
  - response_format={"type": "json_object"} -> Groq's JSON mode guarantees
    the response is a syntactically valid JSON object.

We still keep a defensive strip/parse fallback in case a future model
update reintroduces stray reasoning text or markdown fences.
"""

import base64
import json
import os
import re
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "qwen/qwen3.6-27b"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

THINK_BLOCK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = """You are Design Buddy, a quality inspector for a handmade goods marketplace.
You will be shown one product photo. Assess it like a careful craftsperson would before
it goes live on a listing.
Judge the photo as if it were being reviewed for the homepage of a premium handmade marketplace. Prioritize aesthetic quality and the product's visual appeal over basic technical correctness.
Evaluate the photography from the perspective of a premium handmade marketplace.

Focus primarily on:
- Lighting quality (soft, harsh, directional, overexposed, shadows, glare)
- Product visibility (are important details easy to see?)
- Camera angle and composition (top-down, eye-level, close-up, perspective)
- Styling and presentation (props, background, color harmony, premium aesthetic)
- Focus and sharpness
- Cropping and framing

Do NOT default to mentioning background clutter unless it genuinely distracts from the product.
Treat intentional props and styled backgrounds as positives when they enhance the presentation.

Every photography issue should explain WHY it hurts the presentation and suggest a specific improvement.

Respond with ONLY a single valid JSON object — no markdown fences, no commentary,
nothing before or after it. Match this exact shape:

{
  "audience": ["string", "string", "..."],
  "craftsmanship": {
    "good": ["short phrase describing something done well"],
    "issues": ["short phrase describing a specific, fixable flaw"]
  },
  "photography": {
    "good": ["short phrase describing something the photo does well"],
    "issues": ["short phrase describing a specific, fixable photo problem"]
  },
  "sellingPoints": ["short phrase a seller could use in a listing"]
}

Rules:
- 2-4 items in "tags", "audience", and "sellingPoints".
- "good" and "issues" arrays: 1-3 items each. If there is nothing wrong, "issues" can be empty.
- Every "issues" item must be something the seller can actually act on (this doubles as
  the improvement suggestion for that section — do not write a separate list of fixes).
- Check if the photography is worthy of being listed in the site. do not hyperfixate on clutter, it might be props. see if the lighting is profesional and studio - worthy, if the clarity is good, and the aesthetic matches the product.
- Keep every string under 12 words.
- Base everything only on what is visible in the photo.
"""


def _strip_thinking(raw: str) -> str:
    """Safety net in case reasoning_format=hidden ever lets a <think> block slip through."""
    return THINK_BLOCK_RE.sub("", raw).strip()


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = JSON_OBJECT_RE.search(text)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Model response did not contain parseable JSON")


def _with_defaults(data: dict) -> dict:
    data.setdefault("category", {})
    data["category"].setdefault("primary", "Uncategorized")
    data["category"].setdefault("tags", [])

    data.setdefault("audience", [])

    data.setdefault("craftsmanship", {})
    data["craftsmanship"].setdefault("good", [])
    data["craftsmanship"].setdefault("issues", [])

    data.setdefault("photography", {})
    data["photography"].setdefault("good", [])
    data["photography"].setdefault("issues", [])

    data.setdefault("sellingPoints", [])
    return data


def _to_data_uri(image_bytes: bytes, content_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


async def analyze_product_image(image_bytes: bytes, content_type: str = "image/jpeg") -> dict:
    """
    Send the image to Qwen 3.6 27B via Groq and return the parsed,
    defaulted feedback dict ready to hand straight to the frontend.
    """
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    image_data_uri = _to_data_uri(image_bytes, content_type)

    payload = {
        "model": MODEL_NAME,
        "reasoning_format": "hidden",   # drop the <think> block server-side
        "reasoning_effort": "none",     # non-thinking mode: this is a quick visual read, not deep math/coding
        "response_format": {"type": "json_object"},  # force valid JSON back
        "temperature": 0.4,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this product photo and return the JSON."},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(GROQ_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

    raw_content = result["choices"][0]["message"]["content"]
    cleaned = _strip_thinking(raw_content)
    parsed = _extract_json(cleaned)
    return _with_defaults(parsed)