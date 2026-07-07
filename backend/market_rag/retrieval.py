"""
retrieval.py

Retrieval layer for the market-reference RAG. This is the ONE module the
existing chatbot/pricing workflow should import — everything else
(scraping, enrichment, embedding) stays internal to market_rag.

Usage from your existing analyze/chat flow, right before you build the
final prompt to the pricing LLM:

    from market_rag.retrieval import get_market_reference_block

    market_block = get_market_reference_block(user_description)
    # market_block is "" if nothing relevant was found — safe to always
    # concatenate, no branching needed on the caller's side.

    final_prompt = existing_prompt + market_block
"""

from market_rag.market_index_service import search_market, index_exists

# Below this cosine-similarity score, a "match" isn't the same kind of
# product — showing it as a price reference would be misleading rather
# than useful, so it gets dropped rather than forced into the top 5.
MIN_RELEVANCE_SCORE = 0.35
MAX_REFERENCES = 5


def get_market_references(user_description: str) -> list[dict]:
    """
    Returns 0-5 comparable products from market_index, ranked by relevance.
    Never pads or forces a fixed count — a thin or empty market for this
    product type is a valid, meaningful outcome, not an error to work around.

    Fails safe: any problem here (index not built yet, Chroma hiccup, etc.)
    returns an empty list rather than raising, so a retrieval issue never
    takes down the main pricing flow.
    """
    if not user_description or not user_description.strip():
        return []

    if not index_exists():
        return []

    try:
        results = search_market(user_description, top_k=MAX_REFERENCES)
    except Exception as e:
        print(f"  [warn] market reference retrieval failed, continuing without it: {e}")
        return []

    return [r for r in results if r.get("_score", 0) >= MIN_RELEVANCE_SCORE]


def format_market_reference_block(references: list[dict]) -> str:
    """
    Turns retrieved references into the exact text block to inject into the
    pricing prompt. Returns "" when there's nothing worth showing — callers
    should just skip inserting anything in that case (or, more simply,
    always concatenate the result, since "" is a no-op).
    """
    if not references:
        return ""

    lines = [
        "Market reference (for context only — these are NOT the user's product "
        "and must not be copied or matched directly):",
        "The following are comparable products currently listed on a marketplace. "
        "Use them only to understand the general market range for this kind of "
        "product. Still independently estimate a fair price for the user's product "
        "based on their own materials, time, effort, and description above — do not "
        "simply average or copy these prices, especially if only one or two are shown.",
        "",
    ]
    for i, ref in enumerate(references, start=1):
        materials = ", ".join(ref.get("materials", [])) or "not specified"
        price = ref.get("price", 0) or 0
        lines.append(
            f"{i}. \"{ref.get('title', '')}\" — category: {ref.get('category', '')}, "
            f"price: \u20b9{price:.0f}, materials: {materials}"
        )

    return "\n".join(lines)


def get_market_reference_block(user_description: str) -> str:
    """
    Convenience one-shot: retrieve + format in a single call. This is the
    function most callers want — returns "" (safe to always concatenate)
    when there's no usable market context for this product.
    """
    references = get_market_references(user_description)
    return format_market_reference_block(references)