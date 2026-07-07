import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import traceback
import logging
from services.llm_service import analyze_product, check_llm_health, infer_profile_updates_from_result
from services.scraper import research_market
from rag.rag_service import retrieve_similar, format_for_prompt
from market_rag.retrieval import get_market_references, format_market_reference_block
from db.database import (
    save_message, save_analysis, update_session_meta, build_cross_session_context,
    get_profile, update_profile
)

logger = logging.getLogger("uvicorn.error")
router = APIRouter()

class AnalyzeRequest(BaseModel):
    description: str
    session_id: Optional[str] = None
    # Structured facts from the initial form (frontend/pricebuddy.html).
    # These get written straight into the session profile so the LLM never
    # has to be trusted to "remember" them from the prose description.
    product_name: Optional[str] = None
    material_cost: Optional[str] = None
    experience: Optional[str] = None
    quantity: Optional[str] = None
    image_quality_stars: Optional[int] = None  # work_quality from photo analysis

class AnalyzeResponse(BaseModel):
    session_id: str
    chat_reply: str
    product_type: str
    confidence: float
    labor_reasoning: Optional[str] = None
    costs: dict
    floor_price: float
    tiers: dict
    market: Optional[dict] = None
    user_planned_price: Optional[float] = None
    missing_info: list = []
    retrieved_products: list = []
    rag_influence: Optional[str] = None
    follow_up_questions: list = []   # ← new

@router.post("", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        if not await check_llm_health():
            raise HTTPException(503, "Groq API is unreachable. Check your GROQ_API_KEY.")

        session_id = req.session_id or str(uuid.uuid4())
        save_message(session_id, "user", req.description)

        # ── Step 0: seed the session profile from structured form fields ───
        # experience_level is marked "given" (rather than a star number) since
        # the form collects free text, not a 1-5 rating — this just stops the
        # follow-up logic from re-asking about experience.
        seed_updates = {}
        if req.product_name:
            seed_updates["product_name"] = req.product_name
        if req.material_cost:
            seed_updates["material_cost"] = req.material_cost
        if req.quantity:
            seed_updates["quantity"] = req.quantity
        if req.experience:
            seed_updates["labor_hours"] = req.experience
        if req.image_quality_stars:
            seed_updates["work_quality"] = req.image_quality_stars
        profile = update_profile(session_id, seed_updates)

        # ── Step 1: RAG retrieval (existing pricing-history RAG — unchanged) ─
        retrieved = []
        rag_context = ""
        try:
            retrieved = await retrieve_similar(req.description, n_results=5)
            if retrieved:
                rag_context = format_for_prompt(retrieved)
        except Exception as e:
            logger.warning(f"RAG retrieval failed (non-fatal): {e}")

        # ── Step 1b: Market reference retrieval (NEW, backend-only) ────────
        # Pulls up to 5 comparable marketplace listings from market_index to
        # ground the LLM's pricing in real market data. Never forced to a
        # fixed count — 0, 1, or 5 results are all valid. Folded straight
        # into rag_context so llm_service.py needs no changes at all: it
        # already treats rag_context as one opaque block of prompt text.
        # Never surfaced in AnalyzeResponse — purely internal to the prompt.
        try:
            market_refs = await asyncio.to_thread(get_market_references, req.description)
            if market_refs:
                logger.info(
                    f"Market references for session {session_id} ({len(market_refs)} found): " +
                    "; ".join(
                        f"\"{r.get('title','')}\" ₹{r.get('price',0):.0f} "
                        f"(score={r.get('_score',0):.2f})"
                        for r in market_refs
                    )
                )
            else:
                logger.info(f"Market references for session {session_id}: none found above relevance threshold")

            market_block = format_market_reference_block(market_refs)
            if market_block:
                rag_context = f"{rag_context}\n\n{market_block}" if rag_context else market_block
        except Exception as e:
            logger.warning(f"Market reference retrieval failed (non-fatal): {e}")

        # ── Step 2: Cross-session context ─────────────────────────────────
        cross_ctx = ""
        try:
            cross_ctx = build_cross_session_context(limit=3)
        except Exception as e:
            logger.warning(f"Cross-session context failed (non-fatal): {e}")

        # ── Step 3: LLM analysis ──────────────────────────────────────────
        try:
            result = await analyze_product(
                req.description,
                rag_context=rag_context,
                cross_session_context=cross_ctx,
                known_profile=profile
            )
        except Exception as e:
            logger.error(f"analyze_product failed: {e}\n{traceback.format_exc()}")
            raise HTTPException(500, f"LLM call failed: {str(e)}")

        if "error" in result:
            logger.error(f"Parse error. Raw: {result.get('raw', '')[:300]}")
            raise HTTPException(500, result.get("chat_reply", "AI analysis failed"))

        required = ["costs", "tiers", "floor_price", "chat_reply", "product_type"]
        missing = [k for k in required if k not in result]
        if missing:
            raise HTTPException(500, f"AI response incomplete, missing: {missing}")

        # ── Step 4: Market research ───────────────────────────────────────
        market_data = {"market_min": None, "market_avg": None, "market_max": None,
                       "sample_count": 0, "note": "Market research skipped"}
        try:
            query = result.get("market_search_query") or result.get("product_type") or "handmade product India"
            market_data = await research_market(query)
        except Exception as e:
            logger.warning(f"Market research failed (non-fatal): {e}")
            market_data["note"] = str(e)

        # ── Step 5: Persist ───────────────────────────────────────────────
        try:
            save_analysis(session_id, result)
            update_session_meta(session_id, result.get("product_type", ""), result.get("chat_reply", "")[:120])
            save_message(session_id, "assistant", result.get("chat_reply", ""))
            # Sync profile with anything the LLM confirmed (product_type, etc.)
            update_profile(session_id, infer_profile_updates_from_result(result))
        except Exception as e:
            logger.warning(f"DB save failed (non-fatal): {e}")

        return AnalyzeResponse(
            session_id=session_id,
            chat_reply=result.get("chat_reply", ""),
            product_type=result.get("product_type", "unknown"),
            confidence=float(result.get("confidence", 0.5)),
            labor_reasoning=result.get("labor_reasoning"),
            costs=result.get("costs", {}),
            floor_price=float(result.get("floor_price", 0)),
            tiers=result.get("tiers", {}),
            market=market_data,
            user_planned_price=result.get("user_planned_price"),
            missing_info=result.get("missing_info", []),
            retrieved_products=retrieved,
            rag_influence=result.get("rag_influence"),
            follow_up_questions=result.get("follow_up_questions", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled /analyze error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Unexpected error: {str(e)}")