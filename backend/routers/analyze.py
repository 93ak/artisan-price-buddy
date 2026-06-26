from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import traceback
import logging
from services.llm_service import analyze_product, check_llm_health
from services.scraper import research_market
from rag.rag_service import retrieve_similar, format_for_prompt
from db.database import save_message, save_analysis, update_session_meta, build_cross_session_context

logger = logging.getLogger("uvicorn.error")
router = APIRouter()

class AnalyzeRequest(BaseModel):
    description: str
    session_id: Optional[str] = None

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

@router.post("", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    try:
        if not await check_llm_health():
            raise HTTPException(503, "Groq API is unreachable. Check your GROQ_API_KEY.")

        session_id = req.session_id or str(uuid.uuid4())
        save_message(session_id, "user", req.description)

        # ── Step 1: RAG retrieval ──────────────────────────────────────────
        retrieved = []
        rag_context = ""
        try:
            retrieved = await retrieve_similar(req.description, n_results=5)
            if retrieved:
                rag_context = format_for_prompt(retrieved)
        except Exception as e:
            logger.warning(f"RAG retrieval failed (non-fatal): {e}")

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
                cross_session_context=cross_ctx
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
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled /analyze error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Unexpected error: {str(e)}")
