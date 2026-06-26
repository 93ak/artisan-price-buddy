from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import re
import logging
import traceback
from services.llm_service import (
    call_llm, extract_json, CONVERSATIONAL_SYSTEM,
    analyze_product, build_reasoning_reply
)
from rag.rag_service import retrieve_similar, format_for_prompt
from db.database import (
    save_message, get_session_messages, save_analysis,
    update_session_meta, build_cross_session_context
)

logger = logging.getLogger("uvicorn.error")
router = APIRouter()

# Pure question patterns — these and ONLY these get conversational replies
PURE_QUESTION = re.compile(
    r'^(why|what|how|where|which|who|when|explain|tell me|can you explain|'
    r'what does|what is|what are|what was|how does|how did|how do)'
    r'[^.!]*\?\s*$',
    re.IGNORECASE
)

def is_cost_update(message: str) -> bool:
    """
    Default to re-analysis UNLESS message is clearly a pure question with no new info.
    Better to re-analyse too often than miss a cost update.
    """
    msg = message.strip()
    # If it ends in ? and starts with a question word and has no numbers/cost words → conversational
    if PURE_QUESTION.match(msg):
        # But if it contains numbers or currency, still re-analyse
        if not re.search(r'[₹\d]|rs\.?|rupees?|hours?|hrs?|mins?', msg, re.IGNORECASE):
            return False
    return True

def build_combined_description(history: list, new_message: str) -> str:
    """
    Flatten conversation history into a single description for re-analysis.
    Picks only user messages to avoid LLM hallucination contamination.
    """
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    user_msgs.append(new_message)
    return " | ".join(user_msgs)

async def run_reanalysis(session_id: str, history: list, new_message: str,
                         cross_ctx: str) -> dict:
    """Full re-analysis using all user info gathered so far in the session."""
    combined = build_combined_description(history, new_message)

    # RAG retrieval on combined description
    rag_context = ""
    try:
        retrieved = await retrieve_similar(combined, n_results=5)
        if retrieved:
            rag_context = format_for_prompt(retrieved)
    except Exception as e:
        logger.warning(f"RAG failed in reanalysis (non-fatal): {e}")

    result = await analyze_product(
        combined,
        rag_context=rag_context,
        cross_session_context=cross_ctx
    )

    if "error" in result:
        return None

    # Recalculate in Python
    costs = result.get("costs", {})
    floor = sum(v for v in costs.values() if isinstance(v, (int, float)))
    result["floor_price"] = round(floor)
    tiers = result.setdefault("tiers", {})
    tiers.setdefault("basic", {})["price"] = round(floor * 1.13)
    tiers.setdefault("standard", {})["price"] = round(floor * 1.35)
    tiers.setdefault("premium", {})["price"] = round(floor * 1.85)

    # Persist updated analysis
    try:
        save_analysis(session_id, result)
        update_session_meta(
            session_id,
            result.get("product_type", ""),
            result.get("chat_reply", "")[:120]
        )
    except Exception as e:
        logger.warning(f"DB save failed (non-fatal): {e}")

    return result

async def run_conversational(message: str, history: list, cross_ctx: str) -> str:
    """Handle a pure question/explanation without touching pricing."""
    recent = history[-6:]
    messages = recent + [{"role": "user", "content": message}]
    system = CONVERSATIONAL_SYSTEM + (f"\n\n{cross_ctx}" if cross_ctx else "")
    raw = await call_llm(messages, system=system)

    parsed = extract_json(raw)
    if parsed:
        reply = parsed.get("chat_reply", "")
        if reply:
            return reply.strip()

    # Strip raw JSON if it leaked
    clean = raw.strip()
    if clean.startswith("{"):
        m = re.search(r'"chat_reply"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
        return m.group(1).replace("\\n", "\n") if m else "Could you rephrase that?"
    return clean


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


@router.post("")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    prior = get_session_messages(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in prior]

    save_message(session_id, "user", req.message)

    cross_ctx = ""
    try:
        cross_ctx = build_cross_session_context(limit=2)
    except Exception:
        pass

    try:
        if is_cost_update(req.message):
            # User gave new cost info → full re-analysis with everything we know
            logger.info(f"Cost update detected in: '{req.message[:60]}' — running reanalysis")
            result = await run_reanalysis(session_id, history, req.message, cross_ctx)

            if result:
                reply = result.get("chat_reply", "I've updated your pricing.")
                save_message(session_id, "assistant", reply)
                return {
                    "session_id": session_id,
                    "reply": reply,
                    "type": "reanalysis",
                    "data": result
                }
            # Reanalysis failed — fall through to conversational

        # Pure question/explanation
        reply = await run_conversational(req.message, history, cross_ctx)
        save_message(session_id, "assistant", reply)
        return {"session_id": session_id, "reply": reply, "type": "conversational"}

    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Chat failed: {str(e)}")