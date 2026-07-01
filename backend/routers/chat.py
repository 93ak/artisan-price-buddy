from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import re
import logging
import traceback
from services.llm_service import (
    call_llm, extract_json, CONVERSATIONAL_SYSTEM,
    analyze_product, resolve_star_answers
)
from rag.rag_service import retrieve_similar, format_for_prompt
from db.database import (
    save_message, get_session_messages, save_analysis,
    update_session_meta, build_cross_session_context,
    get_profile, update_profile
)

logger = logging.getLogger("uvicorn.error")
router = APIRouter()

PURE_QUESTION = re.compile(
    r'^(why|what|how|where|which|who|when|explain|tell me|can you explain|'
    r'what does|what is|what are|what was|how does|how did|how do)'
    r'[^.!]*\?\s*$',
    re.IGNORECASE
)

def is_cost_update(message: str) -> bool:
    msg = message.strip()
    if PURE_QUESTION.match(msg):
        if not re.search(r'[₹\d]|rs\.?|rupees?|hours?|hrs?|mins?', msg, re.IGNORECASE):
            return False
    return True

def build_combined_description(history: list, extra_messages: list[str]) -> str:
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    user_msgs.extend(extra_messages)
    return " | ".join(user_msgs)

async def run_reanalysis(session_id: str, history: list,
                         extra_messages: list[str], cross_ctx: str,
                         known_profile: dict) -> dict:
    combined = build_combined_description(history, extra_messages)

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
        cross_session_context=cross_ctx,
        known_profile=known_profile  # ← pass profile so labor/fees use correct values
    )

    if "error" in result:
        return None

    # Floor is already recalculated inside analyze_product, but recalc tiers too
    floor = result.get("floor_price", 0)
    tiers = result.setdefault("tiers", {})
    tiers["basic"]    = {"price": round(floor * 1.13), "label": "Basic",    "note": "Covers costs + 13% margin"}
    tiers["standard"] = {"price": round(floor * 1.35), "label": "Standard", "note": "Healthy margin, market competitive"}
    tiers["premium"]  = {"price": round(floor * 1.85), "label": "Premium",  "note": "For boutique / gift buyers"}

    try:
        save_analysis(session_id, result)
        update_session_meta(session_id, result.get("product_type", ""), result.get("chat_reply", "")[:120])
    except Exception as e:
        logger.warning(f"DB save failed (non-fatal): {e}")

    return result

async def run_conversational(message: str, history: list, cross_ctx: str) -> str:
    recent = history[-6:]
    messages = recent + [{"role": "user", "content": message}]
    system = CONVERSATIONAL_SYSTEM + (f"\n\n{cross_ctx}" if cross_ctx else "")
    raw = await call_llm(messages, system=system)
    parsed = extract_json(raw)
    if parsed:
        reply = parsed.get("chat_reply", "")
        if reply:
            return reply.strip()
    clean = raw.strip()
    if clean.startswith("{"):
        m = re.search(r'"chat_reply"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
        return m.group(1).replace("\\n", "\n") if m else "Could you rephrase that?"
    return clean


class ChatRequest(BaseModel):
    message: Optional[str] = ""
    session_id: Optional[str] = None
    question_answers: Optional[Dict[str, Any]] = None
    user_summary: Optional[str] = None


@router.post("")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    prior = get_session_messages(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in prior]

    cross_ctx = ""
    try:
        cross_ctx = build_cross_session_context(limit=2)
    except Exception:
        pass

    try:
        # ── Structured question answers ────────────────────────────────────
        if req.question_answers:
            # Save ALL answers to profile immediately — this is what prevents re-asking
            profile_updates = {k: v for k, v in req.question_answers.items() if v is not None}
            profile = update_profile(session_id, profile_updates)
            logger.info(f"Profile updated with: {profile_updates} → {profile}")

            answer_text = resolve_star_answers(req.question_answers)
            full_message = f"{req.message.strip()} {answer_text}".strip() if req.message else answer_text
            display_message = req.user_summary or full_message

            save_message(session_id, "user", display_message)

            result = await run_reanalysis(session_id, history, [full_message], cross_ctx, profile)
            if result:
                reply = result.get("chat_reply", "I've updated your pricing.")
                save_message(session_id, "assistant", reply)
                return {
                    "session_id": session_id,
                    "reply": reply,
                    "type": "reanalysis",
                    "data": result,
                    "follow_up_questions": result.get("follow_up_questions", [])
                }

            fail_reply = "I got your answers but had trouble recalculating. Please try again."
            save_message(session_id, "assistant", fail_reply)
            return {"session_id": session_id, "reply": fail_reply, "type": "conversational", "follow_up_questions": []}

        # ── Regular text message ───────────────────────────────────────────
        msg = req.message.strip() if req.message else ""
        if not msg:
            return {"session_id": session_id, "reply": "Please describe your product.", "type": "conversational", "follow_up_questions": []}

        save_message(session_id, "user", msg)
        profile = get_profile(session_id)

        if is_cost_update(msg):
            logger.info(f"Cost update: '{msg[:60]}' — reanalysing")
            result = await run_reanalysis(session_id, history, [msg], cross_ctx, profile)
            if result:
                reply = result.get("chat_reply", "I've updated your pricing.")
                save_message(session_id, "assistant", reply)
                return {
                    "session_id": session_id,
                    "reply": reply,
                    "type": "reanalysis",
                    "data": result,
                    "follow_up_questions": result.get("follow_up_questions", [])
                }

        reply = await run_conversational(msg, history, cross_ctx)
        save_message(session_id, "assistant", reply)
        return {"session_id": session_id, "reply": reply, "type": "conversational", "follow_up_questions": []}

    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Chat failed: {str(e)}")