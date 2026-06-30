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
    update_session_meta, build_cross_session_context
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
    """
    Flatten all user messages from history + any extra new messages into a
    FACTS list. Later facts are listed last and are explicitly marked as
    overriding any earlier conflicting fact, since the LLM otherwise tends
    to fall back to its own defaults instead of using the latest answer.
    """
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    user_msgs.extend(extra_messages)
    numbered = "\n".join(f"- {m}" for m in user_msgs if m and m.strip())
    return (
        "FACTS (each line is a fact already given by the user; later lines "
        "override earlier ones on the same field; treat ALL of them as final, "
        "do not re-guess any value covered below):\n" + numbered
    )

async def run_reanalysis(session_id: str, history: list,
                         extra_messages: list[str], cross_ctx: str) -> dict:
    """Full re-analysis. extra_messages are new info not yet in history."""
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
        cross_session_context=cross_ctx
    )

    if "error" in result:
        return None

    costs = result.get("costs", {})
    floor = sum(v for v in costs.values() if isinstance(v, (int, float)))
    result["floor_price"] = round(floor)
    tiers = result.setdefault("tiers", {})
    tiers.setdefault("basic", {})["price"] = round(floor * 1.13)
    tiers.setdefault("standard", {})["price"] = round(floor * 1.35)
    tiers.setdefault("premium", {})["price"] = round(floor * 1.85)

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
    user_summary: Optional[str] = None  # human-readable summary of answers, for saving to chat history


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
            answer_text = resolve_star_answers(req.question_answers)
            full_message = f"{req.message.strip()} {answer_text}".strip() if req.message else answer_text

            # Save human-readable summary to chat history (what the user sees)
            # Falls back to technical text if no summary provided
            display_message = req.user_summary or full_message
            logger.info(f"Question answers: {req.question_answers} → '{answer_text}'")
            save_message(session_id, "user", display_message)

            # Pass answer_text as extra_message — NOT yet in history
            result = await run_reanalysis(session_id, history, [full_message], cross_ctx)

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

        # ── Regular text message ───────────────────────────────────────────
        msg = req.message.strip() if req.message else ""
        if not msg:
            return {"session_id": session_id, "reply": "Please describe your product.", "type": "conversational", "follow_up_questions": []}

        save_message(session_id, "user", msg)

        if is_cost_update(msg):
            logger.info(f"Cost update: '{msg[:60]}' — reanalysing")
            result = await run_reanalysis(session_id, history, [msg], cross_ctx)
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