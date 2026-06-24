from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import logging
import traceback
from services.ollama_service import analyze_product, build_system, call_ollama, extract_json
from rag.rag_service import retrieve_similar, format_for_prompt
from db.database import save_message, get_session_messages, save_analysis, update_session_meta

logger = logging.getLogger("uvicorn.error")
router = APIRouter()

CONVERSATIONAL_SYSTEM = """You are a pricing mentor for Indian artisans. The user is asking a follow-up question about their pricing analysis.

If the question is purely conversational (e.g. "why is labor X?", "explain this", "what does floor price mean?"):
- Reply ONLY with JSON: {{"chat_reply": "your explanation here", "is_conversational": true}}

If the user provides new product information that changes the pricing:
- Reply with the full pricing JSON (same schema as before).

Labor math reminder: labor = hours × rate. 10 hours × ₹250/hr = ₹2500. Never use any other formula.
Always confirm the exact hours and rate you are using in your reply."""

async def handle_conversational(message: str, history: list) -> dict:
    """Handle follow-up questions that don't need full re-analysis."""
    messages = history[-6:] + [{"role": "user", "content": message}]  # last 3 exchanges
    raw = await call_ollama(messages, system=CONVERSATIONAL_SYSTEM)
    parsed = extract_json(raw)
    if parsed and parsed.get("is_conversational"):
        return {"conversational": True, "reply": parsed.get("chat_reply", raw)}
    # Returned full pricing JSON — treat as re-analysis
    return {"conversational": False, "data": parsed, "raw": raw}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@router.post("")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    prior = get_session_messages(session_id)
    history = [{"role": m["role"], "content": m["content"]} for m in prior]

    save_message(session_id, "user", req.message)

    try:
        result = await handle_conversational(req.message, history)
    except Exception as e:
        logger.error(f"Chat error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Chat failed: {str(e)}")

    if result["conversational"]:
        reply = result["reply"]
        save_message(session_id, "assistant", reply)
        return {"session_id": session_id, "reply": reply, "type": "conversational"}

    # Got updated pricing — return full analysis data
    data = result.get("data")
    if data and "costs" in data:
        costs = data.get("costs", {})
        floor = sum(v for v in costs.values() if isinstance(v, (int, float)))
        data["floor_price"] = round(floor)
        tiers = data.setdefault("tiers", {})
        tiers.setdefault("basic", {})["price"] = round(floor * 1.13)
        tiers.setdefault("standard", {})["price"] = round(floor * 1.35)
        tiers.setdefault("premium", {})["price"] = round(floor * 1.85)
        reply = data.get("chat_reply", "I've updated the analysis.")
        save_message(session_id, "assistant", reply)
        return {"session_id": session_id, "reply": reply, "type": "reanalysis", "data": data}

    # Fallback
    reply = result.get("raw", "I couldn't process that. Could you rephrase?")[:300]
    save_message(session_id, "assistant", reply)
    return {"session_id": session_id, "reply": reply, "type": "fallback"}