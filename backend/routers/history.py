from fastapi import APIRouter
from db.database import get_all_sessions, get_session_messages

router = APIRouter()

@router.get("")
async def list_sessions():
    return get_all_sessions()

@router.get("/{session_id}")
async def get_session(session_id: str):
    return get_session_messages(session_id)
