"""
routers/market_index.py

Admin/trigger endpoints for the market-reference pipeline (Yuukke fetch ->
LLM enrich -> market_index build). Separate from routers/market.py, which
handles your existing /market-research flow — no overlap, no shared state.
"""

from fastapi import APIRouter, BackgroundTasks

from market_rag.build_market_index import run as run_market_pipeline
from market_rag.market_index_service import search_market, index_exists

router = APIRouter()

_pipeline_status = {"running": False, "last_error": None}


async def _run_pipeline_task():
    _pipeline_status["running"] = True
    _pipeline_status["last_error"] = None
    try:
        await run_market_pipeline()
    except Exception as e:
        _pipeline_status["last_error"] = str(e)
    finally:
        _pipeline_status["running"] = False


@router.post("/build")
async def build(background_tasks: BackgroundTasks):
    """Kicks off fetch -> enrich -> index as a background task. Takes a while
    for ~1500 products (LLM calls are the bottleneck) — poll /status."""
    if _pipeline_status["running"]:
        return {"status": "already_running"}
    background_tasks.add_task(_run_pipeline_task)
    return {"status": "started"}


@router.get("/status")
async def status():
    return {
        "running": _pipeline_status["running"],
        "last_error": _pipeline_status["last_error"],
        "index_exists": index_exists(),
    }


@router.get("/search")
async def search(q: str, top_k: int = 5):
    """Quick manual sanity-check endpoint — query market_index directly."""
    return {"results": search_market(q, top_k)}