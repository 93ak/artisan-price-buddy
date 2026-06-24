from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, market, chat, history
from db.database import init_db
import logging

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Price Buddy API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    # Index RAG dataset on startup (skips if already indexed)
    try:
        from rag.rag_service import index_dataset
        result = await index_dataset()
        logger.info(f"RAG startup: {result}")
    except Exception as e:
        logger.warning(f"RAG indexing failed on startup (non-fatal): {e}")

app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
app.include_router(market.router, prefix="/market-research", tags=["market"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(history.router, prefix="/history", tags=["history"])

@app.post("/rag/index")
async def reindex():
    """Force re-index the RAG dataset."""
    from rag.rag_service import index_dataset
    return await index_dataset(force=True)

@app.get("/rag/status")
async def rag_status():
    from rag.rag_service import get_index_status
    return await get_index_status()

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.1.0"}
