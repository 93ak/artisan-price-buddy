from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import analyze, market, chat, history, design, price_image, market_index
from db.database import init_db
import logging

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Price Buddy API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
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
app.include_router(design.router, tags=["design"])
app.include_router(price_image.router, tags=["price-image"])
app.include_router(market_index.router, prefix="/market-index", tags=["market-index"])
from routers import business_buddy
app.include_router(business_buddy.router, prefix="/business-buddy", tags=["business-buddy"])
from routers import connect_buddy
app.include_router(connect_buddy.router, prefix="/connect", tags=["connect-buddy"])

@app.post("/rag/index")
async def reindex():
    from rag.rag_service import index_dataset
    return await index_dataset(force=True)

@app.get("/rag/status")
async def rag_status():
    from rag.rag_service import get_index_status
    return await get_index_status()

@app.get("/health")
async def health():
    return {"status": "ok", "version": "3.0.0"}