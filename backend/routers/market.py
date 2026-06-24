from fastapi import APIRouter
from pydantic import BaseModel
from services.scraper import research_market

router = APIRouter()

class MarketRequest(BaseModel):
    query: str

@router.post("")
async def market_research(req: MarketRequest):
    result = await research_market(req.query)
    return result
