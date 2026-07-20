"""
Business Buddy — router
Insert this file at: backend/routers/business_buddy.py

Then in backend/main.py add (matching your market_index convention):

    from routers import business_buddy
    app.include_router(business_buddy.router, prefix="/business-buddy", tags=["business-buddy"])

Frontend calls: POST /business-buddy/analyze
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.business_buddy_service import get_business_insights

router = APIRouter()


class AnalyzeTextRequest(BaseModel):
    product_name: str
    category: Optional[str] = None
    materials: Optional[str] = None
    style: Optional[str] = None
    keywords: Optional[str] = None


@router.post("/analyze")
async def analyze_text(payload: AnalyzeTextRequest):
    """Text-only path: user typed a product name (optionally with extra details)."""
    try:
        insights = await get_business_insights(
            product_name=payload.product_name,
            category=payload.category or "",
            materials=payload.materials or "",
            style=payload.style or "",
            keywords=payload.keywords or "",
        )
        return insights
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not generate insights: {e}")


@router.post("/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    """
    Image path: run Qwen Vision first to get product_name/category/materials/
    style/keywords, then feed that into get_business_insights.

    TODO: wire this to whatever Qwen Vision helper Design Buddy already uses
    (looks like it may live under backend/services or backend/rag — reuse
    that identification function here instead of duplicating it).
    """
    try:
        # Placeholder — replace with your existing vision identification call.
        # identified = qwen_vision_identify(await image.read())
        raise HTTPException(
            status_code=501,
            detail="Wire this endpoint to your existing Qwen Vision identify() function.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {e}")