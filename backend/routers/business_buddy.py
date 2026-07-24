"""
Business Buddy — router
Insert this file at: backend/routers/business_buddy.py

Then in backend/main.py add (matching your market_index convention):

    from routers import business_buddy
    app.include_router(business_buddy.router, prefix="/business-buddy", tags=["business-buddy"])

Frontend calls: POST /business-buddy/analyze, POST /business-buddy/analyze-image
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional

from services.business_buddy_service import get_business_insights
from services.llm_service import identify_product_from_image  # adjust to match your actual module path/name

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
    Image path: run Qwen Vision (via llmservice.identify_product_from_image)
    to get product_name/category/materials/style/keywords from the photo,
    then feed that into get_business_insights for the market analysis.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        identified = await identify_product_from_image(image_bytes, image.content_type)

        insights = await get_business_insights(
            product_name=identified.get("product_name", ""),
            category=identified.get("category", ""),
            materials=identified.get("materials", ""),
            style=identified.get("style", ""),
            keywords=identified.get("keywords", ""),
        )
        # Let the frontend show what was identified from the photo, if useful.
        insights["identified_from_image"] = identified
        return insights
    except HTTPException:
        raise
    except ValueError as e:
        # Raised by identify_product_from_image when the vision response
        # can't be parsed, or by get_business_insights on a bad LLM response.
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {e}")