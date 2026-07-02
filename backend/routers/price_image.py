"""
routers/price_image.py

POST /price-image-analyze
Accepts a product photo and returns a quality assessment + product hints
specifically for feeding into the Price Buddy pricing flow.
Unlike design-analyze, this does NOT critique photography or listing quality —
only craft quality (mapped to 1-5 stars) and visible product details.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from services.qwen_service import analyze_image_for_pricing

router = APIRouter()
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/price-image-analyze")
async def price_image_analyze(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Please upload a JPG, PNG, or WEBP image.")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded file was empty.")
    try:
        return await analyze_image_for_pricing(image_bytes, content_type=image.content_type)
    except Exception as exc:
        raise HTTPException(502, f"Image analysis failed: {exc}")