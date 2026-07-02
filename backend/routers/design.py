"""
routers/design.py

POST /design-analyze
Accepts a single product photo, runs it through Qwen (services/qwen_service.py),
returns structured feedback for the Design Buddy frontend.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException

from services.qwen_service import analyze_product_image

router = APIRouter()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/design-analyze")
async def design_analyze(image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Please upload a JPG, PNG, or WEBP image.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    try:
        feedback = await analyze_product_image(image_bytes, content_type=image.content_type)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Design Buddy couldn't inspect this photo: {exc}")

    return feedback