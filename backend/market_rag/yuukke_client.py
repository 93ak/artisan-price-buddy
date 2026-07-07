"""
yuukke_client.py

Pulls every page of products from the Yuukke Marketplace API and saves
the raw, unmodified response items to disk.

API contract (as observed):
  POST https://shop.yuukke.com/api/getProducts
  body: {"filters": {"page": "<n>", "limit": "12", "offset": <n>}}
  resp: {"products": [...], "info": {"total_record", "page", "total_page"}}
"""

import asyncio
import json
import os
from pathlib import Path

import httpx

YUUKKE_URL = "https://shop.yuukke.com/api/getProducts"
PAGE_LIMIT = 12
MAX_RETRIES = 3

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market"
RAW_OUTPUT_PATH = DATA_DIR / "raw_products.json"

# The Yuukke site issues a short-lived JWT (~1hr, per its own exp claim) as a
# Bearer token for this endpoint. Grab a fresh one from DevTools -> Network ->
# a getProducts call -> Request Headers -> "authorization", and set it as an
# env var before running the pipeline:
#   PowerShell:  $env:YUUKKE_BEARER_TOKEN="eyJ0eXAi..."
#   bash/zsh:    export YUUKKE_BEARER_TOKEN="eyJ0eXAi..."
YUUKKE_BEARER_TOKEN = os.environ.get("YUUKKE_BEARER_TOKEN")


def _headers() -> dict:
    if not YUUKKE_BEARER_TOKEN:
        raise RuntimeError(
            "YUUKKE_BEARER_TOKEN is not set. Grab a fresh bearer token from DevTools "
            "(Network tab -> a getProducts request -> Request Headers -> authorization) "
            "and set it as an env var — it expires roughly every hour, so grab a new one "
            "each time you run this."
        )
    return {"Authorization": f"Bearer {YUUKKE_BEARER_TOKEN}"}


async def _fetch_page(client: httpx.AsyncClient, page: int) -> dict:
    payload = {
        "filters": {
            "page": str(page),
            "limit": str(PAGE_LIMIT),
            "offset": (page - 1) * PAGE_LIMIT,
        }
    }
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(YUUKKE_URL, json=payload, headers=_headers(), timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, ValueError) as e:
            last_error = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch page {page} after {MAX_RETRIES} attempts: {last_error}")


async def fetch_all_products() -> list[dict]:
    """Walks every page until total_page is exhausted, returns the full flat list."""
    products: list[dict] = []
    page = 1
    total_pages = None

    async with httpx.AsyncClient() as client:
        while total_pages is None or page <= total_pages:
            data = await _fetch_page(client, page)
            page_products = data.get("products", [])
            products.extend(page_products)

            info = data.get("info", {})
            total_pages = int(info.get("total_page", page))

            print(f"  fetched page {page}/{total_pages} "
                  f"({len(page_products)} items, {len(products)} total so far)")
            page += 1

    return products


def save_raw(products: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(products)} raw products -> {RAW_OUTPUT_PATH}")


async def fetch_and_save_all() -> list[dict]:
    products = await fetch_all_products()
    save_raw(products)
    return products