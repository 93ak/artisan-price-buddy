"""
dataset_builder.py

Turns raw Yuukke product dicts + LLM-extracted fields into the clean,
unified dataset shape used to build market_index:

{
  "id": "", "title": "", "description": "", "price": 0.0, "seller": "",
  "category": "", "materials": [], "keywords": [], "source": "Yuukke Marketplace"
}

Resumable: if data/market/market_dataset.json already exists, products
already present (by id) are skipped rather than re-sent to the LLM.
Progress is checkpointed to disk every batch, so a crash partway through
~1500 products doesn't cost you the whole run.
"""

import asyncio
import json
from pathlib import Path

import httpx

from market_rag.market_llm_service import extract_market_fields

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "market"
DATASET_PATH = DATA_DIR / "market_dataset.json"

CONCURRENCY = 2   # simultaneous in-flight LLM calls — Groq free-tier limits are easy to hit;
                  # raise this only if you're not seeing 429s in the output
BATCH_SIZE = 25   # checkpoint to disk after every batch


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_existing() -> dict:
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {r["id"]: r for r in records}
    return {}


def _save(records: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


async def _process_one(client: httpx.AsyncClient, product: dict, semaphore: asyncio.Semaphore) -> dict | None:
    async with semaphore:
        extracted = await extract_market_fields(client, product)
    if extracted is None:
        return None  # exhausted retries — leave unprocessed so a later run retries it
    return {
        "id": str(product.get("id")),
        "title": product.get("name", ""),
        "description": product.get("details", ""),
        "price": _safe_float(product.get("price")),
        "seller": product.get("w_name", ""),
        "category": extracted["category"],
        "materials": extracted["materials"],
        "keywords": extracted["keywords"],
        "source": "Yuukke Marketplace",
    }


async def build_clean_dataset(raw_products: list[dict]) -> list[dict]:
    existing = _load_existing()
    todo = [p for p in raw_products if str(p.get("id")) not in existing]

    print(f"{len(existing)} products already processed, {len(todo)} left to enrich via LLM")

    semaphore = asyncio.Semaphore(CONCURRENCY)
    results = list(existing.values())

    async with httpx.AsyncClient() as client:
        for i in range(0, len(todo), BATCH_SIZE):
            batch = todo[i:i + BATCH_SIZE]
            enriched = await asyncio.gather(*[
                _process_one(client, p, semaphore) for p in batch
            ])
            succeeded = [r for r in enriched if r is not None]
            failed_count = len(enriched) - len(succeeded)

            results.extend(succeeded)
            _save(results)

            status = f"  enriched {min(i + BATCH_SIZE, len(todo))}/{len(todo)}"
            if failed_count:
                status += f"  ({failed_count} failed this batch — will retry on next run)"
            print(status)

    print(f"Clean dataset ready: {len(results)} products -> {DATASET_PATH}")
    return results